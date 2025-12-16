#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-consistency TIR inference script using LMDeploy.

Refactored from self_consistency_strategy.py to use LMDeploy pipeline (offline engine)
instead of vLLM server-client architecture.
"""

from __future__ import annotations

# ============================================================
# Standard Library Imports
# ============================================================

import contextlib
import logging
import math
import os
import queue
import re
import subprocess
import threading
import time
import random
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# Third-Party Imports
# ============================================================

import numpy as np
import pandas as pd
import polars as pl
import torch
from transformers import set_seed

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
    RenderConversationConfig,
    ToolNamespaceConfig,
    Author,
    TextContent,
)

from lmdeploy import pipeline as lm_pipeline, TurbomindEngineConfig, GenerationConfig as LMDeployGenerationConfig

import kaggle_evaluation.aimo_3_inference_server as aimo_server


# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Environment Setup
# ============================================================

ENV_VARS = {
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_FLAX": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "CUDA_VISIBLE_DEVICES": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TIKTOKEN_ENCODINGS_BASE": "/kaggle/usr/lib/lmdeploy_package/tiktoken_encodings",
}

for _k, _v in ENV_VARS.items():
    os.environ.setdefault(_k, _v)

SAVE_DIR = Path("/kaggle/working/saved_responses")
SAVE_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class LMDeployConfig:
    model_path: str
    gpu_indices: Optional[List[int]] = None
    max_batch_size: int = 12
    enable_prefix_caching: bool = True
    cache_max_entry_count: float = 0.96
    max_prefill_token_num: int = 4096
    session_len: int = 64 * 1024 + 4096 + 128
    dtype: str = "auto"


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.02
    seed: int = 42
    sample_count: int = 6
    max_iter: int = 100
    high_budget_samples: int = 6
    mid_budget_samples: int = 6
    low_budget_samples: int = 6
    high_budget_seconds: float = 300.0
    low_budget_seconds: float = 180.0
    majority_threshold: int = 4
    token_limit: int = 60000
    context_early_window: int = 100
    context_recent_window: int = 100
    max_new_tokens: int = 64 * 1024
    skip_special_tokens: bool = False
    do_sample: bool = True


@dataclass
class ToolConfig:
    local_jupyter_timeout: float = 60.0


@dataclass
class TimingConfig:
    total_hours: float = 4 + 55 / 60
    checkpoints: int = 50
    base_budget_seconds: float = 180.0


@dataclass
class DatasetConfig:
    reference_csv: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    submission_parquet: str = "submission.parquet"
    use_server_for_eval: bool = True


@dataclass
class PromptTemplate:
    system: str
    user_suffix: str
    number: int


@dataclass
class AppConfig:
    lmdeploy: LMDeployConfig
    generation: GenerationConfig
    tool: ToolConfig
    timing: TimingConfig
    dataset: DatasetConfig
    prompt_list: List[PromptTemplate]


# ============================================================
# Concrete Configuration Instance
# ============================================================

CONFIG = AppConfig(
    lmdeploy=LMDeployConfig(
        model_path="/kaggle/input/gpt-oss-120b/transformers/default/1",
        gpu_indices=[0],
        max_batch_size=12,
        session_len=64 * 1024,
        enable_prefix_caching=True,
    ),
    generation=GenerationConfig(
        sample_count=8,
        high_budget_samples=12,
        mid_budget_samples=10,
        low_budget_samples=8,
        high_budget_seconds=480,
        low_budget_seconds=300,
        majority_threshold=4,
        token_limit=60*1024,
        context_early_window=100,
        context_recent_window=100,
        max_new_tokens=60*1024,
        skip_special_tokens=False,
    ),
    tool=ToolConfig(
        local_jupyter_timeout=60,
    ),
    timing=TimingConfig(
        total_hours=4 + 55 / 60,
        checkpoints=50,
        base_budget_seconds=4.4*60,
    ),
    dataset=DatasetConfig(
        use_server_for_eval=True,
    ),
    prompt_list=[
        PromptTemplate(
            system="",
            user_suffix=(
                "Please reason step by step and use the python tool to solve the math problem."
                "\nFinally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess."
            ),
            number=1,
        )
    ],
)


# ============================================================
# Time Management (Cutoffs)
# ============================================================

class TimeManager:
    """Dynamic time budgeting with setup time accounted for."""

    def __init__(self, total_hours: float, steps: int, base_budget_seconds: float) -> None:
        self.config_total_seconds = total_hours * 3600
        self.steps = steps
        self.base_budget_seconds = base_budget_seconds
        self.setup_seconds: float = 0.0
        self.usable_seconds: float = self.config_total_seconds
        self.start_time: float = time.time()
        self.final_cutoff_time: float = self.start_time + self.usable_seconds
        self.questions_served: int = 0
        self._cutoffs: List[int] = []
        self.reset_after_setup(0.0)

    def reset_after_setup(self, setup_seconds: float) -> None:
        """Recompute budgets after setup to subtract loading time."""
        self.setup_seconds = max(0.0, setup_seconds)
        self.usable_seconds = max(0.0, self.config_total_seconds - self.setup_seconds)
        self.start_time = time.time()
        self.final_cutoff_time = self.start_time + self.usable_seconds
        cutoff_array = np.linspace(
            self.final_cutoff_time,
            self.start_time,
            self.steps + 1,
        )
        cutoff_list = [int(x) for x in cutoff_array]
        cutoff_list.pop()  # drop earliest
        self._cutoffs = cutoff_list
        self.questions_served = 0

    def after_final_cutoff(self) -> bool:
        return time.time() > self.final_cutoff_time

    def should_downweight(self) -> bool:
        if not self._cutoffs:
            return True
        return time.time() > self._cutoffs[-1]

    def consume_iteration(self) -> None:
        if self._cutoffs:
            self._cutoffs.pop()
        self.questions_served += 1

    def question_time_budget(self, explicit_budget: Optional[float] = None) -> float:
        """
        Linear high-to-low budget allocation with a per-question base:
        - Reserve base_budget_seconds for every remaining question.
        - Distribute remaining time linearly (highest weight for the current question).
        """
        if explicit_budget is not None:
            return max(float(explicit_budget), 0.0)

        remaining_time = self.final_cutoff_time - time.time()
        if remaining_time <= 0:
            return 0.0

        remaining_questions = max(len(self._cutoffs), 1)
        base_total = self.base_budget_seconds * remaining_questions
        extra_time = max(0.0, remaining_time - base_total)

        # Linear weights: current question gets the largest share.
        # Weights sum = n(n+1)/2; current weight = remaining_questions.
        weight_sum = remaining_questions * (remaining_questions + 1) / 2
        weight_current = remaining_questions
        extra_for_current = extra_time * (weight_current / weight_sum)

        budget = self.base_budget_seconds + extra_for_current

        return max(0.0, min(budget, remaining_time))


TIME_MANAGER = TimeManager(
    CONFIG.timing.total_hours,
    CONFIG.timing.checkpoints,
    CONFIG.timing.base_budget_seconds,
)


# ============================================================
# Python Tool (Local Jupyter Kernel)
# ============================================================

%%writefile local_python_tool.py
"""Python tool using Jupyter kernel for stateful execution."""
import os
import queue
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from uuid import UUID, uuid4

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)


def add_libs(code: str) -> str:
    """Add common math libraries to code."""
    return "import math\nimport numpy as np\nimport sympy as sp\nfrom sympy import *\n" + code


def ensure_last_print(code: str) -> str:
    """Ensure the last expression is printed."""
    lines = code.strip().split("\n")
    if lines and "print(" not in lines[-1] and "import" not in lines[-1]:
        if "#" in lines[-1]:
            lines[-1] = lines[-1].split("#")[0]
        lines[-1] = "print(" + lines[-1] + ")"
    return "\n".join(lines)


class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Get next available ports for kernel connection."""
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            from pathlib import Path
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            # Allocate unique ports to avoid conflicts when running multiple kernels
            ports = self._get_next_ports(5)
            km = KernelManager()
            km.shell_port = ports[0]
            km.iopub_port = ports[1]
            km.stdin_port = ports[2]
            km.hb_port = ports[3]
            km.control_port = ports[4]
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr."""
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for kernel output.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain shell channel
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for execution reply.") from exc

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."

        return stdout

    def close(self):
        import contextlib
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(self, execution_backend: str | None = None, local_jupyter_timeout: float = 60.0):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        # Lazy initialization to avoid port conflicts during object creation
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."""

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=self.get_tool_name())
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        self._ensure_session()
        script = message.content[0].text
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
        return [self._make_response(output, channel=message.channel)]

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()

from local_python_tool import PythonTool

# ============================================================
# LMDeploy Pipeline Initialization
# ============================================================

_PIPELINE = None  # type: ignore[assignment]


def get_pipeline():
    """Lazily initialize and return the LMDeploy pipeline."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    lmdeploy_cfg = CONFIG.lmdeploy

    # GPU selection
    if lmdeploy_cfg.gpu_indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, lmdeploy_cfg.gpu_indices))
        num_gpus = len(lmdeploy_cfg.gpu_indices)
    else:
        num_gpus = torch.cuda.device_count()

    backend_cfg = TurbomindEngineConfig(
        tp=num_gpus,
        session_len=lmdeploy_cfg.session_len,
        max_batch_size=lmdeploy_cfg.max_batch_size,
        enable_prefix_caching=lmdeploy_cfg.enable_prefix_caching,
        cache_max_entry_count=lmdeploy_cfg.cache_max_entry_count,
        max_prefill_token_num=lmdeploy_cfg.max_prefill_token_num,
    )

    logger.info("Initializing LMDeploy pipeline...")
    _PIPELINE = lm_pipeline(lmdeploy_cfg.model_path, backend_cfg)
    logger.info("LMDeploy pipeline initialized.")
    return _PIPELINE


# ============================================================
# Harmony TIR Inferencer
# ============================================================

class HarmonyTIRInferencer:
    """Inferencer using Harmony protocol with TIR (Tool-Integrated Reasoning) and LMDeploy."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.lmdeploy_cfg = cfg.lmdeploy
        self.gen_cfg = cfg.generation
        self.tool_cfg = cfg.tool
        self.prompt_list = cfg.prompt_list

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.render_cfg = RenderConversationConfig(auto_drop_analysis=False)

        # LMDeploy generation config
        self.lmdeploy_gen_cfg = LMDeployGenerationConfig(
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            top_k=self.gen_cfg.top_k,
            min_p=self.gen_cfg.min_p,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            skip_special_tokens=self.gen_cfg.skip_special_tokens,
        )

    def _determine_sample_count(self, time_budget: Optional[float] = None) -> int:
        if time_budget is not None:
            if time_budget >= self.gen_cfg.high_budget_seconds:
                return max(1, self.gen_cfg.high_budget_samples)
            if time_budget <= self.gen_cfg.low_budget_seconds:
                return max(1, self.gen_cfg.low_budget_samples)
            return max(1, self.gen_cfg.mid_budget_samples)

        return max(1, self.gen_cfg.sample_count)

    def _apply_chat_template(self, prompt: str, python_tool: PythonTool) -> list[Message]:
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config),
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(CONFIG.prompt_list[0].user_suffix)
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

    def _trim_context(self, messages: list[Message]) -> list[Message]:
        """
        Keep only a leading and trailing window of the conversation (excluding system/user).
        early_window: number of messages kept after the initial system+user.
        recent_window: number of most recent messages to keep.
        """
        early = max(0, self.gen_cfg.context_early_window)
        recent = max(0, self.gen_cfg.context_recent_window)

        # Preserve initial system + user messages
        prefix = messages[:2]
        rest = messages[2:]

        if early + recent >= len(rest):
            return messages

        trimmed = prefix + rest[:early] + rest[-recent:]
        return trimmed

    def _format_prompts(self, problem: str, time_budget: Optional[float] = None) -> list[str]:
        """
        Build prompt strings following aimo.py weighting:
        - Each PromptTemplate has a `number` weight.
        - Repeat prompts proportionally to weights to reach sample count.
        """
        num_samples = max(1, self._determine_sample_count(time_budget))
        base_total = sum(max(1, tpl.number) for tpl in self.prompt_list)
        prompts: list[str] = []

        for tpl in self.prompt_list:
            repeat = max(1, round(num_samples * max(1, tpl.number) / base_total))
            for _ in range(repeat):
                # if tpl.system:
                #     user_content = tpl.system + "\n" + problem + tpl.user_suffix
                # else:
                #     user_content = problem + tpl.user_suffix
                user_content = problem
                prompts.append(user_content)
        return prompts[:num_samples]

    def inference(
        self,
        problem: str,
        deadline: float,
        time_budget: Optional[float] = None,
    ) -> tuple[int, list[str], list[Optional[int]], list[int], list[str]]:
        self._deadline = deadline
        start_time = time.time()

        prompts = self._format_prompts(problem, time_budget)
        responses, token_lens, finish_reasons = self._inference_parallel(prompts)

        duration = time.time() - start_time
        logger.info("[inference] Took %.2fs", duration)

        prediction, parsed_answers = self.parse_responses(responses)
        return prediction, responses, parsed_answers, token_lens, finish_reasons

    def _single_sample_worker(
        self,
        sample_idx: int,
        prompt: str,
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ) -> None:
        """
        Worker function that handles a single sample through multiple TIR iterations.
        Each sample can independently execute Python tools and continue generation.
        """
        python_tool = None
        try:
            python_tool = PythonTool(local_jupyter_timeout=self.tool_cfg.local_jupyter_timeout)
            messages = self._apply_chat_template(prompt, python_tool)
            final_answer = None
            token_count = 0
            finish_reason = ""
            final_answer_found = ""

            pipe = get_pipeline()

            for iteration in range(self.gen_cfg.max_iter):
                if self._deadline and time.time() >= self._deadline:
                    logger.warning("Sample %d: Deadline reached", sample_idx)
                    finish_reason = "deadline"
                    break
                if final_answer_found or (stop_event and stop_event.is_set()):
                    if not finish_reason:
                        finish_reason = "stop_event" if stop_event and stop_event.is_set() else "boxed"
                    break

                # Render conversation to prompt
                prompt_ids = self.encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages),
                    Role.ASSISTANT,
                )
                prompt_str = self.encoding.decode_utf8(prompt_ids)

                max_tokens = self.lmdeploy_cfg.session_len - len(prompt_ids)
                if max_tokens < 1:
                    logger.warning("Sample %d: Context full", sample_idx)
                    break

                gen_cfg = LMDeployGenerationConfig(
                    temperature=self.gen_cfg.temperature,
                    top_p=self.gen_cfg.top_p,
                    top_k=self.gen_cfg.top_k,
                    
                    max_new_tokens=min(max_tokens, self.gen_cfg.max_new_tokens),
                    skip_special_tokens=self.gen_cfg.skip_special_tokens,
                )

                token_buffer: list[int] = []
                token_buffer_str = ""
                breaking = False

                # Stream generation for this sample
                for resp in pipe.stream_infer(
                    prompts=[prompt_str],
                    gen_config=gen_cfg,
                    do_preprocess=False,
                ):
                    if stop_event.is_set():
                        breaking = True
                        finish_reason = "stop_event"
                        break

                    if resp.token_ids:
                        token_buffer.extend(resp.token_ids)
                        token_buffer_str += resp.text
                        token_count += resp.generate_token_len

                    if self._deadline and time.time() >= self._deadline:
                        finish_reason = "deadline"
                        breaking = True
                        break

                    if token_count > self.gen_cfg.token_limit:
                        logger.warning("Sample %d: Token limit exceeded", sample_idx)
                        finish_reason = "token_limit"
                        breaking = True
                        break

                    # Check for boxed answer
                    if "}" in resp.text and self.extract_boxed_text(token_buffer_str) is not None:
                        finish_reason = "boxed"
                        final_answer_found += token_buffer_str
                        breaking = True
                        break

                if breaking:
                    break

                if token_buffer:
                    new_messages = self.encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                    messages.extend(new_messages)
                    messages = self._trim_context(messages)

                    last_message = messages[-1]

                    # Check if stream ended
                    if last_message.channel == "final" or token_buffer[-1] == 200002:
                        if not finish_reason:
                            finish_reason = "stream_end"
                        break

                    # Execute Python tool if requested (independent of other samples)
                    if last_message.recipient == "python":
                        logger.info("Sample %d: Executing Python tool...", sample_idx)
                        response_msgs = python_tool.process_sync_plus(last_message)
                        messages.extend(response_msgs)
                        # Continue to next iteration


            if final_answer_found:
                result_queue.put(("answer", sample_idx, final_answer_found, token_count, finish_reason or "boxed"))

            if not finish_reason:
                finish_reason = "max_iter" if iteration + 1 >= self.gen_cfg.max_iter else "unknow"

            all_text = self.encoding.decode_utf8(
                self.encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    self.render_cfg,
                )
            )
            result_queue.put(("complete", sample_idx, all_text, token_count, finish_reason))

        except Exception as exc:
            logger.exception("Sample %d: Error in generation: %s", sample_idx, exc)
            result_queue.put(("complete", sample_idx, "", 0, "error"))
        finally:
            if python_tool:
                python_tool.close()

    def _inference_parallel(self, prompts: list[str]) -> tuple[list[str], list[int], list[str]]:
        """
        Run parallel inference with independent per-sample TIR loops.
        Uses ThreadPoolExecutor for true independence, but batching happens
        naturally when multiple samples hit the GPU at the same time.
        """
        answers_collected: list[int] = []
        raw_responses = [""] * len(prompts)
        token_lens = [0] * len(prompts)
        finish_reasons = [""] * len(prompts)
        majority_threshold = self.gen_cfg.majority_threshold
        majority_reached = False

        logger.info(
            "Sampling %d times (threshold: >= %d)...",
            len(prompts),
            majority_threshold,
        )

        stop_event = threading.Event()
        result_queue: queue.Queue = queue.Queue()
        completed_count = 0

        # Launch all workers
        executor = ThreadPoolExecutor(max_workers=len(prompts))
        futures = []
        for idx, prompt in enumerate(prompts):
            future = executor.submit(
                self._single_sample_worker,
                idx,
                prompt,
                stop_event,
                result_queue,
            )
            futures.append(future)

        try:
            # Process results as they come in
            while completed_count < len(prompts) and not majority_reached:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if result[0] == "answer" or result[0] == "complete":
                    # Intermediate answer found
                    _, sample_idx, text, tok_len, reason = result
                    raw_responses[sample_idx] = text
                    token_lens[sample_idx] = tok_len
                    finish_reasons[sample_idx] = reason

                    ans = self.extract_boxed_text(text)
                    if ans is not None:
                        answers_collected.append(answer)
                        counts = Counter(answers_collected)
                        most_common_ans, count = counts.most_common(1)[0]
                        if count >= majority_threshold:
                            logger.info(
                                "Majority reached: %s appeared %d times",
                                most_common_ans,
                                count,
                            )
                            majority_reached = True
                            stop_event.set()
                            break

                    completed_count += 1

        finally:
            # Cleanup
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=False)

        return raw_responses, token_lens, finish_reasons

    @staticmethod
    def extract_boxed_text(text: str) -> int | None:
        """Extract answer from \\boxed{} or 'final answer is' patterns."""
        pattern = r"oxed{(.*?)}"
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        clean_match = match.strip().replace(",", "").replace(" ", "")
                        val = int(float(clean_match[:20]))
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        pattern = r"(?i)final\s+answer\s*(?:is|:)?\s*(\d+)"
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        val = int(match)
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        return None

    def parse_responses(self, responses: list[str]) -> tuple[int, list[Optional[int]]]:
        answers: list[Optional[int]] = [self.extract_boxed_text(r) for r in responses]
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            logger.warning("No valid answers found; returning 0")
            return 0, answers

        counter = Counter(valid_answers)
        logger.info("Answers: %s", counter)

        most_common = counter.most_common(1)[0][0]
        return most_common % 100000, answers


INFERENCER = HarmonyTIRInferencer(CONFIG)


# ============================================================
# Prediction Tracking
# ============================================================

def save_response_record(
    q_id: int,
    question: str,
    responses: list[str],
    parsed: list[int | None],
    final_pred: int,
    token_lens: list[int],
    finish_reasons: list[str]
):
    """
    Save the full response record for later debugging/analysis.
    """
    record = {
        "id": q_id,
        "question": question,
        "responses": responses,
        "parsed_answers": parsed,
        "final_prediction": final_pred,
        "token_lengths": token_lens,
        "finish_reasons": finish_reasons,
        "timestamp": time.time(),
    }

    out_path = SAVE_DIR / f"{q_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    logger.info("Saved response to %s", out_path)

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: pl.DataFrame = None,) -> pl.DataFrame | pl.DataFrame:
    question_id = id_.item(0)
    question_text = question.item(0)

    logger.info("=" * 60)
    logger.info("Question ID: %s", question_id)
    logger.info("Question: %s", question_text)
    logger.info("=" * 60)

    # In competition rerun mode, skip heavy generation.
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN") and not CONFIG.dataset.use_server_for_eval:
        logger.info("No local eval mode detected, returning dummy prediction.")
        return pl.DataFrame({"id": question_id, "answer": 49})

    # if "Let $n \geq 6$ be a positive integer. We call a positive integer $n$-Norwegian" not in question_text:
    #     return pl.DataFrame({"id": question_id, "answer": 49})

    if TIME_MANAGER.after_final_cutoff():
        logger.warning("Final cutoff exceeded; returning default answer 49.")
        return pl.DataFrame({"id": question_id, "answer": 49})

    question_budget = TIME_MANAGER.question_time_budget()
    deadline = min(time.time() + question_budget, TIME_MANAGER.final_cutoff_time)
    logger.info(
        "Budget for this question: %.2fs (deadline: %s)",
        question_budget,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)),
    )

    start_time = time.time()
    (
        prediction,
        raw_responses,
        parsed_answers,
        token_lens,
        finish_reasons,
    ) = INFERENCER.inference(
        question_text, deadline=deadline, time_budget=question_budget
    )
    consumed_time = time.time() - start_time

    # Save response for offline debugging/evaluation
    if CONFIG.dataset.use_server_for_eval:
        save_response_record(
            q_id=question_id,
            question=question_text,
            responses=raw_responses,
            parsed=parsed_answers,
            final_pred=prediction,
            token_lens=token_lens,
            finish_reasons=finish_reasons
        )

    logger.info("Raw parsed predictions: %s", parsed_answers)
    logger.info("Response token lengths: %s", token_lens)
    logger.info("Finish reasons: %s", finish_reasons)
    logger.info("Question time budget: %.2fs, elapsed: %.2fs", question_budget, consumed_time)
    logger.info("Final aggregated prediction: %s", prediction)
    logger.info("=" * 60)

    # Consume one cutoff step per question, like original pop()
    TIME_MANAGER.consume_iteration()
    
    return pl.DataFrame({"id": question_id, "answer": prediction})


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_reference_dataset(cfg: DatasetConfig) -> Dict[str | int, int]:
    df = pd.read_csv(cfg.reference_csv)
    ground_truth = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}
    df.drop("answer", axis=1, errors="ignore").to_csv(cfg.submission_csv, index=False)
    logger.info("Prepared reference dataset at %s (ground truth available: %s)", cfg.submission_csv, bool(ground_truth))
    return ground_truth


# ============================================================
# Main Entrypoint
# ============================================================

def _set_random_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def main() -> None:
    program_start = time.time()
    logger.info("Setting random seed to %d", CONFIG.generation.seed)
    _set_random_seeds(CONFIG.generation.seed)

    # Initialize pipeline
    logger.info("Initializing LMDeploy pipeline...")
    _ = get_pipeline()

    setup_seconds = time.time() - program_start
    TIME_MANAGER.reset_after_setup(setup_seconds)
    logger.info(
        "Setup time: %.2fs, usable generation time: %.2fs",
        setup_seconds,
        TIME_MANAGER.usable_seconds,
    )

    inference_server = aimo_server.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        logger.info("Starting inference server in competition rerun mode...")
        inference_server.serve()
        return

    if CONFIG.dataset.use_server_for_eval:
        logger.info("Running local gateway for evaluation...")
        start_eval = time.time()
        inference_server.run_local_gateway((CONFIG.dataset.reference_csv,))

        ref_df = pd.read_csv(CONFIG.dataset.reference_csv)
        sub_df = pd.read_parquet(CONFIG.dataset.submission_parquet)

        total = 0
        correct = 0

        for _, row in ref_df.iterrows():
            q_id = row["id"]
            true_ans = row["answer"]

            # safe slice to avoid KeyError if missing
            sub_rows = sub_df[sub_df["id"] == q_id]
            if sub_rows.empty:
                logger.warning("Missing prediction for question ID %s", q_id)
                continue

            pred_ans = sub_rows.iloc[0]["answer"]

            status = "✅" if true_ans == pred_ans else "❌"
            logger.info("  %s: pred=%s, gt=%s %s", qid, pred_ans, true_ans, status)

            total += 1
            if true_ans == pred_ans:
                correct += 1

        logger.info("=" * 60)
        logger.info("Accuracy: %d/%d", correct, total)
        end_eval = time.time()
        logger.info("Consumed time: %.2f mins", (end_eval - start_eval) / 60.0)


if __name__ == "__main__":
    main()