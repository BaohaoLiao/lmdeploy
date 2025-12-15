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
# Quiet noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================
# Environment Setup
# ============================================================

ENV_VARS = {
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_FLAX": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "CUDA_VISIBLE_DEVICES": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TIKTOKEN_ENCODINGS_BASE": "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings",
}

for _k, _v in ENV_VARS.items():
    os.environ.setdefault(_k, _v)


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class LMDeployConfig:
    model_path: str
    gpu_indices: Optional[List[int]] = None
    max_batch_size: int = 64
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
    sample_count: int = 8
    use_budget: bool = False
    max_iter: int = 100
    base_budget_seconds: float = 60 * 5.5
    initial_budget_seconds: float = 370.0
    high_budget_samples: int = 8
    mid_budget_samples: int = 6
    low_budget_samples: int = 4
    high_budget_seconds: float = 300.0
    low_budget_seconds: float = 180.0
    majority_threshold: float = 0.5  # fraction of samples required to stop early
    token_limit: int = 60000
    context_early_window: int = 2
    context_recent_window: int = 2
    max_new_tokens: int = 32000
    skip_special_tokens: bool = False  # Must be False to preserve Harmony protocol tokens
    do_sample: bool = True


@dataclass
class ToolConfig:
    local_jupyter_timeout: float = 60.0


@dataclass
class TimingConfig:
    total_hours: float = 4 + 55 / 60
    checkpoints: int = 50
    early_minutes: float = 12.0
    base_budget_seconds: float = 180.0


@dataclass
class DatasetConfig:
    reference_csv: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    submission_csv: str = "reference.csv"


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
    ),
    generation=GenerationConfig(),
    tool=ToolConfig(),
    timing=TimingConfig(),
    dataset=DatasetConfig(),
    prompt_list=[
        PromptTemplate(
            system="",
            user_suffix="\nPlease reason step by step, and put the final answer (only integer) within \\boxed{}.",
            number=16,
        )
    ],
)


# ============================================================
# Time Management (Cutoffs)
# ============================================================

class TimeManager:
    """Dynamic time budgeting with setup time accounted for."""

    def __init__(self, total_hours: float, early_minutes: float, steps: int, base_budget_seconds: float) -> None:
        self.config_total_seconds = total_hours * 3600
        self.early_minutes = early_minutes
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
            self.start_time + self.early_minutes * 60,
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
    CONFIG.timing.early_minutes,
    CONFIG.timing.checkpoints,
    CONFIG.timing.base_budget_seconds,
)


# ============================================================
# Utility Functions
# ============================================================

def set_random_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_model_cache(path: str, exts: Sequence[str] = (".bin", ".pt", ".safetensors"), num_workers: Optional[int] = None, chunk_mb: int = 256) -> int:
    """Pre-read model weight files into OS page cache."""
    import multiprocessing

    def _warmup_file(fpath: str) -> tuple[str, int]:
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(tuple(exts))
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        raise ValueError(f"No model files found under: {path}")

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    logger.info("[cache_model] %d file(s), %d worker(s)", len(files), num_workers)
    start = time.time()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_warmup_file, f): f for f in files}
        for idx, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            logger.info("[%d/%d] cached %s", idx, len(files), os.path.basename(fpath))

    elapsed = time.time() - start
    gb = total_bytes / 1024**3
    logger.info("[cache_model] total read ≈ %.2f GB in %.2fs", gb, elapsed)
    return total_bytes


# ============================================================
# Python Tool (Local Jupyter Kernel)
# ============================================================

class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0) -> None:
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:  # pragma: no cover - dependency is expected in runtime env
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._km: "KernelManager | None" = None

        if connection_file:
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
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

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self) -> None:  # pragma: no cover
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(self, local_jupyter_timeout: float = 60.0) -> None:
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def instruction(self) -> str:
        return "Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name=self.get_tool_name(), description=self.instruction, tools=[])

    def _ensure_session(self) -> None:
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=self.get_tool_name())
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        script = message.content[0].text
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
        return [self._make_response(output, channel=message.channel)]

    def close(self) -> None:
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self) -> None:  # pragma: no cover
        self.close()


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

        self.budget_seconds = self.gen_cfg.initial_budget_seconds
        self.render_cfg = RenderConversationConfig(auto_drop_analysis=False)

        # LMDeploy generation config
        self.lmdeploy_gen_cfg = LMDeployGenerationConfig(
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            top_k=self.gen_cfg.top_k,
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

        if not self.gen_cfg.use_budget:
            logger.info("Budget disabled -> N: %d", self.gen_cfg.sample_count)
            return max(1, self.gen_cfg.sample_count)

        estimated = (self.budget_seconds - 190) / 90
        ret = min(self.gen_cfg.sample_count, math.floor(estimated))
        ret = max(1, ret)
        logger.info("Budget: %.2fs -> N: %d", self.budget_seconds, ret)
        return ret

    def _apply_chat_template(self, prompt: str, python_tool: PythonTool) -> list[Message]:
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config),
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
                if tpl.system:
                    user_content = tpl.system + "\n" + problem + tpl.user_suffix
                else:
                    user_content = problem + tpl.user_suffix
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

        if self.gen_cfg.use_budget:
            budget_left = max(0.0, self.budget_seconds - duration)
            self.budget_seconds = self.gen_cfg.base_budget_seconds + budget_left
            logger.info("[inference] Updated budget: %.2fs", self.budget_seconds)

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
        Worker function that handles a single sample generation.
        Generates with tool config for Harmony protocol format, but doesn't parse/execute tools.
        """
        try:
            # Create tool config for proper Harmony protocol format
            python_tool = PythonTool(local_jupyter_timeout=self.tool_cfg.local_jupyter_timeout)
            tool_config = python_tool.tool_config

            # Render prompt with tool configuration
            messages = [
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new()
                    .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                    .with_tools(tool_config),
                ),
                Message.from_role_and_content(Role.USER, prompt),
            ]

            prompt_ids = self.encoding.render_conversation_for_completion(
                Conversation.from_messages(messages),
                Role.ASSISTANT,
            )
            prompt_str = self.encoding.decode_utf8(prompt_ids)

            pipe = get_pipeline()

            max_tokens = self.lmdeploy_cfg.session_len - len(prompt_ids)
            if max_tokens < 1:
                logger.warning("Sample %d: Context full", sample_idx)
                result_queue.put(("complete", sample_idx, "", 0, "context_full", None))
                return

            gen_cfg = LMDeployGenerationConfig(
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                top_k=self.gen_cfg.top_k,
                max_new_tokens=min(max_tokens, self.gen_cfg.max_new_tokens),
                skip_special_tokens=self.gen_cfg.skip_special_tokens,
                stop_token_ids=self.stop_token_ids,  # Harmony protocol stop tokens
            )

            token_buffer_str = ""
            token_count = 0
            finish_reason = ""
            final_answer = None

            # Stream generation for this sample
            for resp in pipe.stream_infer(
                prompts=[prompt_str],
                gen_config=gen_cfg,
                do_preprocess=False,
            ):
                if stop_event.is_set():
                    finish_reason = "stop_event"
                    break

                if resp.text:
                    token_buffer_str += resp.text
                    token_count = resp.generate_token_len

                if self._deadline and time.time() >= self._deadline:
                    finish_reason = "deadline"
                    break

                if token_count > self.gen_cfg.token_limit:
                    logger.warning("Sample %d: Token limit exceeded", sample_idx)
                    finish_reason = "token_limit"
                    break

                # Check for boxed answer
                if "}" in resp.text:
                    candidate = self.extract_boxed_text(token_buffer_str)
                    if candidate is not None:
                        final_answer = candidate
                        finish_reason = "boxed"
                        # Notify via queue immediately
                        result_queue.put(("answer", sample_idx, candidate))
                        break

            if not finish_reason:
                finish_reason = "completed"

            # Send final result
            result_queue.put(("complete", sample_idx, token_buffer_str, token_count, finish_reason, final_answer))

            # Cleanup
            python_tool.close()

        except Exception as exc:
            logger.exception("Sample %d: Error in generation: %s", sample_idx, exc)
            result_queue.put(("complete", sample_idx, "", 0, "error", None))
            if 'python_tool' in locals():
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
        majority_threshold = len(prompts) * self.gen_cfg.majority_threshold
        majority_reached = False

        logger.info(
            "Sampling %d times (threshold: > %.2f, cfg=%.2f)...",
            len(prompts),
            majority_threshold,
            self.gen_cfg.majority_threshold,
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

                if result[0] == "answer":
                    # Intermediate answer found
                    _, sample_idx, answer = result
                    answers_collected.append(answer)

                    # Check for majority
                    counts = Counter(answers_collected)
                    most_common_ans, count = counts.most_common(1)[0]
                    if count > majority_threshold:
                        logger.info(
                            "Majority reached: %s appeared %d times",
                            most_common_ans,
                            count,
                        )
                        majority_reached = True
                        stop_event.set()
                        break

                elif result[0] == "complete":
                    # Sample completed
                    _, sample_idx, text, tok_len, reason, final_ans = result
                    raw_responses[sample_idx] = text
                    token_lens[sample_idx] = tok_len
                    finish_reasons[sample_idx] = reason

                    # Add final answer if not already added
                    if final_ans is not None and final_ans not in answers_collected:
                        answers_collected.append(final_ans)

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

@dataclass
class PredictionTracker:
    ground_truth: Dict[str | int, int] = field(default_factory=dict)
    predictions: Dict[str | int, int] = field(default_factory=dict)
    correct: int = 0
    total: int = 0

    def record(self, q_id: str | int, answer: int) -> None:
        self.predictions[q_id] = answer
        self.total += 1
        if self.ground_truth and q_id in self.ground_truth and answer == self.ground_truth[q_id]:
            self.correct += 1

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total


PREDICTION_TRACKER = PredictionTracker()


# ============================================================
# Prediction Function (Kaggle API)
# ============================================================

def predict(
    id_: pl.DataFrame,
    question: pl.DataFrame,
    answer: pl.DataFrame | None = None,
) -> pl.DataFrame:
    question_id = id_.item(0)
    question_text = question.item(0)

    logger.info("=" * 60)
    logger.info("Question ID: %s", question_id)
    logger.info("Question: %s", question_text)
    logger.info("=" * 60)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        logger.info("Rerun mode detected, returning dummy prediction.")
        return pl.DataFrame({"id": question_id, "answer": 0})

    if TIME_MANAGER.after_final_cutoff():
        logger.warning("Final cutoff exceeded; returning default answer 0.")
        return pl.DataFrame({"id": question_id, "answer": 0})

    question_budget = TIME_MANAGER.question_time_budget()
    deadline = min(time.time() + question_budget, TIME_MANAGER.final_cutoff_time)
    logger.info(
        "Budget for this question: %.2fs (deadline: %s)",
        question_budget,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)),
    )

    (
        prediction,
        raw_responses,
        parsed_answers,
        token_lens,
        finish_reasons,
    ) = INFERENCER.inference(
        question_text,
        deadline=deadline,
        time_budget=question_budget,
    )
    # raw_responses: list[str], parsed_answers: list[Optional[int]], token_lens: list[int], finish_reasons: list[str]
    PREDICTION_TRACKER.record(question_id, prediction)
    TIME_MANAGER.consume_iteration()

    logger.info("Prediction: %s", prediction)
    logger.info("Running accuracy: %d/%d (%.1f%%)", PREDICTION_TRACKER.correct, PREDICTION_TRACKER.total, PREDICTION_TRACKER.accuracy())
    logger.info("=" * 60)

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

def main() -> None:
    program_start = time.time()
    set_random_seeds(CONFIG.generation.seed)

    # Optional warmup if explicitly requested
    if os.getenv("WARMUP_MODEL_CACHE") == "1":
        warmup_model_cache(CONFIG.lmdeploy.model_path, chunk_mb=1024)

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
    PREDICTION_TRACKER.ground_truth = prepare_reference_dataset(CONFIG.dataset)

    inference_server = aimo_server.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        logger.info("Starting inference server in competition rerun mode...")
        inference_server.serve()
        return

    logger.info("Running local gateway for evaluation...")
    inference_server.run_local_gateway((CONFIG.dataset.submission_csv,))

    if PREDICTION_TRACKER.ground_truth and PREDICTION_TRACKER.total > 0:
        logger.info("=" * 50)
        logger.info("FINAL ACCURACY: %d/%d (%.1f%%)", PREDICTION_TRACKER.correct, PREDICTION_TRACKER.total, PREDICTION_TRACKER.accuracy())
        logger.info("=" * 50)
        for qid, pred in PREDICTION_TRACKER.predictions.items():
            gt = PREDICTION_TRACKER.ground_truth.get(qid)
            if gt is None:
                continue
            status = "✅" if pred == gt else "❌"
            logger.info("  %s: pred=%s, gt=%s %s", qid, pred, gt, status)


if __name__ == "__main__":
    main()
