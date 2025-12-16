#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-consistency TIR inference script using LMDeploy with proper TIR iterations.

This version includes actual tool execution in independent worker threads.
"""

from __future__ import annotations

# ============================================================
# Standard Library Imports
# ============================================================

import contextlib
import json
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
    DeveloperContent,
    ReasoningEffort,
    RenderConversationConfig,
)

# ============================================================
# LMDeploy Imports
# ============================================================

from lmdeploy import pipeline as lm_pipeline
from lmdeploy import GenerationConfig as LMDeployGenerationConfig
from lmdeploy import TurbomindEngineConfig

# ============================================================
# Local Imports
# ============================================================

import aimo_server
from aimo import (
    AppConfig,
    PromptTemplate,
    load_config,
    PythonTool,
)

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

CONFIG = load_config()

# ============================================================
# LMDeploy Configuration
# ============================================================

@dataclass
class LMDeployConfig:
    """Configuration for LMDeploy pipeline."""
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
    """Generation parameters for self-consistency sampling."""
    sample_count: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    seed: int = 42
    use_budget: bool = False
    initial_budget_seconds: float = 600.0
    base_budget_seconds: float = 300.0
    mid_budget_samples: int = 6
    high_budget_samples: int = 8
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
    max_iter: int = 20  # Maximum TIR iterations per sample


@dataclass
class ToolConfig:
    local_jupyter_timeout: float = 60.0


# ============================================================
# Pipeline Management
# ============================================================

_PIPELINE = None


def get_pipeline():
    """Get or create the global LMDeploy pipeline."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    lmdeploy_cfg = CONFIG.lmdeploy

    # Determine GPU configuration
    if lmdeploy_cfg.gpu_indices:
        num_gpus = len(lmdeploy_cfg.gpu_indices)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, lmdeploy_cfg.gpu_indices))
    else:
        num_gpus = torch.cuda.device_count()

    logger.info("Initializing LMDeploy with %d GPUs", num_gpus)

    backend_cfg = TurbomindEngineConfig(
        tp=num_gpus,
        session_len=lmdeploy_cfg.session_len,
        max_batch_size=lmdeploy_cfg.max_batch_size,
        enable_prefix_caching=lmdeploy_cfg.enable_prefix_caching,
        cache_max_entry_count=lmdeploy_cfg.cache_max_entry_count,
        max_prefill_token_num=lmdeploy_cfg.max_prefill_token_num,
    )

    _PIPELINE = lm_pipeline(
        lmdeploy_cfg.model_path,
        backend_config=backend_cfg,
    )

    logger.info("LMDeploy pipeline initialized")
    return _PIPELINE


# ============================================================
# Time Management
# ============================================================

class TimeManager:
    """Manages time budgets for the competition."""

    def __init__(self, total_seconds: float, warmup_seconds: float = 0.0):
        self.total_seconds = total_seconds
        self.warmup_seconds = warmup_seconds
        self.usable_seconds = total_seconds - warmup_seconds
        self.start_time = time.time()
        self.final_cutoff_time = self.start_time + total_seconds
        self.iterations_consumed = 0

    def reset_after_setup(self, setup_seconds: float) -> None:
        """Reset timing after model setup is complete."""
        self.start_time = time.time()
        self.usable_seconds = self.total_seconds - setup_seconds
        self.final_cutoff_time = self.start_time + self.usable_seconds

    def question_time_budget(self) -> float:
        """Calculate remaining time budget per question."""
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.usable_seconds - elapsed)
        questions_left = max(1, 50 - self.iterations_consumed)
        return remaining / questions_left

    def consume_iteration(self) -> None:
        """Mark one question as completed."""
        self.iterations_consumed += 1


TIME_MANAGER = TimeManager(
    total_seconds=CONFIG.total_seconds,
    warmup_seconds=CONFIG.warmup_seconds,
)


# ============================================================
# Utility Functions
# ============================================================

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_model_cache(model_path: str, chunk_mb: int = 1024) -> None:
    """Warmup model cache by loading model files."""
    logger.info("Warming up model cache from %s", model_path)
    model_dir = Path(model_path)
    if not model_dir.exists():
        logger.warning("Model path does not exist: %s", model_path)
        return

    for file_path in model_dir.rglob("*.safetensors"):
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_mb * 1024 * 1024)
                    if not chunk:
                        break
        except Exception as e:
            logger.warning("Error reading %s: %s", file_path, e)

    logger.info("Model cache warmup complete")


# ============================================================
# TIR Inferencer with Proper Token Counting
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
        messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config),
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

        # Add developer instructions if configured
        if CONFIG.prompt_list and CONFIG.prompt_list[0].user_suffix:
            messages.insert(1, Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(CONFIG.prompt_list[0].user_suffix)
            ))

        return messages

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
        Worker function that handles a single sample through multiple TIR iterations.
        Each sample can independently execute Python tools and continue generation.

        FIXED: Proper token counting across all iterations.
        """
        python_tool = None
        try:
            python_tool = PythonTool(local_jupyter_timeout=self.tool_cfg.local_jupyter_timeout)
            messages = self._apply_chat_template(prompt, python_tool)

            # Initialize tracking variables OUTSIDE iteration loop
            total_token_count = 0  # Accumulates across ALL iterations
            finish_reason = ""
            final_answer = None
            final_answer_found = False

            pipe = get_pipeline()

            for iteration in range(self.gen_cfg.max_iter):
                # Check termination conditions
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
                    logger.warning("Sample %d: Context full at iteration %d", sample_idx, iteration)
                    finish_reason = "context_full"
                    break

                gen_cfg = LMDeployGenerationConfig(
                    temperature=self.gen_cfg.temperature,
                    top_p=self.gen_cfg.top_p,
                    top_k=self.gen_cfg.top_k,
                    max_new_tokens=min(max_tokens, self.gen_cfg.max_new_tokens),
                    skip_special_tokens=self.gen_cfg.skip_special_tokens,
                    stop_token_ids=self.stop_token_ids,
                )

                # Token buffer for THIS iteration only
                token_buffer: list[int] = []
                token_buffer_str = ""
                breaking = False

                # Stream generation for this iteration
                for resp in pipe.stream_infer(
                    prompts=[prompt_str],
                    gen_config=gen_cfg,
                    do_preprocess=False,
                ):
                    if stop_event.is_set():
                        breaking = True
                        finish_reason = "stop_event"
                        break

                    # Accumulate tokens for this iteration
                    if resp.token_ids:
                        token_buffer.extend(resp.token_ids)
                        token_buffer_str = pipe.tokenizer.decode(token_buffer, skip_special_tokens=False)

                    if self._deadline and time.time() >= self._deadline:
                        finish_reason = "deadline"
                        breaking = True
                        break

                    # Check TOTAL token limit (across all iterations)
                    if total_token_count + len(token_buffer) > self.gen_cfg.token_limit:
                        logger.warning("Sample %d: Token limit exceeded (total: %d)",
                                     sample_idx, total_token_count + len(token_buffer))
                        finish_reason = "token_limit"
                        breaking = True
                        break

                    # Check for boxed answer
                    if "}" in resp.text:
                        candidate = self.extract_boxed_text(token_buffer_str)
                        if candidate is not None:
                            final_answer = candidate
                            finish_reason = "boxed"
                            final_answer_found = True
                            # Notify immediately
                            result_queue.put(("answer", sample_idx, candidate))
                            breaking = True
                            break

                # Add this iteration's tokens to total count
                total_token_count += len(token_buffer)

                if breaking:
                    break

                # Parse messages from this iteration's output
                if token_buffer:
                    new_messages = self.encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                    messages.extend(new_messages)
                    messages = self._trim_context(messages)

                    last_message = messages[-1]

                    # Check if stream ended naturally
                    if last_message.channel == "final" or (token_buffer and token_buffer[-1] == 200002):
                        if not finish_reason:
                            finish_reason = "stream_end"
                        break

                    # Execute Python tool if requested (independent of other samples)
                    if last_message.recipient == "python":
                        logger.info("Sample %d iteration %d: Executing Python tool...", sample_idx, iteration)
                        response_msgs = python_tool.process_sync_plus(last_message)
                        messages.extend(response_msgs)
                        # Continue to next iteration

            # Set final reason if not already set
            if not finish_reason:
                finish_reason = "max_iter" if iteration + 1 >= self.gen_cfg.max_iter else "unknown"

            # Render final conversation
            all_text = self.encoding.decode_utf8(
                self.encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    self.render_cfg,
                )
            )

            # Send completion with accurate total token count
            result_queue.put(("complete", sample_idx, all_text, total_token_count, finish_reason, final_answer))

        except Exception as exc:
            logger.exception("Sample %d: Error in generation: %s", sample_idx, exc)
            result_queue.put(("complete", sample_idx, "", 0, "error", None))
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

        # FIXED: Calculate actual threshold (not using float directly)
        majority_threshold = int(len(prompts) * self.gen_cfg.majority_threshold)
        majority_reached = False

        logger.info(
            "Sampling %d times (threshold: >= %d, cfg=%.2f)...",
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
                    _, sample_idx, ans = result  # FIXED: Use 'ans' not 'answer'
                    answers_collected.append(ans)

                    # Check for majority
                    counts = Counter(answers_collected)
                    most_common_ans, count = counts.most_common(1)[0]
                    if count >= majority_threshold:  # FIXED: Use >= instead of >
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

                        # Check for majority again
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
# Prediction Tracker
# ============================================================

@dataclass
class PredictionTracker:
    ground_truth: Dict[str | int, int] = field(default_factory=dict)
    predictions: Dict[str | int, int] = field(default_factory=dict)
    correct: int = 0
    total: int = 0
    total_tokens: int = 0  # Total tokens generated across all questions
    total_time: float = 0.0  # Total generation time in seconds
    question_tokens: Dict[str | int, int] = field(default_factory=dict)  # Tokens per question
    question_times: Dict[str | int, float] = field(default_factory=dict)  # Time per question

    def record(self, q_id: str | int, answer: int, tokens: int = 0, duration: float = 0.0) -> None:
        self.predictions[q_id] = answer
        self.total += 1
        self.total_tokens += tokens
        self.total_time += duration
        self.question_tokens[q_id] = tokens
        self.question_times[q_id] = duration
        if self.ground_truth and q_id in self.ground_truth and answer == self.ground_truth[q_id]:
            self.correct += 1

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def tokens_per_second(self) -> float:
        """Calculate overall tokens per second."""
        if self.total_time <= 0:
            return 0.0
        return self.total_tokens / self.total_time

    def avg_tokens_per_question(self) -> float:
        """Calculate average tokens per question."""
        if self.total == 0:
            return 0.0
        return self.total_tokens / self.total

    def save_to_file(self, filepath: str) -> None:
        """Save detailed per-question metrics to a JSON file."""
        data = {
            "summary": {
                "total_questions": self.total,
                "correct": self.correct,
                "accuracy": self.accuracy(),
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "overall_tokens_per_second": self.tokens_per_second(),
                "avg_tokens_per_question": self.avg_tokens_per_question(),
            },
            "per_question": []
        }

        for q_id in self.predictions.keys():
            question_data = {
                "question_id": str(q_id),
                "prediction": self.predictions[q_id],
                "ground_truth": self.ground_truth.get(q_id) if self.ground_truth else None,
                "correct": self.predictions[q_id] == self.ground_truth.get(q_id) if self.ground_truth and q_id in self.ground_truth else None,
                "tokens": self.question_tokens.get(q_id, 0),
                "duration": self.question_times.get(q_id, 0.0),
                "tokens_per_second": self.question_tokens.get(q_id, 0) / self.question_times.get(q_id, 1.0) if self.question_times.get(q_id, 0) > 0 else 0.0,
            }
            data["per_question"].append(question_data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Saved detailed metrics to %s", filepath)


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

    question_budget = TIME_MANAGER.question_time_budget()
    deadline = min(time.time() + question_budget, TIME_MANAGER.final_cutoff_time)
    logger.info(
        "Budget for this question: %.2fs (deadline: %s)",
        question_budget,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)),
    )

    question_start = time.time()
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
    question_duration = time.time() - question_start

    # raw_responses: list[str], parsed_answers: list[Optional[int]], token_lens: list[int], finish_reasons: list[str]
    # Calculate total tokens for this question
    total_question_tokens = sum(token_lens)

    # Record with token and time tracking
    PREDICTION_TRACKER.record(question_id, prediction, tokens=total_question_tokens, duration=question_duration)
    TIME_MANAGER.consume_iteration()

    # Calculate tokens/s for this question
    question_tps = total_question_tokens / question_duration if question_duration > 0 else 0.0

    logger.info("Prediction: %s", prediction)
    logger.info("Question tokens: %d, time: %.2fs, throughput: %.2f tokens/s",
                total_question_tokens, question_duration, question_tps)
    logger.info("Running accuracy: %d/%d (%.1f%%)", PREDICTION_TRACKER.correct, PREDICTION_TRACKER.total, PREDICTION_TRACKER.accuracy())
    logger.info("Overall throughput: %.2f tokens/s (avg %.1f tokens/question)",
                PREDICTION_TRACKER.tokens_per_second(), PREDICTION_TRACKER.avg_tokens_per_question())
    logger.info("=" * 60)

    return pl.DataFrame({"id": question_id, "answer": prediction})


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_reference_dataset(cfg) -> Dict[str | int, int]:
    """Prepare reference dataset with ground truth answers."""
    if not Path(cfg.submission_csv).exists():
        logger.warning("Submission CSV not found: %s", cfg.submission_csv)
        return {}

    df = pl.read_csv(cfg.submission_csv)
    if "answer" not in df.columns:
        logger.warning("No 'answer' column in submission CSV")
        return {}

    ground_truth = {row["id"]: row["answer"] for row in df.iter_rows(named=True)}
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

    # Save metrics to file
    metrics_file = "prediction_metrics_tir.json"
    if PREDICTION_TRACKER.total > 0:
        PREDICTION_TRACKER.save_to_file(metrics_file)

    if PREDICTION_TRACKER.ground_truth and PREDICTION_TRACKER.total > 0:
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info("Accuracy: %d/%d (%.1f%%)",
                    PREDICTION_TRACKER.correct,
                    PREDICTION_TRACKER.total,
                    PREDICTION_TRACKER.accuracy())
        logger.info("Total tokens: %d", PREDICTION_TRACKER.total_tokens)
        logger.info("Total time: %.2fs", PREDICTION_TRACKER.total_time)
        logger.info("Overall throughput: %.2f tokens/s", PREDICTION_TRACKER.tokens_per_second())
        logger.info("Average tokens per question: %.1f", PREDICTION_TRACKER.avg_tokens_per_question())
        logger.info("=" * 60)
        logger.info("Per-Question Results:")
        logger.info("=" * 60)
        for qid, pred in PREDICTION_TRACKER.predictions.items():
            gt = PREDICTION_TRACKER.ground_truth.get(qid)
            if gt is None:
                continue
            status = "✅" if pred == gt else "❌"
            q_tokens = PREDICTION_TRACKER.question_tokens.get(qid, 0)
            q_time = PREDICTION_TRACKER.question_times.get(qid, 0.0)
            q_tps = q_tokens / q_time if q_time > 0 else 0.0
            logger.info("  %s: pred=%s, gt=%s %s | %d tokens, %.2fs, %.2f tok/s",
                       qid, pred, gt, status, q_tokens, q_time, q_tps)


if __name__ == "__main__":
    main()
