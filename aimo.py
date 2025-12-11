#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Kaggle AIMO-3 Inference Script (Improved)

This script is a cleaned and refactored version of the notebook code.
It keeps the original logic and hyperparameters but improves structure,
robustness, and readability.
"""

from __future__ import annotations

# ============================================================
# Standard Library Imports
# ============================================================

import os
import re
import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence

# ============================================================
# Third-Party Imports
# ============================================================

import numpy as np
import pandas as pd
import polars as pl
import torch
from collections import defaultdict

from lmdeploy import pipeline as lm_pipeline, TurbomindEngineConfig, GenerationConfig
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
pd.set_option("display.max_colwidth", None)
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/usr/lib/lmdeploy_package/tiktoken_encodings"
os.environ["TIKTOKEN_RS_CACHE_DIR"] = "/kaggle/usr/lib/lmdeploy_package/tiktoken_encodings"


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class ModelConfig:
    model_path: str
    gpu_indices: Optional[List[int]] = None
    reasoning_effort: Optional[str] = "high"
    enable_thinking: Optional[bool] = None


@dataclass
class InferenceConfig:
    max_batch_size: int = 16
    enable_prefix_caching: bool = True
    cache_max_entry_count: float = 0.97
    max_prefill_token_num: int = 4096
    gen_max_new_tokens: int = 32000  # used in session length, then removed


@dataclass
class PromptConfig:
    system: str
    user_suffix: str
    number: int


@dataclass
class ActorGenConfig:
    temperature: float = 1.0
    skip_special_tokens: bool = True
    max_new_tokens: int = 32000
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True


@dataclass
class ActorConfig:
    gen_cfg: ActorGenConfig
    prompt_list: List[PromptConfig]


@dataclass
class AppConfig:
    main_model: ModelConfig
    inference_cfg: InferenceConfig
    actor: ActorConfig
    exam_dataset_files: str
    output_path: str = "results"
    use_server_for_eval: bool = True
    seed: Optional[int] = 42


# ============================================================
# Concrete Configuration Instance
# ============================================================

CONFIG = AppConfig(
    main_model=ModelConfig(
        model_path="/kaggle/input/gpt-oss-120b/transformers/default/1",
        gpu_indices=[0],
        reasoning_effort="high",
        enable_thinking=None,
    ),
    inference_cfg=InferenceConfig(),
    actor=ActorConfig(
        gen_cfg=ActorGenConfig(),
        prompt_list=[
            PromptConfig(
                system="",
                user_suffix="\nPlease reason step by step, and put the final answer (only integer) within \\boxed{}.",
                number=16,
            )
        ],
    ),
    exam_dataset_files="/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv",
    output_path="results",
    use_server_for_eval=True,
    seed=42,
)


# ============================================================
# Time Management (Cutoffs)
# ============================================================

class TimeManager:
    """
    Encapsulates timing logic:
    - final hard cutoff for generation
    - a sequence of intermediate cutoffs to adjust weight
    """

    def __init__(self, total_hours: float = 4.75, early_minutes: float = 12.0, steps: int = 50):
        self.start_time: float = time.time()
        self.final_cutoff_time: float = self.start_time + total_hours * 3600

        # Create descending cutoff times from final_cutoff_time down to (start + early_minutes)
        cutoff_array = np.linspace(
            self.final_cutoff_time,
            self.start_time + early_minutes * 60,
            steps + 1,
        )
        cutoff_list = [int(x) for x in cutoff_array]
        cutoff_list.pop()  # Remove the earliest (smallest) time
        self._cutoffs: List[int] = cutoff_list  # descending list

    def after_final_cutoff(self) -> bool:
        return time.time() > self.final_cutoff_time

    def should_downweight(self) -> bool:
        """
        Return True if current time is past the last remaining cutoff.
        Does not consume the cutoff.
        """
        if not self._cutoffs:
            # No cutoffs left: always downweight
            return True
        return time.time() > self._cutoffs[-1]

    def consume_iteration(self) -> None:
        """Consume one cutoff per prediction call to mimic original pop behavior."""
        if self._cutoffs:
            self._cutoffs.pop()


TIME_MANAGER = TimeManager()


# ============================================================
# Lazy Pipeline Initialization
# ============================================================

_PIPELINE = None  # type: ignore[assignment]


def get_pipeline():
    """
    Lazily initialize and return the model pipeline.
    This keeps import-time side effects small and easier to debug.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    model_cfg = CONFIG.main_model
    inf_cfg = CONFIG.inference_cfg

    # GPU selection
    if model_cfg.gpu_indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, model_cfg.gpu_indices))
        num_gpus = len(model_cfg.gpu_indices)
    else:
        num_gpus = torch.cuda.device_count()

    # Inference config is used both for TurbomindEngineConfig and session_len
    inf_cfg_dict: Dict[str, Any] = inf_cfg.__dict__.copy()
    gen_max_new_tokens = inf_cfg_dict.pop("gen_max_new_tokens")
    max_prefill = inf_cfg_dict["max_prefill_token_num"]

    backend_cfg = TurbomindEngineConfig(
        tp=num_gpus,
        session_len=gen_max_new_tokens + max_prefill + 128,
        **inf_cfg_dict,
    )

    logger.info("Initializing model pipeline...")
    _PIPELINE = lm_pipeline(model_cfg.model_path, backend_cfg)
    logger.info("Model pipeline initialized.")
    return _PIPELINE


# ============================================================
# Response Processing (Parsing & Aggregation)
# ============================================================

class ResponseProcessor:
    """
    Extracts numeric answers from model text and aggregates multiple samples.
    """

    BOXED_PATTERN = re.compile(
        r"\\boxed\s*{\s*([-+]?[0-9]*\.?[0-9]+)\s*}",
        flags=re.MULTILINE,
    )

    @staticmethod
    def extract_boxed_answer(response: str) -> Optional[int]:
        """
        Extract the last \\boxed{} value as an integer modulo 100000.
        Falls back to the last plain integer in the text if no box is found.
        """
        if not response:
            return None

        matches = ResponseProcessor.BOXED_PATTERN.findall(response)
        content: Optional[str] = matches[-1] if matches else None

        # Fallback: grab last integer in the text if no boxed one.
        if content is None:
            fallback = re.findall(r"([-+]?[0-9]+)", response)
            if not fallback:
                return None
            content = fallback[-1]

        try:
            num = int(content)
        except ValueError:
            try:
                num = int(float(content))
                if math.isinf(num):
                    logger.warning("Parsed infinite value from %s", content)
                    return None
            except (ValueError, OverflowError):
                return None

        return num % 100_000

    @staticmethod
    def answer_aggregator(answers: Sequence[Optional[int]]) -> int:
        """
        Aggregate multiple numeric answers using a simple weighted voting scheme
        similar to the original code:

        - Ignore None and negative values.
        - Base weight = 1.0
        - Answers <= 20 or multiples of 100 get weight 0.5
        - Choose answer with highest total weight; on ties, prefer larger value.
        """
        valid_answers = [int(a) for a in answers if a is not None and int(a) >= 0]
        if not valid_answers:
            return 49  # default fallback from original code

        weights: Dict[int, float] = defaultdict(float)

        for a in valid_answers:
            weight = 1.0
            if a <= 20 or a % 100 == 0:
                weight = 0.5
            weights[a] += weight

        # Sort by weight desc, then value desc
        best_answer = sorted(
            weights.items(),
            key=lambda item: (-item[1], -item[0]),
        )[0][0]

        return best_answer % 100_000


# ============================================================
# Actor (Prompt Construction & Generation)
# ============================================================

class Actor:
    """
    High-level interface for generating answers given a question.
    """

    def __init__(self) -> None:
        self.actor_cfg = CONFIG.actor
        self.gen_cfg = self.actor_cfg.gen_cfg
        self.gen_config = GenerationConfig(**self.gen_cfg.__dict__)
        self.reasoning_effort = CONFIG.main_model.reasoning_effort
        self.enable_thinking = CONFIG.main_model.enable_thinking

    def _build_messages(self, question: str, weight: float = 1.0) -> List[List[Dict[str, str]]]:
        """
        Build a list of chat message lists for the pipeline, one per sample.
        """
        prompts: List[List[Dict[str, str]]] = []

        for pcfg in self.actor_cfg.prompt_list:
            repeat = int(pcfg.number * weight)
            if repeat <= 0:
                repeat = 1

            for _ in range(repeat):
                if pcfg.system:
                    prompts.append(
                        [
                            {"role": "system", "content": pcfg.system},
                            {"role": "user", "content": question + pcfg.user_suffix},
                        ]
                    )
                else:
                    prompts.append(
                        [
                            {"role": "user", "content": question + pcfg.user_suffix},
                        ]
                    )
        return prompts

    def generate(self, question: str, weight: float = 1.0):
        """
        Generate multiple responses for a given question.
        Returns: (responses, token_lengths, finish_reasons)
        """
        messages = self._build_messages(question, weight=weight)

        pipe = get_pipeline()
        outputs = pipe(
            messages,
            gen_config=self.gen_config,
            reasoning_effort=self.reasoning_effort,
            enable_thinking=self.enable_thinking,
            use_tqdm=True,
        )

        responses: List[str] = []
        token_lens: List[int] = []
        finish_reasons: List[str] = []

        for out in outputs:
            responses.append(out.text)
            token_lens.append(out.generate_token_len)
            finish_reasons.append(out.finish_reason)

        return responses, token_lens, finish_reasons


# ============================================================
# Prediction Function (Kaggle API)
# ============================================================

def predict(
    id_: pl.DataFrame,
    question: pl.DataFrame,
    answer: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Inference API function used by Kaggle.

    Parameters
    ----------
    id_ : pl.DataFrame
        Polars DataFrame containing a single id.
    question : pl.DataFrame
        Polars DataFrame containing a single question string.
    answer : pl.DataFrame | None
        Unused, kept for compatibility with evaluation server.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: "id", "answer"
    """
    q_id = id_.item(0)
    q_text = question.item(0)

    logger.info("=" * 60)
    logger.info("Question ID: %s", q_id)
    logger.info("Question: %s", q_text)
    logger.info("=" * 60)

    # In competition rerun mode, skip heavy generation.
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        logger.info("Rerun mode detected, returning dummy prediction.")
        return pl.DataFrame({"id": q_id, "answer": 0})

    # Global hard cutoff
    if TIME_MANAGER.after_final_cutoff():
        logger.warning("Final cutoff exceeded; returning default answer 49.")
        return pl.DataFrame({"id": q_id, "answer": 49})

    processor = ResponseProcessor()
    actor = Actor()

    # Weight adjustment near cutoff time
    weight = 1.0
    if TIME_MANAGER.should_downweight():
        weight = 0.66

    # Run generation
    responses, gen_token_lens, finish_reasons = actor.generate(q_text, weight=weight)

    # Parse predictions
    parsed_answers = [processor.extract_boxed_answer(r) for r in responses]
    final_prediction = processor.answer_aggregator(parsed_answers)

    logger.info("Raw parsed predictions: %s", parsed_answers)
    logger.info("Response token lengths: %s", gen_token_lens)
    logger.info("Finish reasons: %s", finish_reasons)
    logger.info("Final aggregated prediction: %s", final_prediction)
    logger.info("=" * 60)

    # Consume one cutoff step per question, like original pop()
    TIME_MANAGER.consume_iteration()

    return pl.DataFrame({"id": q_id, "answer": final_prediction})


# ============================================================
# Main Entrypoint (Local Evaluation / Server)
# ============================================================

def _set_random_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    # Seed everything (if configured)
    if CONFIG.seed is not None:
        logger.info("Setting random seed to %d", CONFIG.seed)
        _set_random_seeds(CONFIG.seed)

    exam_dataset_files = CONFIG.exam_dataset_files.split(",")
    inference_server = aimo_server.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        logger.info("Starting inference server in competition rerun mode...")
        inference_server.serve()
        return

    if CONFIG.use_server_for_eval:
        logger.info("Running local gateway for evaluation...")
        start_eval = time.time()

        inference_server.run_local_gateway(tuple(exam_dataset_files))

        # Local accuracy check
        ref_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
        sub_path = "/kaggle/working/submission.parquet"

        ref_df = pd.read_csv(ref_path)
        sub_df = pd.read_parquet(sub_path)

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

            logger.info(
                "Question ID: %s\tAnswer: %s\tPrediction: %s",
                q_id,
                true_ans,
                pred_ans,
            )

            total += 1
            if true_ans == pred_ans:
                correct += 1

        logger.info("=" * 60)
        logger.info("Accuracy: %d/%d", correct, total)
        end_eval = time.time()
        logger.info("Consumed time: %.2f mins", (end_eval - start_eval) / 60.0)


if __name__ == "__main__":
    main()