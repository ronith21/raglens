"""Evaluation harness: dataset, metrics, runner, synthetic QA and ablation benchmark."""

from raglens.eval.dataset import QAExample, load_dataset, save_dataset
from raglens.eval.runner import DEFAULT_KS, EvalReport, QuestionResult, evaluate
from raglens.eval.synthetic import generate_synthetic_qa

__all__ = [
    "DEFAULT_KS",
    "EvalReport",
    "QAExample",
    "QuestionResult",
    "evaluate",
    "generate_synthetic_qa",
    "load_dataset",
    "save_dataset",
]
