"""Evaluation dataset: JSONL rows of {question, answer, relevant_doc_ids}."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["lexical", "paraphrase", "unanswerable", "synthetic"]


class QAItem(BaseModel):
    """One labelled question. ``relevant_doc_ids`` are document keys (file stems)."""

    id: str
    question: str
    answer: str = ""
    relevant_doc_ids: list[str] = Field(default_factory=list)
    type: QuestionType = "lexical"
    split: Literal["dev", "test"] = "test"

    @property
    def answerable(self) -> bool:
        """Questions without relevant documents should be refused."""
        return bool(self.relevant_doc_ids)


def load_dataset(path: str | Path, split: str | None = None) -> list[QAItem]:
    """Load a JSONL file, optionally keeping only ``dev`` or ``test`` rows."""
    items: list[QAItem] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = QAItem(**json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid row: {exc}") from exc
            if split is None or item.split == split:
                items.append(item)
    if not items:
        raise ValueError(f"No rows loaded from {path} (split={split})")
    return items
