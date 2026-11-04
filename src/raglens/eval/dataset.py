"""Evaluation dataset model and JSONL I/O.

``relevant_doc_ids`` holds document *slugs* (the source file stem, e.g. ``01_gradient_descent``),
which every chunk carries in ``metadata["doc_slug"]``. Slugs are readable and stay stable when a
document's text is edited, unlike the content-hash ``doc_id``. An empty list marks a question the
corpus cannot answer; the pipeline is expected to refuse it.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class QAExample(BaseModel):
    """One evaluation question."""

    id: str = ""
    question: str = Field(..., min_length=1)
    answer: str = ""
    relevant_doc_ids: list[str] = Field(default_factory=list)
    synthetic: bool = False
    kind: str = Field(
        default="",
        description="Question type, e.g. direct, paraphrase, multi or unanswerable "
        "(defaults to 'unanswerable' when there are no relevant documents, else 'direct')",
    )

    def model_post_init(self, __context: object) -> None:
        """Fill in a default kind."""
        if not self.kind:
            self.kind = "direct" if self.relevant_doc_ids else "unanswerable"

    @property
    def answerable(self) -> bool:
        """True when at least one relevant document exists."""
        return bool(self.relevant_doc_ids)


def load_dataset(path: str | Path) -> list[QAExample]:
    """Load a JSONL dataset, assigning ``q<line>`` ids to rows without one."""
    examples: list[QAExample] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        example = QAExample(**row)
        if not example.id:
            example.id = f"q{lineno}"
        examples.append(example)
    return examples


def save_dataset(examples: list[QAExample], path: str | Path) -> None:
    """Write examples as JSONL (one object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(e.model_dump(), ensure_ascii=False) + "\n" for e in examples),
        encoding="utf-8",
    )
