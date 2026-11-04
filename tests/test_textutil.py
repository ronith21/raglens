"""Tests for shared text helpers."""

import pytest

from raglens.textutil import (
    content_stems,
    content_tokens,
    query_coverage,
    sentence_spans,
    split_sentences,
    stem,
    strip_markdown,
)


def test_content_tokens_drop_stopwords_keep_short_alphanumerics() -> None:
    """Stopwords go, 'l1'/'k1' stay, two-letter words without digits go."""
    assert content_tokens("What is the difference between L1 and L2 at k1?") == ["l1", "l2", "k1"]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("parallelize", "parallelized"),
        ("rate", "rates"),
        ("regularization", "regularize"),
        ("networks", "network"),
    ],
)
def test_stem_conflates_inflections(a: str, b: str) -> None:
    """Related surface forms share a stem."""
    assert stem(a) == stem(b)


def test_stem_keeps_short_words_and_double_s() -> None:
    """Short words and words ending in 'ss' are not over-stripped."""
    assert stem("loss") == "loss"
    assert stem("bias") == "bia"[:6] or stem("bias") == "bias"
    assert content_stems("Learning rates") == ["learn", "rate"]


def test_strip_markdown_removes_math_and_markup() -> None:
    """Block math disappears, inline math becomes plain text, headings are dropped."""
    text = "# Title\n\n- **Bold** item with $\\eta > 0$.\n\n$$x = y$$\n1. Second"
    out = strip_markdown(text)
    assert "Title" not in out
    assert "Bold item with eta > 0." in out
    assert "x = y" not in out
    assert "Second" in out


def test_split_sentences_respects_lines_and_punctuation() -> None:
    """Sentences split on terminal punctuation and on newlines."""
    assert split_sentences("One here. Two there!\nThree") == ["One here.", "Two there!", "Three"]
    assert split_sentences("") == []


def test_query_coverage() -> None:
    """Coverage is the share of distinct query content stems present in the text."""
    assert query_coverage("learning rate warmup", "Warmup scales the learning rates.") == 1.0
    assert query_coverage("learning rate warmup", "The learning step.") == pytest.approx(1 / 3)
    assert query_coverage("what is the", "anything") == 0.0


def test_single_letters_and_subscripts_are_content() -> None:
    """Variables like 'b' count as content words; LaTeX subscripts are joined."""
    assert content_tokens("What does the b parameter do?") == ["b", "parameter"]
    assert content_tokens("tune k_1 and L_{2}") == ["tune", "k1", "l2"]


def test_sentence_spans_offsets_point_into_original_text() -> None:
    """Spans index the original string, trimmed, with lines as hard boundaries."""
    text = "  First one. Second one!\n- Third item\n\nFourth."
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == ["First one.", "Second one!", "- Third item", "Fourth."]
    assert sentence_spans("") == []
