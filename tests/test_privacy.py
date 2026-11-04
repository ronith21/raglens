"""Tests for PII redaction."""

import pytest

from raglens.privacy import redact_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mail jan.k@example.co.uk now", "Mail [EMAIL] now"),
        ("IBAN DE89 3704 0044 0532 0130 00 please", "IBAN [IBAN] please"),
        ("IBAN DE89370400440532013000.", "IBAN [IBAN]."),
        ("Card 4111 1111 1111 1111 declined", "Card [CARD] declined"),
        ("Call +49 30 1234567 or 030 1234567", "Call [PHONE] or [PHONE]"),
    ],
)
def test_redacts_personal_data(text: str, expected: str) -> None:
    """E-mails, IBANs, Luhn-valid cards and phone numbers are masked."""
    assert redact_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "What is the CET1 ratio in 2024?",
        "Deposits are covered up to 100,000 euros.",
        "Is the 2019-2025 period covered?",
        "What does a pacs.008 message represent?",
        "Card 1234 5678 9012 3456 fails the Luhn check",
    ],
)
def test_leaves_ordinary_banking_text_alone(text: str) -> None:
    """Years, amounts, message types and non-Luhn digit runs are not masked."""
    assert redact_pii(text) == text
