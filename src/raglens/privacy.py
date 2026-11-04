"""PII redaction applied before questions are written to the query log.

Masks e-mail addresses, IBANs, payment card numbers (Luhn-checked) and phone numbers. Redaction
is deliberately conservative: it is a safety net for logs, not a substitute for a data protection
review.
"""

import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Country code, two check digits, then uppercase alphanumerics in groups of four (spaces optional),
# with a shorter final group: "DE89 3704 0044 0532 0130 00" or "DE89370400440532013000".
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# Phone numbers start with +, 0 (national/trunk prefix) or "(" - this avoids masking year ranges.
_PHONE_RE = re.compile(r"(?<!\w)(?:\+|0|\()[\d ()/.-]{7,}\d\b")


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum used by payment card numbers."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _iban_like(match: re.Match[str]) -> str:
    compact = match.group(0).replace(" ", "")
    return (
        "[IBAN]"
        if 15 <= len(compact) <= 34 and any(c.isdigit() for c in compact[4:])
        else (match.group(0))
    )


def _card_like(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_valid(digits) else match.group(0)


def _phone_like(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[PHONE]" if 8 <= len(digits) <= 15 else match.group(0)


def redact_pii(text: str) -> str:
    """Replace e-mails, IBANs, card numbers and phone numbers with placeholders.

    >>> redact_pii("Mail jan@example.com about DE89 3704 0044 0532 0130 00")
    'Mail [EMAIL] about [IBAN]'
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _IBAN_RE.sub(_iban_like, text)
    text = _CARD_RE.sub(_card_like, text)
    return _PHONE_RE.sub(_phone_like, text)
