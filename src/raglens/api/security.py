"""HTTP security helpers: upload sanitising, admin trust, rate limiting, origin checks, headers."""

import ipaddress
import re
import secrets
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

#: Host that Starlette's in-process TestClient reports; only reachable from inside the process
#: (embedded UI mode and tests), never over the network.
IN_PROCESS_CLIENT = "testclient"

MAX_FILENAME_STEM = 100
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])$", re.IGNORECASE)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
#: JSON API responses never need to load anything.
API_CSP = "default-src 'none'; frame-ancestors 'none'"
#: Swagger UI loads its bundled JS/CSS from this origin and runs one inline bootstrap script.
DOCS_CSP = (
    "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
)
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def safe_upload_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe basename that keeps its extension.

    Strips POSIX and Windows directory parts, normalises Unicode, replaces anything outside
    ``[A-Za-z0-9._ -]``, removes leading dots, shortens long stems and defuses Windows device
    names such as ``CON``. Raises ``ValueError`` when nothing usable is left.
    """
    name = PureWindowsPath(PurePosixPath(raw).name).name
    name = unicodedata.normalize("NFKC", name)
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip(" .")
    suffix = Path(name).suffix.lower()
    stem = name[: -len(suffix)] if suffix else name
    stem = stem.strip(" ._")[:MAX_FILENAME_STEM].rstrip(" .")
    if not stem or not suffix:
        raise ValueError(f"Invalid file name {raw[:80]!r}")
    if _WINDOWS_RESERVED.match(stem):
        stem = f"_{stem}"
    return f"{stem}{suffix}"


def validate_upload_content(name: str, data: bytes) -> None:
    """Reject files whose bytes do not match their extension (``ValueError``)."""
    if not data:
        raise ValueError(f"{name!r} is empty")
    if name.endswith(".pdf"):
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"{name!r} is not a PDF file")
        return
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name!r} is not UTF-8 text") from exc
    if "\x00" in text:
        raise ValueError(f"{name!r} contains binary data")


def keys_match(given: str | None, expected: str) -> bool:
    """Constant-time API-key comparison that also handles non-ASCII header values."""
    return secrets.compare_digest((given or "").encode("utf-8"), expected.encode("utf-8"))


def is_trusted_client(host: str | None, networks: list[str] | tuple[str, ...]) -> bool:
    """True when the client address is inside ``networks`` (or is the in-process client)."""
    if host == IN_PROCESS_CLIENT:
        return True
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in ipaddress.ip_network(net, strict=False) for net in networks)


def clean_request_id(value: str | None) -> str | None:
    """Return the client's request id if it is short and harmless, else ``None``."""
    return value if value and _REQUEST_ID_RE.fullmatch(value) else None


def strip_control_chars(text: str) -> str:
    """Remove ASCII control characters except tab, newline and carriage return."""
    return _CONTROL_CHARS.sub("", text)


def origin_allowed(origin: str | None, host: str | None, allowed: list[str]) -> bool:
    """Browser CSRF check: allow requests without ``Origin``, same-origin, or listed origins."""
    if not origin:
        return True
    if origin == "null":
        return False
    if origin.rstrip("/") in {a.rstrip("/") for a in allowed}:
        return True
    return bool(host) and urlsplit(origin).netloc.lower() == (host or "").lower()


def public_source_path(path: str) -> str:
    """Hide server directory layout: relative paths stay, absolute ones become the file name."""
    if not path:
        return path
    candidate = Path(path)
    if not candidate.is_absolute():
        return path
    try:
        return candidate.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return candidate.name


class RateLimiter:
    """Thread-safe sliding-window limiter keyed by client and bucket (in memory, per process)."""

    def __init__(self, window_s: float = 60.0, max_keys: int = 10_000) -> None:
        self.window_s = window_s
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> float | None:
        """Record one request; return seconds to wait if ``limit`` is exceeded, else ``None``."""
        if limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self.max_keys:
                    self._evict(now)
                hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        """Drop expired keys; if still full, drop the oldest half."""
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
            del self._hits[key]
        if len(self._hits) >= self.max_keys:
            for key in sorted(self._hits, key=lambda k: self._hits[k][-1])[: self.max_keys // 2]:
                del self._hits[key]
