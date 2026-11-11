"""Public, immutable upload contract for local regulation navigation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class DocumentInput:
    id: str
    name: str
    content: bytes
    metadata: dict | None = None
