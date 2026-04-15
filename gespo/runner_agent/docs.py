from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocRef:
    name: str
    path: Path


def list_docs(docs_root: Path) -> list[DocRef]:
    docs_root = docs_root.expanduser()
    if not docs_root.exists():
        return []
    docs: list[DocRef] = []
    for p in sorted(docs_root.glob("GESPO_AGENT_WORKFLOW_*.md")):
        docs.append(DocRef(name=p.name, path=p))
    return docs


def resolve_docs(docs_root: Path, selectors: Iterable[str]) -> list[Path]:
    docs_root = docs_root.expanduser()
    sel = [s.strip() for s in selectors if s and s.strip()]
    if not sel:
        return []
    if len(sel) == 1 and sel[0].lower() == "all":
        return [d.path for d in list_docs(docs_root)]

    out: list[Path] = []
    for raw in sel:
        # Allow comma-separated.
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            p = (docs_root / part).expanduser()
            if p.exists():
                out.append(p)
                continue
            # Allow passing absolute/relative paths too.
            p2 = Path(part).expanduser()
            if p2.exists():
                out.append(p2)
                continue
            raise FileNotFoundError(f"Doc not found: {part} (docs_root={docs_root})")
    # De-dupe while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def load_docs_text(
    doc_paths: Iterable[Path],
    *,
    max_total_chars: int = 60000,
    max_chars_per_doc: int = 15000,
) -> str:
    chunks: list[str] = []
    used = 0
    for p in doc_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(errors="ignore")
        if max_chars_per_doc > 0 and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n...<truncated>...\n"
        header = f"\n\n===== {p} =====\n"
        block = header + text
        if max_total_chars > 0 and used + len(block) > max_total_chars:
            remaining = max(0, max_total_chars - used)
            if remaining > 0:
                chunks.append(block[:remaining] + "\n...<context truncated>...\n")
            break
        chunks.append(block)
        used += len(block)
    return "".join(chunks).strip()

