"""Parse document files into LangChain ``Document`` objects with metadata."""

from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document

FRONT_MATTER_DELIMITER = "---"


class DocumentParseError(ValueError):
    """Raised when a document file cannot be parsed."""


def parse_document(file_path: str | Path) -> Document:
    """Parse a single document file into a ``Document``."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise DocumentParseError(f"Document file does not exist: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocumentParseError(f"Could not read {path}: {error}") from error

    front_matter, body = _split_front_matter(raw, path)

    metadata: dict[str, Any] = {
        **front_matter,
        "source": str(path.resolve()),
        "file_name": path.name,
    }

    return Document(page_content=body, metadata=metadata, id=path.stem)


def _split_front_matter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    """Split a leading YAML front matter block from the document body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        return {}, text.strip()

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONT_MATTER_DELIMITER:
            # Keep the newline that ended the last front matter line, so YAML
            # block scalars (e.g. ``answer: |``) keep their final line break.
            front_matter_text = "\n".join(lines[1:index]) + "\n"
            body = "\n".join(lines[index + 1 :])
            return _parse_front_matter(front_matter_text, path), body.strip()

    raise DocumentParseError(
        f"{path}: front matter starts with '{FRONT_MATTER_DELIMITER}' "
        f"but has no closing '{FRONT_MATTER_DELIMITER}'"
    )


def _parse_front_matter(front_matter_text: str, path: Path) -> dict[str, Any]:
    """Parse the YAML front matter text into a metadata mapping."""
    try:
        parsed = yaml.safe_load(front_matter_text)
    except yaml.YAMLError as error:
        raise DocumentParseError(f"{path}: invalid YAML front matter: {error}") from error

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise DocumentParseError(
            f"{path}: front matter must be a YAML mapping, got {type(parsed).__name__}"
        )
    return parsed
