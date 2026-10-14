"""Tests for the document parser in ``libs.core.documents``."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from libs.core.documents import DocumentParseError, parse_document

FRONT_MATTER_DOC = """\
---
question: "How do Wars work?"
answer: |
  Wars are resolved in two stages.

  **Stage 1:** Battle.
---
# How do Wars work?

Wars are resolved in two stages.

**Stage 1:** Battle.
"""


def write_document(tmp_path: Path, content: str, name: str = "sample.md") -> Path:
    """Write ``content`` to a file inside ``tmp_path`` and return its path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_transfers_front_matter_into_metadata(tmp_path: Path) -> None:
    doc = parse_document(write_document(tmp_path, FRONT_MATTER_DOC))

    assert doc.metadata["question"] == "How do Wars work?"
    assert doc.metadata["answer"] == (
        "Wars are resolved in two stages.\n\n**Stage 1:** Battle.\n"
    )


def test_page_content_is_body_without_front_matter(tmp_path: Path) -> None:
    doc = parse_document(write_document(tmp_path, FRONT_MATTER_DOC))

    assert doc.page_content.startswith("# How do Wars work?")
    assert doc.page_content.endswith("**Stage 1:** Battle.")
    assert "---" not in doc.page_content
    assert doc.page_content == FRONT_MATTER_DOC.split("---\n", 2)[2].strip()


def test_returns_langchain_document(tmp_path: Path) -> None:
    doc = parse_document(write_document(tmp_path, FRONT_MATTER_DOC))

    assert isinstance(doc, Document)


def test_source_and_file_name_metadata(tmp_path: Path) -> None:
    path = write_document(tmp_path, FRONT_MATTER_DOC, name="war-mechanics.md")

    doc = parse_document(path)

    assert doc.metadata["source"] == str(path.resolve())
    assert doc.metadata["file_name"] == "war-mechanics.md"


def test_file_metadata_wins_over_front_matter(tmp_path: Path) -> None:
    content = """\
---
source: /fake/source.md
file_name: fake.md
---
Body.
"""
    path = write_document(tmp_path, content, name="real.md")

    doc = parse_document(path)

    assert doc.metadata["source"] == str(path.resolve())
    assert doc.metadata["file_name"] == "real.md"


def test_file_without_front_matter(tmp_path: Path) -> None:
    path = write_document(tmp_path, "# Title\n\nJust body text.\n")

    doc = parse_document(path)

    assert doc.page_content == "# Title\n\nJust body text."
    assert doc.metadata == {
        "source": str(path.resolve()),
        "file_name": "sample.md",
    }


def test_empty_front_matter(tmp_path: Path) -> None:
    path = write_document(tmp_path, "---\n---\n\nBody.\n")

    doc = parse_document(path)

    assert doc.page_content == "Body."
    assert doc.metadata == {
        "source": str(path.resolve()),
        "file_name": "sample.md",
    }


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    doc = parse_document(write_document(tmp_path, "\n\n  Body.  \n\n"))

    assert doc.page_content == "Body."


def test_front_matter_must_start_at_first_line(tmp_path: Path) -> None:
    content = "\n---\nquestion: hi\n---\nBody.\n"

    doc = parse_document(write_document(tmp_path, content))

    assert doc.metadata == {
        "source": str((tmp_path / "sample.md").resolve()),
        "file_name": "sample.md",
    }
    assert doc.page_content.startswith("---")


def test_non_string_front_matter_values_are_preserved(tmp_path: Path) -> None:
    content = """\
---
tags:
  - wars
  - claims
version: 2
draft: true
---
Body.
"""
    doc = parse_document(write_document(tmp_path, content))

    assert doc.metadata["tags"] == ["wars", "claims"]
    assert doc.metadata["version"] == 2
    assert doc.metadata["draft"] is True


def test_utf8_content_is_preserved(tmp_path: Path) -> None:
    content = '---\nquestion: "¿Cómo funcionan las guerras? 戦争"\n---\nCuerpo ✨\n'

    doc = parse_document(write_document(tmp_path, content))

    assert doc.metadata["question"] == "¿Cómo funcionan las guerras? 戦争"
    assert doc.page_content == "Cuerpo ✨"


def test_crlf_line_endings_are_supported(tmp_path: Path) -> None:
    content = "---\r\nquestion: hi\r\n---\r\n# Title\r\n\r\nBody.\r\n"

    doc = parse_document(write_document(tmp_path, content))

    assert doc.metadata["question"] == "hi"
    assert doc.page_content == "# Title\n\nBody."


def test_accepts_string_path(tmp_path: Path) -> None:
    path = write_document(tmp_path, FRONT_MATTER_DOC)

    doc = parse_document(str(path))

    assert doc.metadata["file_name"] == "sample.md"


def test_expands_tilde_in_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_document(tmp_path, FRONT_MATTER_DOC, name="notes.md")

    doc = parse_document("~/notes.md")

    assert doc.metadata["source"] == str((tmp_path / "notes.md").resolve())
    assert doc.page_content.startswith("# How do Wars work?")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentParseError, match="does not exist"):
        parse_document(tmp_path / "missing.md")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentParseError, match="does not exist"):
        parse_document(tmp_path)


def test_unclosed_front_matter_raises(tmp_path: Path) -> None:
    path = write_document(tmp_path, "---\nquestion: hi\n# heading\n")

    with pytest.raises(DocumentParseError, match="no closing"):
        parse_document(path)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = write_document(tmp_path, "---\nquestion: [unclosed\n---\nbody\n")

    with pytest.raises(DocumentParseError, match="invalid YAML front matter"):
        parse_document(path)


@pytest.mark.parametrize("front_matter", ["- a\n- b", "just a string"])
def test_non_mapping_front_matter_raises(
    tmp_path: Path, front_matter: str
) -> None:
    path = write_document(tmp_path, f"---\n{front_matter}\n---\nbody\n")

    with pytest.raises(DocumentParseError, match="must be a YAML mapping"):
        parse_document(path)
