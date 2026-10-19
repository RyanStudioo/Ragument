"""Tests for the hash-based database sync in ``libs.rag_db.setup_db``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from libs.rag_db.setup_db import (
    CONTENT_HASH_KEY,
    DocumentSyncError,
    hash_file,
    sync_documents,
)


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings so tests never download a model."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:8]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def write_document(folder: Path, name: str, content: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def folders(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "documents", tmp_path / "db"


def sync(folders: tuple[Path, Path], embeddings: FakeEmbeddings):
    documents_folder, db_path = folders
    return sync_documents(
        embeddings=embeddings,
        documents_folder=documents_folder,
        db_path=db_path,
    )


def stored_entries(db_path: Path) -> dict[str, dict]:
    """Read raw Chroma entries so tests can verify what was persisted."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="documents",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(db_path),
    )
    result = store.get(include=["metadatas", "documents"])
    return {
        doc_id: {"metadata": metadata, "content": content}
        for doc_id, metadata, content in zip(
            result["ids"], result["metadatas"], result["documents"], strict=False
        )
    }


def test_new_document_is_added(folders: tuple[Path, Path]) -> None:
    documents_folder, db_path = folders
    write_document(documents_folder, "alpha.md", "Alpha body.\n")

    report = sync(folders, FakeEmbeddings())

    assert report.added == ["alpha"]
    assert report.updated == []
    assert report.deleted == []
    entries = stored_entries(db_path)
    assert set(entries) == {"alpha"}
    assert entries["alpha"]["content"] == "Alpha body."


def test_unchanged_document_is_not_reembedded(folders: tuple[Path, Path]) -> None:
    documents_folder, _ = folders
    write_document(documents_folder, "alpha.md", "Alpha body.\n")
    sync(folders, FakeEmbeddings())

    embeddings = FakeEmbeddings()
    report = sync(folders, embeddings)

    assert report.unchanged == ["alpha"]
    assert report.added == report.updated == report.deleted == []
    assert embeddings.embedded_texts == []


def test_changed_document_is_reembedded_and_overwritten(
    folders: tuple[Path, Path],
) -> None:
    documents_folder, db_path = folders
    write_document(documents_folder, "alpha.md", "Old body.\n")
    sync(folders, FakeEmbeddings())

    write_document(documents_folder, "alpha.md", "New body.\n")
    report = sync(folders, FakeEmbeddings())

    assert report.updated == ["alpha"]
    assert report.added == report.deleted == []
    entries = stored_entries(db_path)
    assert set(entries) == {"alpha"}
    assert entries["alpha"]["content"] == "New body."
    assert entries["alpha"]["metadata"][CONTENT_HASH_KEY] == hash_file(
        documents_folder / "alpha.md"
    )


def test_removed_document_is_deleted(folders: tuple[Path, Path]) -> None:
    documents_folder, db_path = folders
    alpha = write_document(documents_folder, "alpha.md", "Alpha.\n")
    write_document(documents_folder, "beta.md", "Beta.\n")
    sync(folders, FakeEmbeddings())

    alpha.unlink()
    report = sync(folders, FakeEmbeddings())

    assert report.deleted == ["alpha"]
    assert report.added == report.updated == []
    assert set(stored_entries(db_path)) == {"beta"}


def test_hash_change_alone_triggers_reembed(folders: tuple[Path, Path]) -> None:
    documents_folder, _ = folders
    write_document(documents_folder, "alpha.md", "Same visible body.\n")
    sync(folders, FakeEmbeddings())

    # Same parsed body (trailing whitespace), different raw bytes.
    write_document(documents_folder, "alpha.md", "Same visible body.  \n")
    report = sync(folders, FakeEmbeddings())

    assert report.updated == ["alpha"]
    assert report.unchanged == []


def test_metadata_includes_content_hash(folders: tuple[Path, Path]) -> None:
    documents_folder, db_path = folders
    path = write_document(
        documents_folder,
        "alpha.md",
        "---\ntags:\n  - one\n  - two\nversion: 2\n---\nBody.\n",
    )
    sync(folders, FakeEmbeddings())

    metadata = stored_entries(db_path)["alpha"]["metadata"]
    assert metadata[CONTENT_HASH_KEY] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["version"] == 2
    assert metadata["tags"] == '["one", "two"]'
    assert metadata["file_name"] == "alpha.md"


def test_duplicate_ids_raise(folders: tuple[Path, Path]) -> None:
    documents_folder, _ = folders
    write_document(documents_folder, "alpha.md", "One.\n")
    write_document(documents_folder, "alpha.txt", "Two.\n")

    with pytest.raises(DocumentSyncError, match="Duplicate document id"):
        sync(folders, FakeEmbeddings())


def test_missing_documents_folder_clears_database(
    folders: tuple[Path, Path],
) -> None:
    documents_folder, db_path = folders
    write_document(documents_folder, "alpha.md", "Alpha.\n")
    sync(folders, FakeEmbeddings())

    for path in documents_folder.iterdir():
        path.unlink()
    documents_folder.rmdir()

    report = sync(folders, FakeEmbeddings())

    assert report.deleted == ["alpha"]
    assert stored_entries(db_path) == {}


def test_nested_folders_are_scanned(folders: tuple[Path, Path]) -> None:
    documents_folder, _ = folders
    write_document(documents_folder / "lore", "alpha.md", "Alpha.\n")

    report = sync(folders, FakeEmbeddings())

    assert report.added == ["alpha"]


def test_second_sync_with_no_changes_is_a_noop(folders: tuple[Path, Path]) -> None:
    documents_folder, _ = folders
    write_document(documents_folder, "alpha.md", "Alpha.\n")
    write_document(documents_folder, "beta.md", "Beta.\n")
    first = sync(folders, FakeEmbeddings())
    second = sync(folders, FakeEmbeddings())

    assert first.added == ["alpha", "beta"]
    assert second.unchanged == ["alpha", "beta"]
    assert second.added == second.updated == second.deleted == []
