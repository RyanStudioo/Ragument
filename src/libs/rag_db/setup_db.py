"""Build and synchronise the vector database from the documents folder.

The sync is content-hash based:

* a file whose id is not in the database is embedded and added;
* a file whose id exists but whose hash changed is re-embedded and overwritten;
* a file whose id and hash both match is left untouched;
* database entries whose id no longer matches a file are deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from libs.core.config_parse import ConfigParse
from libs.core.documents import parse_document
from libs.core.global_paths import DOCUMENTS_FOLDER, ROOT_FOLDER

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "documents"
CONTENT_HASH_KEY = "content_hash"

_READ_SIZE = 1024 * 1024


class DocumentSyncError(ValueError):
    """Raised when the documents cannot be synchronised into the database."""


@dataclass(frozen=True)
class SyncReport:
    """Summary of what a database sync changed."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Number of document files processed."""
        return len(self.added) + len(self.updated) + len(self.unchanged)


def hash_file(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(_READ_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_documents(
    *,
    embeddings: Embeddings,
    documents_folder: str | Path = DOCUMENTS_FOLDER,
    db_path: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> SyncReport:
    """Synchronise the documents folder into a persistent Chroma database.

    Only changed documents are re-embedded. Returns a report describing which
    document ids were added, updated, left unchanged, and deleted.
    """
    desired = _load_documents(Path(documents_folder))

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(Path(db_path)),
    )
    existing = store.get(include=["metadatas"])
    existing_ids: list[str] = existing.get("ids") or []
    existing_metadatas: list[dict[str, Any] | None] = (
        existing.get("metadatas") or []
    )
    existing_hashes = {
        doc_id: (metadata or {}).get(CONTENT_HASH_KEY)
        for doc_id, metadata in zip(existing_ids, existing_metadatas, strict=False)
    }

    to_add: list[Document] = []
    to_update: list[Document] = []
    unchanged: list[str] = []

    for doc_id, (content_hash, document) in desired.items():
        if doc_id not in existing_hashes:
            to_add.append(document)
        elif existing_hashes[doc_id] != content_hash:
            to_update.append(document)
        else:
            unchanged.append(doc_id)

    stale_ids = [doc_id for doc_id in existing_ids if doc_id not in desired]

    if to_add:
        logger.info("Adding %d document(s)", len(to_add))
        store.add_documents(to_add, ids=_document_ids(to_add))
    if to_update:
        logger.info("Re-embedding %d changed document(s)", len(to_update))
        store.update_documents(_document_ids(to_update), to_update)
    if stale_ids:
        logger.info("Deleting %d stale document(s)", len(stale_ids))
        store.delete(ids=stale_ids)

    return SyncReport(
        added=sorted(document.id for document in to_add),
        updated=sorted(document.id for document in to_update),
        unchanged=sorted(unchanged),
        deleted=sorted(stale_ids),
    )


def setup_db(config: ConfigParse | None = None) -> SyncReport:
    """Build or refresh the database configured through ``ConfigParse``."""
    config = config or ConfigParse()

    db_path = Path(config.db_path).expanduser()
    if not db_path.is_absolute():
        db_path = ROOT_FOLDER / db_path

    return sync_documents(
        embeddings=build_embeddings(config),
        documents_folder=DOCUMENTS_FOLDER,
        db_path=db_path,
    )


def build_embeddings(config: ConfigParse) -> Embeddings:
    """Create the embedding model selected by ``EMBEDDING_PROVIDER``."""
    if config.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.embedding_model_name,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
        )

    return _SentenceTransformerEmbeddings(config.embedding_model_name)


class _SentenceTransformerEmbeddings(Embeddings):
    """Minimal LangChain ``Embeddings`` adapter for sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise DocumentSyncError(
                "EMBEDDING_PROVIDER=local requires the 'sentence-transformers' "
                "package. Install it with: uv add sentence-transformers"
            ) from error
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _load_documents(
    documents_folder: Path,
) -> dict[str, tuple[str, Document]]:
    """Parse every file in ``documents_folder`` and key it by document id."""
    if not documents_folder.exists():
        logger.warning("Documents folder does not exist: %s", documents_folder)
        return {}
    if not documents_folder.is_dir():
        raise DocumentSyncError(f"Documents path is not a folder: {documents_folder}")

    desired: dict[str, tuple[str, Document]] = {}
    paths = sorted(path for path in documents_folder.rglob("*") if path.is_file())

    for path in paths:
        document = parse_document(path)
        if document.id in desired:
            previous_path = desired[document.id][1].metadata["source"]
            raise DocumentSyncError(
                f"Duplicate document id {document.id!r}: "
                f"{previous_path} and {path}. File names must be unique."
            )

        content_hash = hash_file(path)
        document.metadata = _sanitize_metadata(document.metadata)
        document.metadata[CONTENT_HASH_KEY] = content_hash
        desired[document.id] = (content_hash, document)

    return desired


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only values Chroma can store, JSON-encoding complex ones."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[str(key)] = value
        else:
            clean[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
    return clean


def _document_ids(documents: list[Document]) -> list[str]:
    return [document.id for document in documents if document.id is not None]


def main() -> None:
    """Run the database setup and print a summary."""
    report = setup_db()

    print(
        f"Documents: {report.total} "
        f"({len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.unchanged)} unchanged, {len(report.deleted)} deleted)"
    )
    for label, ids in (
        ("Added", report.added),
        ("Updated", report.updated),
        ("Deleted", report.deleted),
    ):
        if ids:
            print(f"{label}: {', '.join(ids)}")


if __name__ == "__main__":
    main()
