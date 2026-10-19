"""Tests for the project path constants in ``libs.core.global_paths``."""

from libs.core.global_paths import (
    DOCUMENTS_FOLDER,
    LIBS_FOLDER,
    ROOT_FOLDER,
    SRC_FOLDER,
)


def test_root_folder_is_the_project_root() -> None:
    assert (ROOT_FOLDER / "pyproject.toml").is_file()
    assert ROOT_FOLDER.name == "Ragument"


def test_src_folder_is_under_root() -> None:
    assert SRC_FOLDER == ROOT_FOLDER / "src"
    assert (SRC_FOLDER / "libs").is_dir()


def test_documents_folder_is_at_the_root() -> None:
    assert DOCUMENTS_FOLDER == ROOT_FOLDER / "documents"


def test_libs_folder_is_under_src() -> None:
    assert LIBS_FOLDER == SRC_FOLDER / "libs"
