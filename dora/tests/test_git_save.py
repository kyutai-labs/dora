"""Git snapshot validation must reject malformed state without overwriting files."""

from types import SimpleNamespace

import pytest

from dora import git_save


def test_malformed_git_status_is_rejected(tmp_path, monkeypatch):
    main = SimpleNamespace(dora=SimpleNamespace(grid_package="dora.tests.integ.grids"))
    monkeypatch.setattr(git_save, "run_command", lambda _: "unexpected status fields")
    with pytest.raises(AssertionError, match="Invalid git status entry"):
        git_save.check_repo_clean(tmp_path, main)


def test_assign_clone_preserves_unexpected_regular_file(tmp_path):
    code = tmp_path / "code"
    code.write_text("existing data")
    xp = SimpleNamespace(dora=SimpleNamespace(git_save=True), code_folder=code)
    with pytest.raises(AssertionError, match="code folder should be symlink or folder"):
        git_save.assign_clone(xp, tmp_path / "clone")
    assert code.read_text() == "existing data"
    assert not code.is_symlink()
