from src.utils.paths import path_relative_to_repo, repo_root


def test_repo_root_contains_src():
    root = repo_root()
    assert (root / "src").is_dir()
    assert (root / "context" / "structure.md").is_file()


def test_path_relative_to_repo():
    assert path_relative_to_repo("configs", "default.yaml").is_file()
