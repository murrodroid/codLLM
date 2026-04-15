from pathlib import Path


def resolve_source_path(path_value: str, data_raw_dir: str) -> Path:
    """Resolve a source path relative to the raw-data directory when needed."""
    source_path = Path(path_value)
    if source_path.is_absolute() or source_path.exists():
        return source_path
    if not data_raw_dir:
        return source_path
    return Path(data_raw_dir) / source_path
