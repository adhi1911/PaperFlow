"""Utilities for path handling and security."""
from pathlib import Path
from typing import Union


def get_relative_path(path: Union[str, Path], project_root: Path = None) -> str:
    """Convert absolute path to relative for logging/display (security).
    
    Args:
        path: Absolute or relative path
        project_root: Project root for relative calculation. If None, uses current working directory.
    
    Returns:
        Relative path as string for safe logging
    """
    path = Path(path)
    
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)
    
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        # Fallback if path is not relative to project root
        return str(path)
