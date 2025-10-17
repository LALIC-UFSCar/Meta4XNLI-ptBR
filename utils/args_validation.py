"""Utility functions for argument validation."""
import argparse
from pathlib import Path


def validate_file_extension(file_path: str, expected_extension: str) -> Path:
    """Validates the file extension of a given file path.

    Args:
        file_path (str): Path to the file to be validated.
        expected_extension (str): Expected file extension (e.g., '.jsonl').

    Raises:
        argparse.ArgumentTypeError: If the file extension does not match the expected extension.

    Returns:
        Path: The validated file path.
    """
    path = Path(file_path)
    if not path.name.lower().endswith(expected_extension.lower()):
        raise argparse.ArgumentTypeError(
            f"File {file_path} does not have required extension \
            '{expected_extension}'!"
        )
    return path
