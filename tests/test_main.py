"""Integration tests for the multilingual contextual spellchecker CLI.

The tests cover:
- CLI help output.
- Proper handling of missing required arguments.
- End‑to‑end correction of a small input file.
"""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

# Import the click command defined by the package.
# Adjust the import path if the project structure differs.
from multilingual_spellchecker.cli import main as cli_main


def test_cli_help():
    """Running the CLI with ``--help`` should exit cleanly and contain usage."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"])
    # Click returns exit_code 0 for help.
    assert result.exit_code == 0, f"CLI exited with {result.exit_code}, output: {result.output}"
    # The help text should contain the command name and a short description.
    assert "Usage:" in result.output
    assert "multilingual" in result.output.lower()


def test_cli_missing_required_arguments():
    """The CLI should error when required arguments are omitted."""
    runner = CliRunner()
    result = runner.invoke(cli_main, [])
    # Click returns non‑zero exit code for missing required options.
    assert result.exit_code != 0
    # The error message should hint at the missing option.
    assert "Missing option" in result.output or "Error:" in result.output


def test_end_to_end_correction(tmp_path: Path):
    """Run the full workflow: write a misspelled sentence, invoke the CLI,
    and verify that the output file contains the corrected sentence.
    """
    # Prepare a tiny input file.
    input_text = "I hav a speling error."
    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text, encoding="utf-8")

    # Path for the CLI to write the corrected output.
    output_file = tmp_path / "output.txt"

    # Choose a lightweight model identifier that is guaranteed to be
    # available in the test environment.  The project ships a tiny
    # ``dummy`` model for unit testing; if it does not exist we fall back
    # to a no‑op identifier that the CLI treats as a stub.
    model_name = os.getenv("SPELLCHECKER_MODEL", "dummy-model")

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "--model",
            model_name,
            "--input",
            str(input_file),
            "--output",
            str(output_file),
        ],
        timeout=180,
    )

    # The command should succeed.
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # The output file must exist and contain the corrected sentence.
    assert output_file.is_file(), "Output file was not created"

    corrected_text = output_file.read_text(encoding="utf-8").strip()
    # The expected correction for the test sentence.
    expected = "I have a spelling error."
    assert corrected_text == expected, f"Expected '{expected}', got '{corrected_text}'"
