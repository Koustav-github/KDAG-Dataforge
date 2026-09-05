def main() -> None:
    """Print the reproduction commands from README.md's "How to reproduce" section.

    This project is a pipeline of scripts, not a single CLI action, so there is
    no default "run" behaviour to implement here — just a pointer to the right
    commands, in the right order.
    """
    print(
        "bdh-surgery is a pipeline of scripts, not a single command.\n"
        "Reproduction commands (see README.md, \"How to reproduce\"):\n"
        "\n"
        "  uv run pytest tests/                       # 44 tests\n"
        "  uv run python -m bdh_surgery.sweep          # ~60 min, writes artifacts/runs.csv\n"
        "  uv run python -m bdh_surgery.export         # writes web/public/data/\n"
        "  uv run python scripts/measure_locality.py   # locality table; needs node on PATH\n"
        "  uv run python scripts/probe_diagnostic.py   # M under pivot vs. own-direction probes\n"
    )
