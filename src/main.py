"""Benchmark entry point.

Run ``uv run -m src.main`` to print metrics and held-out predictions.
"""

from scripts.evaluate_benchmark import main


if __name__ == "__main__":
    main()
