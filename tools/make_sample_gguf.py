from __future__ import annotations

import argparse
from pathlib import Path

from gguf_explorer.gguf import write_sample_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny BF16/Q8_0 GGUF sample")
    parser.add_argument("path", nargs="?", default="samples/tiny-bf16-q8_0.gguf")
    args = parser.parse_args()
    path = write_sample_gguf(Path(args.path))
    print(path)


if __name__ == "__main__":
    main()

