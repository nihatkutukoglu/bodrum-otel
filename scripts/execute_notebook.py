"""Bir notebooku proje kökünü çalışma dizini yaparak yerinde çalıştırır."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    notebook_path = args.notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Çalıştırıldı: {notebook_path}")


if __name__ == "__main__":
    main()
