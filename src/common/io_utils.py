from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def converter(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Tipo non serializzabile: {type(item).__name__}")

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=converter,
        ),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))