from __future__ import annotations

import json

import pandas as pd

from src.common.config import RESULTS_DIR


def main() -> None:
    rows = []

    for model_name in (
        "logistic_regression",
        "logistic_regression_kfold",
        "linear_svm",
        "linear_svm_kfold",
        "mlp",
        "mlp_kfold",
        "bert",
    ):
        path = RESULTS_DIR / f"{model_name}.json"

        if not path.exists():
            continue

        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model_name,
                **result["test_metrics"],
            }
        )

    if not rows:
        raise FileNotFoundError(
            "Nessun risultato trovato in artifacts/results."
        )

    comparison = pd.DataFrame(rows).set_index("model")
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv")

    print(comparison.to_string())


if __name__ == "__main__":
    main()
