from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from src.common.config import (
    DATASET_PATH,
    KFOLD_N_SPLITS,
    KFOLD_RESULTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
)
from src.common.data import get_all_classes, load_dataset, split_dataset
from src.common.io_utils import save_json
from src.common.metrics import compute_metrics, select_best_threshold


MODEL_NAMES = ("logistic_regression", "linear_svm", "mlp")


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_model(
    model_name: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> OneVsRestClassifier:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=int(params.get("max_iter", 2_000)),
            C=float(params["C"]),
            random_state=random_state,
        )
        return OneVsRestClassifier(estimator, n_jobs=n_jobs)

    if model_name == "linear_svm":
        estimator = LinearSVC(
            class_weight="balanced",
            C=float(params["C"]),
            random_state=random_state,
        )
        return OneVsRestClassifier(estimator, n_jobs=n_jobs)

    if model_name == "mlp":
        hidden_layer_sizes = tuple(
            int(value)
            for value in params["hidden_layer_sizes"]
        )
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            batch_size=int(params["batch_size"]),
            max_iter=int(params["max_iter"]),
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=5,
            random_state=random_state,
            verbose=False,
        )

    raise ValueError(f"Modello non supportato: {model_name}")


def score_model(
    model_name: str,
    model: OneVsRestClassifier,
    x_values: Any,
) -> np.ndarray:
    if model_name in {"logistic_regression", "mlp"}:
        return model.predict_proba(x_values)

    if model_name == "linear_svm":
        return model.decision_function(x_values)

    raise ValueError(f"Modello non supportato: {model_name}")


def get_param_grid(
    model_name: str,
    c_values: list[float],
    mlp_hidden_layers: list[tuple[int, ...]],
    mlp_alpha_values: list[float],
) -> list[dict[str, Any]]:
    if model_name == "logistic_regression":
        return [
            {"C": c_value, "max_iter": 2_000}
            for c_value in c_values
        ]

    if model_name == "linear_svm":
        return [
            {"C": c_value}
            for c_value in c_values
        ]

    if model_name == "mlp":
        return [
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": alpha,
                "learning_rate_init": 0.001,
                "batch_size": 256,
                "max_iter": 60,
            }
            for hidden_layer_sizes in mlp_hidden_layers
            for alpha in mlp_alpha_values
        ]

    raise ValueError(f"Modello non supportato: {model_name}")


def make_param_key(params: dict[str, Any]) -> str:
    return "|".join(
        f"{key}={format_param_value(params[key])}"
        for key in sorted(params)
    )


def format_param_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)

    return str(value)


def make_threshold_values(model_name: str) -> np.ndarray:
    if model_name in {"logistic_regression", "mlp"}:
        return np.arange(0.10, 0.96, 0.05)

    if model_name == "linear_svm":
        return np.arange(-1.0, 1.01, 0.10)

    raise ValueError(f"Modello non supportato: {model_name}")


def build_targets(
    classes: list[str],
    *label_columns: pd.Series,
) -> tuple[MultiLabelBinarizer, list[np.ndarray]]:
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    targets = [mlb.transform(labels) for labels in label_columns]
    return mlb, targets


def summarize_folds(fold_results: list[dict[str, Any]]) -> pd.DataFrame:
    fold_df = pd.DataFrame(fold_results)

    metric_columns = [
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "hamming_loss",
        "subset_accuracy",
        "threshold",
    ]

    summary = (
        fold_df
        .groupby(["model", "param_key"], as_index=False)
        .agg(
            {
                "params": "first",
                **{
                    metric: ["mean", "std"]
                    for metric in metric_columns
                },
            }
        )
    )

    summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary.sort_values(
        ["model", "macro_f1_mean"],
        ascending=[True, False],
    )


def select_best_params(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    best_by_model: dict[str, dict[str, Any]] = {}

    for model_name, model_summary in summary.groupby("model"):
        best_row = model_summary.sort_values(
            ["macro_f1_mean", "micro_f1_mean"],
            ascending=[False, False],
        ).iloc[0]
        best_by_model[str(model_name)] = {
            "params": dict(best_row["params_first"]),
            "threshold": float(best_row["threshold_mean"]),
            "cv_macro_f1_mean": float(best_row["macro_f1_mean"]),
            "cv_micro_f1_mean": float(best_row["micro_f1_mean"]),
        }

    return best_by_model


def run_cross_validation(
    train_val_df: pd.DataFrame,
    y_train_val: np.ndarray,
    models: list[str],
    folds: int,
    max_features: int,
    c_values: list[float],
    mlp_hidden_layers: list[tuple[int, ...]],
    mlp_alpha_values: list[float],
    random_state: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    splitter = MultilabelStratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_indices = list(
        splitter.split(np.zeros(len(train_val_df)), y_train_val)
    )

    fold_results: list[dict[str, Any]] = []

    for model_name in models:
        threshold_values = make_threshold_values(model_name)

        for params in get_param_grid(
            model_name,
            c_values,
            mlp_hidden_layers,
            mlp_alpha_values,
        ):
            param_key = make_param_key(params)

            for fold_number, (train_index, validation_index) in enumerate(
                fold_indices,
                start=1,
            ):
                vectorizer = build_vectorizer(max_features)
                x_train = vectorizer.fit_transform(
                    train_val_df.iloc[train_index]["clean_description"]
                )
                x_validation = vectorizer.transform(
                    train_val_df.iloc[validation_index]["clean_description"]
                )

                y_train = y_train_val[train_index]
                y_validation = y_train_val[validation_index]

                missing_classes = int((y_train.sum(axis=0) == 0).sum())
                if missing_classes:
                    print(
                        "Warning: "
                        f"{missing_classes} classi assenti nel training "
                        f"del fold {fold_number} per {model_name} "
                        f"({param_key})."
                    )

                model = build_model(
                    model_name,
                    params,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )

                print(
                    f"{model_name} | {param_key} | "
                    f"fold {fold_number}/{folds}"
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=ConvergenceWarning,
                    )
                    model.fit(x_train, y_train)

                validation_scores = score_model(
                    model_name,
                    model,
                    x_validation,
                )
                threshold, _ = select_best_threshold(
                    y_validation,
                    validation_scores,
                    threshold_values,
                )
                metrics = compute_metrics(
                    y_validation,
                    validation_scores,
                    threshold,
                )

                fold_results.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "params": params,
                        "param_key": param_key,
                        "train_rows": len(train_index),
                        "validation_rows": len(validation_index),
                        "missing_train_classes": missing_classes,
                        "n_iter": (
                            int(model.n_iter_)
                            if hasattr(model, "n_iter_")
                            else None
                        ),
                        **asdict(metrics),
                    }
                )

    return fold_results


def fit_and_evaluate_final_models(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_val: np.ndarray,
    y_test: np.ndarray,
    best_by_model: dict[str, dict[str, Any]],
    max_features: int,
    random_state: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    final_results: list[dict[str, Any]] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, best in best_by_model.items():
        params = best["params"]
        threshold = float(best["threshold"])

        vectorizer = build_vectorizer(max_features)
        x_train_val = vectorizer.fit_transform(
            train_val_df["clean_description"]
        )
        x_test = vectorizer.transform(test_df["clean_description"])
        model_artifact: dict[str, Any] = {
            "vectorizer": vectorizer,
            "threshold": threshold,
            "params": params,
        }

        model = build_model(
            model_name,
            params,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        print(
            f"Training finale {model_name} con params={params} "
            f"e threshold CV medio={threshold:.4f}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
            )
            model.fit(x_train_val, y_train_val)

        test_scores = score_model(model_name, model, x_test)
        test_metrics = compute_metrics(y_test, test_scores, threshold)

        model_artifact["model"] = model
        joblib.dump(
            model_artifact,
            MODELS_DIR / f"{model_name}_kfold.joblib",
        )

        result = {
            "model": model_name,
            "params": params,
            "selected_threshold": threshold,
            "cv_macro_f1_mean": best["cv_macro_f1_mean"],
            "cv_micro_f1_mean": best["cv_micro_f1_mean"],
            "test_metrics": asdict(test_metrics),
            "training_rows": len(train_val_df),
            "test_rows": len(test_df),
            "n_iter": (
                int(model.n_iter_)
                if hasattr(model, "n_iter_")
                else None
            ),
        }
        save_json(
            RESULTS_DIR / f"{model_name}_kfold.json",
            result,
        )
        final_results.append(result)

    return final_results


def parse_positive_float_values(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not values:
        raise argparse.ArgumentTypeError(
            "Specificare almeno un valore."
        )

    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "Tutti i valori devono essere positivi."
        )

    return values


def parse_mlp_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not layers:
        raise argparse.ArgumentTypeError(
            "Specificare almeno un layer nascosto per l'MLP."
        )

    if any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "Tutti i layer nascosti dell'MLP devono essere positivi."
        )

    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "K-fold hyperparameter search for TF-IDF multilabel models. "
            "The same fold indices are reused across models and the final "
            "test set is kept as a holdout."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "all"),
        default="all",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=KFOLD_N_SPLITS,
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=TFIDF_MAX_FEATURES,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )
    parser.add_argument(
        "--c-values",
        type=parse_positive_float_values,
        default=parse_positive_float_values("0.5,1.0,2.0"),
        help=(
            "Comma-separated C values to test, for example "
            "0.25,0.5,1.0,2.0."
        ),
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=parse_mlp_hidden_layers,
        nargs="+",
        default=[
            parse_mlp_hidden_layers("128"),
            parse_mlp_hidden_layers("256"),
        ],
        help=(
            "One or more MLP architectures. Examples: "
            "--mlp-hidden-layers 128 256 or --mlp-hidden-layers 512,256."
        ),
    )
    parser.add_argument(
        "--mlp-alpha-values",
        type=parse_positive_float_values,
        default=parse_positive_float_values("0.001"),
        help="Comma-separated MLP L2 alpha values, for example 0.0001,0.001.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel jobs inside OneVsRestClassifier. Keep 1 on memory "
            "constrained machines."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds deve essere almeno 2.")

    models = (
        list(MODEL_NAMES)
        if args.model == "all"
        else [args.model]
    )

    KFOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    classes = get_all_classes(df)

    train_df, validation_df, test_df = split_dataset(
        df,
        random_state=args.random_state,
    )
    train_val_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    _, targets = build_targets(
        classes,
        train_val_df["labels"],
        test_df["labels"],
    )
    y_train_val, y_test = targets

    fold_results = run_cross_validation(
        train_val_df=train_val_df,
        y_train_val=y_train_val,
        models=models,
        folds=args.folds,
        max_features=args.max_features,
        c_values=args.c_values,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_alpha_values=args.mlp_alpha_values,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        KFOLD_RESULTS_DIR / "fold_metrics.csv",
        index=False,
    )

    summary = summarize_folds(fold_results)
    summary.to_csv(
        KFOLD_RESULTS_DIR / "hyperparameter_summary.csv",
        index=False,
    )

    best_by_model = select_best_params(summary)
    final_results = fit_and_evaluate_final_models(
        train_val_df=train_val_df,
        test_df=test_df,
        y_train_val=y_train_val,
        y_test=y_test,
        best_by_model=best_by_model,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    save_json(
        KFOLD_RESULTS_DIR / "kfold_experiment.json",
        {
            "dataset": str(args.dataset),
            "random_state": args.random_state,
            "folds": args.folds,
            "max_features": args.max_features,
            "c_values": args.c_values,
            "mlp_hidden_layers": args.mlp_hidden_layers,
            "mlp_alpha_values": args.mlp_alpha_values,
            "models": models,
            "n_classes": len(classes),
            "train_validation_rows": len(train_val_df),
            "test_rows": len(test_df),
            "best_by_model": best_by_model,
            "final_results": final_results,
        },
    )

    print()
    print("Migliori configurazioni K-fold:")
    for model_name, best in best_by_model.items():
        print(
            f"{model_name}: params={best['params']} | "
            f"cv_macro_f1={best['cv_macro_f1_mean']:.4f} | "
            f"threshold={best['threshold']:.4f}"
        )

    print()
    print("Metriche finali sul test holdout:")
    for result in final_results:
        metrics = result["test_metrics"]
        print(
            f"{result['model']}_kfold | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"micro_f1={metrics['micro_f1']:.4f}"
        )


if __name__ == "__main__":
    main(parse_args())
