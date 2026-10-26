"""Train and evaluate the BB84 intrusion detector.

Run from the backend directory:

    python -m app.detection.train --sessions 24000

The generated dataset contains sessions across all noise channels, nominal
noise levels, qubit counts, Eve strategies, and strengths.  The classifier
only sees the Observation feature vector; Eve ground truth is used solely as
the training label and for subgroup reporting after evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from app.config import (
    ARTIFACT_DIR,
    BLOCK_COUNT,
    MODEL_NAME,
    MODEL_VERSION,
    STANDARD_QBER_THRESHOLD,
)
from app.detection.features import FEATURE_NAMES, extract_features
from app.simulation.bb84 import (
    Engine,
    EveMode,
    SimulationRequest,
    simulate_session,
)
from app.simulation.channels import ChannelType


@dataclass(slots=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    qber: np.ndarray
    modes: np.ndarray
    channels: np.ndarray
    noise: np.ndarray
    qubits: np.ndarray
    strengths: np.ndarray


def _new_estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )


def generate_dataset(sessions: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    features = np.empty((sessions, len(FEATURE_NAMES)), dtype=np.float64)
    labels = np.empty(sessions, dtype=np.int8)
    qber = np.empty(sessions, dtype=np.float64)
    modes = np.empty(sessions, dtype=object)
    channels = np.empty(sessions, dtype=object)
    noise_values = np.empty(sessions, dtype=np.int16)
    qubit_values = np.empty(sessions, dtype=np.int32)
    strengths = np.empty(sessions, dtype=np.int16)

    channel_pool = np.array(
        [channel.value for channel in ChannelType], dtype=object
    )
    started = time.perf_counter()

    for index in range(sessions):
        qubits = int(rng.integers(1, 51)) * 100
        noise = int(rng.integers(0, 21))
        channel = ChannelType(str(rng.choice(channel_pool)))
        mode = str(
            rng.choice(
                ["off", "steady", "burst"], p=[0.45, 0.275, 0.275]
            )
        )
        strength = int(rng.integers(1, 101)) if mode != "off" else 100

        result = simulate_session(
            SimulationRequest(
                eve=EveMode(mode),
                eve_strength=float(strength),
                noise=float(noise),
                qubits=qubits,
                channel=channel,
                engine=Engine.CLASSICAL,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
        )
        observation = result.observation
        features[index] = extract_features(observation)
        labels[index] = 1 if result.attack_present else 0
        qber[index] = observation.qber
        modes[index] = result.attack_mode
        channels[index] = observation.channel.value
        noise_values[index] = noise
        qubit_values[index] = qubits
        strengths[index] = strength

        if (index + 1) % 2000 == 0 or index + 1 == sessions:
            elapsed = time.perf_counter() - started
            rate = (index + 1) / max(elapsed, 1e-9)
            print(
                f"  generated {index + 1:>6}/{sessions} sessions "
                f"({rate:6.0f}/s)",
                flush=True,
            )

    return Dataset(
        features=features,
        labels=labels,
        qber=qber,
        modes=modes,
        channels=channels,
        noise=noise_values,
        qubits=qubit_values,
        strengths=strengths,
    )


def _metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, threshold: float | None
) -> dict[str, object]:
    labels = labels.astype(np.int8)
    predictions = predictions.astype(bool)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions.astype(np.int8), labels=[0, 1]
    ).ravel()
    samples = int(labels.size)
    positives = int(labels.sum())
    negatives = samples - positives

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / samples if samples else 0.0
    fpr = fp / negatives if negatives else 0.0
    fnr = fn / positives if positives else 0.0

    return {
        "samples": samples,
        "positives": positives,
        "negatives": negatives,
        "threshold": None if threshold is None else round(float(threshold), 6),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_positive_rate": round(float(fpr), 6),
        "false_negative_rate": round(float(fnr), 6),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = probabilities >= threshold
    metrics = _metrics_from_predictions(labels, predictions, threshold)
    if np.unique(labels).size > 1:
        metrics["roc_auc"] = round(
            float(roc_auc_score(labels, probabilities)), 6
        )
        metrics["brier_score"] = round(
            float(brier_score_loss(labels, probabilities)), 6
        )
    else:
        metrics["roc_auc"] = None
        metrics["brier_score"] = None
    return metrics


def choose_threshold(
    labels: np.ndarray, probabilities: np.ndarray, max_fpr: float = 0.02
) -> float:
    """Pick the F1-maximising threshold with a strict false-positive budget."""
    best: tuple[float, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions.astype(np.int8), labels=[0, 1]
        ).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        if fpr > max_fpr:
            continue
        if best is None or f1 > best[1] + 1e-12 or (
            abs(f1 - best[1]) <= 1e-12
            and abs(float(threshold) - 0.5) < abs(best[0] - 0.5)
        ):
            best = (float(threshold), float(f1))

    if best is not None:
        return best[0]

    # No threshold met the false-positive budget: choose the best F1 overall.
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = probabilities >= threshold
        f1 = f1_score(labels, predictions.astype(np.int8), zero_division=0)
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold


def _subgroup(
    name: str,
    mask: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[str, dict[str, object]] | None:
    count = int(mask.sum())
    if count == 0:
        return None
    return name, _probability_metrics(labels[mask], probabilities[mask], threshold)


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    qber: np.ndarray,
    modes: np.ndarray,
    channels: np.ndarray,
    qubits: np.ndarray,
    strengths: np.ndarray,
) -> dict[str, object]:
    ml_metrics = _probability_metrics(labels, probabilities, threshold)
    standard_predictions = qber > STANDARD_QBER_THRESHOLD
    standard_metrics = _metrics_from_predictions(
        labels, standard_predictions, None
    )

    standard_positive = standard_predictions
    ml_positive = probabilities >= threshold
    additional_caught = int(np.sum((~standard_positive) & ml_positive & (labels == 1)))
    false_alarms_avoided = int(
        np.sum(standard_positive & (~ml_positive) & (labels == 0))
    )

    by_mode = {}
    for mode in ("none", "steady", "burst"):
        subgroup = _subgroup(
            mode, modes == mode, labels, probabilities, threshold
        )
        if subgroup is not None:
            by_mode[mode] = subgroup[1]

    by_channel = {}
    for channel in ChannelType:
        subgroup = _subgroup(
            channel.value,
            channels == channel.value,
            labels,
            probabilities,
            threshold,
        )
        if subgroup is not None:
            by_channel[channel.value] = subgroup[1]

    low_qber_attacks = _subgroup(
        "qber_at_or_below_threshold",
        (labels == 1) & (qber <= STANDARD_QBER_THRESHOLD),
        labels,
        probabilities,
        threshold,
    )
    benign_high_qber = _subgroup(
        "benign_above_threshold",
        (labels == 0) & (qber > STANDARD_QBER_THRESHOLD),
        labels,
        probabilities,
        threshold,
    )

    by_qubits = {}
    for label, lower, upper in (
        ("100-499", 100, 499),
        ("500-999", 500, 999),
        ("1000-2499", 1000, 2499),
        ("2500-5000", 2500, 5000),
    ):
        mask = (qubits >= lower) & (qubits <= upper)
        if not mask.any():
            continue
        by_qubits[label] = {
            "ml": _probability_metrics(
                labels[mask], probabilities[mask], threshold
            ),
            "standard_rule": _metrics_from_predictions(
                labels[mask],
                (qber[mask] > STANDARD_QBER_THRESHOLD),
                None,
            ),
        }

    by_strength = {}
    for label, lower, upper in (
        ("1-10", 1, 10),
        ("11-25", 11, 25),
        ("26-50", 26, 50),
        ("51-100", 51, 100),
    ):
        mask = (
            (labels == 1)
            & (strengths >= lower)
            & (strengths <= upper)
        )
        if mask.any():
            by_strength[label] = _probability_metrics(
                labels[mask], probabilities[mask], threshold
            )

    return {
        "ml": ml_metrics,
        "standard_rule": standard_metrics,
        "ml_advantage": {
            "additional_attacks_caught": additional_caught,
            "false_alarms_avoided": false_alarms_avoided,
        },
        "by_mode": by_mode,
        "by_channel": by_channel,
        "by_qubit_count": by_qubits,
        "by_eve_strength": by_strength,
        "low_qber_attacks": None if low_qber_attacks is None else low_qber_attacks[1],
        "benign_high_qber": None if benign_high_qber is None else benign_high_qber[1],
    }


def _print_summary(metrics: dict[str, object]) -> None:
    ml = metrics["ml"]  # type: ignore[index]
    standard = metrics["standard_rule"]  # type: ignore[index]
    advantage = metrics["ml_advantage"]  # type: ignore[index]
    print("\nHeld-out test performance")
    auc = ml["roc_auc"]
    auc_text = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"
    print(
        "  ML        : "
        f"accuracy={ml['accuracy']:.4f} precision={ml['precision']:.4f} "
        f"recall={ml['recall']:.4f} f1={ml['f1']:.4f} "
        f"fpr={ml['false_positive_rate']:.4f} auc={auc_text}"
    )
    print(
        "  11% rule  : "
        f"accuracy={standard['accuracy']:.4f} precision={standard['precision']:.4f} "
        f"recall={standard['recall']:.4f} f1={standard['f1']:.4f} "
        f"fpr={standard['false_positive_rate']:.4f}"
    )
    print(
        "  ML advantage: "
        f"+{advantage['additional_attacks_caught']} attacks caught, "
        f"{advantage['false_alarms_avoided']} false alarms avoided"
    )
    low = metrics.get("low_qber_attacks")
    if low:
        print(
            "  Attacks with QBER <= 11%: "
            f"{low['samples']} sessions, recall={low['recall']:.4f}, "
            f"fpr={low['false_positive_rate']:.4f}"
        )


def train(
    sessions: int = 24000,
    seed: int = 20261026,
    output_dir: Path = ARTIFACT_DIR,
) -> dict[str, object]:
    started = time.perf_counter()
    print(f"Generating {sessions} labelled BB84 sessions (seed={seed})...")
    dataset = generate_dataset(sessions, seed)
    features = dataset.features
    labels = dataset.labels

    indices = np.arange(sessions)
    train_idx, temp_idx = _split(indices, labels, test_size=0.30, seed=seed + 1)
    val_idx, test_idx = _split(
        temp_idx, labels[temp_idx], test_size=0.50, seed=seed + 2
    )

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    print(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"(attack rate {labels.mean():.3f})"
    )

    print("Training calibrated gradient-boosted classifier...")
    calibration = CalibratedClassifierCV(
        estimator=_new_estimator(), method="sigmoid", cv=3
    )
    calibration.fit(x_train, y_train)
    validation_probabilities = calibration.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, validation_probabilities)
    print(f"Selected decision threshold: {threshold:.2f}")

    final_training_size = len(train_idx) + len(val_idx)
    final_features = np.concatenate([x_train, x_val], axis=0)
    final_labels = np.concatenate([y_train, y_val], axis=0)
    print(f"Refitting final model on {final_training_size} sessions...")
    final_model = CalibratedClassifierCV(
        estimator=_new_estimator(), method="sigmoid", cv=3
    )
    final_model.fit(final_features, final_labels)

    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
        dataset.qber[test_idx],
        dataset.modes[test_idx],
        dataset.channels[test_idx],
        dataset.qubits[test_idx],
        dataset.strengths[test_idx],
    )

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    duration = time.perf_counter() - started
    metadata: dict[str, object] = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "trained_at": trained_at,
        "feature_names": list(FEATURE_NAMES),
        "decision_threshold": round(float(threshold), 6),
        "standard_threshold_percent": STANDARD_QBER_THRESHOLD,
        "block_count": BLOCK_COUNT,
        "dataset": {
            "sessions": sessions,
            "seed": seed,
            "attack_rate": round(float(labels.mean()), 6),
            "split": {
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
                "final_training": final_training_size,
            },
            "qubit_range": [
                int(dataset.qubits.min()),
                int(dataset.qubits.max()),
            ],
            "noise_range_percent": [
                int(dataset.noise.min()),
                int(dataset.noise.max()),
            ],
            "channels": [channel.value for channel in ChannelType],
            "eve_modes": ["off", "steady", "burst"],
            "eve_strength_range_percent": [
                int(dataset.strengths[dataset.labels == 1].min()),
                int(dataset.strengths[dataset.labels == 1].max()),
            ],
        },
        "model": {
            "type": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
            "calibration": "sigmoid",
            "cv_folds": 3,
            "hyperparameters": {
                "learning_rate": 0.07,
                "max_iter": 350,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 40,
                "l2_regularization": 1.0,
                "early_stopping": True,
                "n_iter_no_change": 25,
            },
        },
        "metrics": metrics,
        "training_seconds": round(duration, 3),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "notes": [
            "Features are post-sift observations plus nominal channel context.",
            "Eve mode, strength, and attack labels are never model inputs.",
            "The 11% rule uses the session-average QBER and can be evaded by "
            "burst attacks whose errors are concentrated in a few blocks.",
            "This model is trained on simulated BB84 data, not on a deployed "
            "QKD link; recalibrate before operational use.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "bb84_detector.joblib"
    card_path = output_dir / "model_card.json"
    joblib.dump(
        {"model": final_model, "metadata": metadata},
        artifact_path,
        compress=3,
    )
    card_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    _print_summary(metrics)  # type: ignore[arg-type]
    print(f"\nSaved model : {artifact_path}")
    print(f"Saved card  : {card_path}")
    print(f"Training time: {duration:.1f}s")
    return metadata


def _split(
    indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", type=int, default=24000, help="number of simulated sessions"
    )
    parser.add_argument(
        "--seed", type=int, default=20261026, help="dataset generation seed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR,
        help="artifact output directory",
    )
    args = parser.parse_args()
    if args.sessions < 1000:
        parser.error("--sessions must be at least 1000")
    train(sessions=args.sessions, seed=args.seed, output_dir=args.output)


if __name__ == "__main__":
    main()
