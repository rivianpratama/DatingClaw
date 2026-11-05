"""Model-bound probability calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.provenance import hash_id_list, stable_hash


class CalibrationUnavailable(ValueError):
    pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_x = np.exp(values[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def fit_platt(
    logits: np.ndarray,
    y: np.ndarray,
    *,
    model_hash: str,
    transform_hash: str,
    recipe_hash: str,
    calibration_ids: np.ndarray,
) -> "Calibrator":
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y).reshape(-1)
    if len(s) != len(labels) or len(s) == 0:
        raise ValueError("logits and labels must align and be non-empty")
    if not np.isfinite(s).all():
        raise ValueError("logits must be finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("calibration labels must be binary")
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise CalibrationUnavailable("calibration requires both classes")
    target = 2.0 * labels.astype(np.float64) - 1.0

    def objective(parameters: np.ndarray) -> float:
        log_a, b = parameters
        a = np.exp(np.clip(log_a, -12.0, 12.0))
        margins = target * (a * s + b)
        return float(np.logaddexp(0.0, -margins).mean())

    result = minimize(objective, x0=np.array([0.0, 0.0]), method="L-BFGS-B")
    if not np.isfinite(result.fun):
        raise CalibrationUnavailable("calibration objective is non-finite")
    log_a, b = result.x
    a = float(np.exp(np.clip(log_a, -12.0, 12.0)))
    if not np.isfinite(a) or a <= 0 or not np.isfinite(b):
        raise CalibrationUnavailable("calibration parameters are invalid")
    return Calibrator(
        a=a,
        b=float(b),
        model_hash=model_hash,
        transform_hash=transform_hash,
        recipe_hash=recipe_hash,
        calibration_ids_hash=hash_id_list(calibration_ids),
        n_pos=n_pos,
        n_neg=n_neg,
        objective=float(result.fun),
        converged=bool(result.success),
        optimizer=str(result.message),
    )


@dataclass(frozen=True)
class Calibrator:
    a: float
    b: float
    model_hash: str
    transform_hash: str
    recipe_hash: str
    calibration_ids_hash: str
    n_pos: int
    n_neg: int
    objective: float
    converged: bool
    optimizer: str

    def apply(
        self,
        logits: np.ndarray,
        *,
        model_hash: str | None = None,
        transform_hash: str | None = None,
        recipe_hash: str | None = None,
    ) -> np.ndarray:
        if model_hash is not None and model_hash != self.model_hash:
            raise ValueError("calibrator model hash mismatch")
        if transform_hash is not None and transform_hash != self.transform_hash:
            raise ValueError("calibrator transform hash mismatch")
        if recipe_hash is not None and recipe_hash != self.recipe_hash:
            raise ValueError("calibrator recipe hash mismatch")
        return sigmoid(self.a * np.asarray(logits, dtype=np.float64) + self.b).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": C.CALIBRATOR_SCHEMA,
            "a": self.a,
            "b": self.b,
            "model_hash": self.model_hash,
            "transform_hash": self.transform_hash,
            "recipe_hash": self.recipe_hash,
            "calibration_ids_hash": self.calibration_ids_hash,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "objective": self.objective,
            "converged": self.converged,
            "optimizer": self.optimizer,
            "digest": stable_hash(self.__dict__),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Calibrator":
        if value.get("schema") != C.CALIBRATOR_SCHEMA:
            raise ValueError("unsupported calibrator schema")
        fields = {key: value[key] for key in (
            "a", "b", "model_hash", "transform_hash", "recipe_hash", "calibration_ids_hash",
            "n_pos", "n_neg", "objective", "converged", "optimizer",
        )}
        return cls(**fields)


def prior_shift(p: np.ndarray, pi_from: float, pi_to: float) -> np.ndarray:
    if not (0 < pi_from < 1 and 0 < pi_to < 1):
        raise ValueError("prevalences must be in (0,1)")
    p = np.asarray(p, dtype=np.float64)
    a = p * (pi_to / pi_from)
    b = (1.0 - p) * ((1.0 - pi_to) / (1.0 - pi_from))
    return (a / (a + b)).astype(np.float32)
