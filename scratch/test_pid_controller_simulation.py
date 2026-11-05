import os
import sys
from pathlib import Path
import json
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratch.evaluate_step1_ensembles import main as eval_main
from face_similarity.store import load_store
from scratch.train_subcenter_pauc_regularized import predict_visual_logits, evaluate_metrics
import joblib
from scipy.stats import rankdata

class BumbleELOPIDController:
    def __init__(
        self,
        val_scores: np.ndarray,
        target_rate: float = 0.225,
        kp: float = 1.5,
        state_path: str = "scratch/bumble_pid_state_test.json"
    ):
        self.target_q = 1.0 - target_rate  # 0.775
        self.target_rate = target_rate
        self.kp = kp
        self.state_path = Path(state_path)

        # Anchor static quality floor (68th pct) and ceiling (85th pct) from validation set
        self.tau_floor = float(np.percentile(val_scores, 68.0))
        self.tau_ceil = float(np.percentile(val_scores, 85.0))

        # Cold-start pre-warming from last 250 validation scores
        init_scores = val_scores[-250:].tolist()
        init_thresh = float(np.percentile(init_scores, self.target_q * 100.0))
        init_decisions = [1 if s >= init_thresh else 0 for s in init_scores[-200:]]
        self.score_window = deque(init_scores, maxlen=250)
        self.decision_window = deque(init_decisions, maxlen=200)
        self.consecutive_likes = 0
        self._save_state()

    def decide(self, z_score: float) -> tuple[bool, dict]:
        self.score_window.append(float(z_score))
        realized_rate_200 = float(np.mean(self.decision_window))
        recent_100_likes = sum(list(self.decision_window)[-99:])

        # 1. Proportional feedback on quantile target
        q_t = float(np.clip(self.target_q + self.kp * (realized_rate_200 - self.target_rate), 0.760, 0.795))
        tau_live = float(np.clip(np.quantile(self.score_window, q_t), self.tau_floor, self.tau_ceil))

        # 2. Streak Damping: 4th consecutive like requires 90th percentile; 5th is blocked
        if self.consecutive_likes == 3:
            tau_live = max(tau_live, float(np.quantile(self.score_window, 0.90)))

        # 3. Final Decision with 100-card ELO circuit breaker (< 24 likes in rolling 100)
        swipe_right = (
            (z_score >= tau_live)
            and (recent_100_likes < 24)
            and (self.consecutive_likes < 4)
        )

        decision_int = 1 if swipe_right else 0
        self.decision_window.append(decision_int)
        self.consecutive_likes = (self.consecutive_likes + 1) if swipe_right else 0
        self._save_state()

        return swipe_right, {
            "z_score": round(z_score, 4),
            "tau_live": round(tau_live, 4),
            "quantile_target": round(q_t, 4),
            "rolling_200_rate": round(float(np.mean(self.decision_window)), 4),
            "rolling_100_likes": recent_100_likes + decision_int,
            "streak": self.consecutive_likes,
        }

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "score_window": list(self.score_window),
            "decision_window": list(self.decision_window),
            "consecutive_likes": self.consecutive_likes
        }))


def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")

    print("=" * 80)
    print("SIMULATING PRODUCTION PID CONFORMAL CONTROLLER ON 3,481 HOLDOUT 2 CARDS")
    print("=" * 80)

    # 1. Load Stores
    ref_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    sig_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))
    clip_30k = np.load(str(root / "embeddings" / "clip_store_bumble_30k_pure.npz"))

    ref_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    sig_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))
    clip_holdout = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))

    y_test = (ref_holdout.ratings >= 50.0).astype(int)
    n_test = len(y_test)
    n_likes = int(y_test.sum())

    # Pre-compute ensemble scores on validation set (26,500 to 29,582)
    val_slice = slice(26500, 29582)
    val_face = ref_30k.embeddings[val_slice]
    val_sig = sig_30k['embeddings'][val_slice]
    val_clip = clip_30k['embeddings'][val_slice]
    val_mlp_X = np.hstack([val_face, val_clip])

    ckpt_26k = str(root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt")
    ckpt_30k = str(root / "models" / "subcenter_phase2" / "best_subcenter_pauc_30k_full.pt")

    val_preds_26k = predict_visual_logits(ckpt_26k, val_sig, val_face)
    val_preds_30k = predict_visual_logits(ckpt_30k, val_sig, val_face)

    mlp_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    mlp = mlp_obj['estimator'] if isinstance(mlp_obj, dict) else mlp_obj
    val_preds_mlp = mlp.predict(val_mlp_X)

    # Rank-average validation scores to form ensemble baseline
    r_val_mlp = rankdata(val_preds_mlp) / len(val_preds_mlp)
    r_val_26k = rankdata(val_preds_26k) / len(val_preds_26k)
    r_val_30k = rankdata(val_preds_30k) / len(val_preds_30k)
    val_scores = 0.40 * r_val_mlp + 0.30 * r_val_26k + 0.30 * r_val_30k

    print(f"Validation pool: {len(val_scores)} profiles. Tau Floor (68th): {np.percentile(val_scores, 68):.4f}, Tau Ceil (85th): {np.percentile(val_scores, 85):.4f}")

    # Compute test scores (Holdout 2)
    test_face = ref_holdout.embeddings
    test_sig = sig_holdout['embeddings']
    test_clip = clip_holdout['embeddings']
    test_mlp_X = np.hstack([test_face, test_clip])

    test_preds_26k = predict_visual_logits(ckpt_26k, test_sig, test_face)
    test_preds_30k = predict_visual_logits(ckpt_30k, test_sig, test_face)
    test_preds_mlp = mlp.predict(test_mlp_X)

    r_test_mlp = rankdata(test_preds_mlp) / n_test
    r_test_26k = rankdata(test_preds_26k) / n_test
    r_test_30k = rankdata(test_preds_30k) / n_test
    test_scores = 0.40 * r_test_mlp + 0.30 * r_test_26k + 0.30 * r_test_30k

    # Run online PID controller simulation through all 3,481 cards
    controller = BumbleELOPIDController(val_scores, target_rate=0.225, kp=1.5)

    decisions = []
    log_data = []
    for i in range(n_test):
        s = test_scores[i]
        dec, info = controller.decide(s)
        decisions.append(int(dec))
        info["card_index"] = i
        info["actual_label"] = int(y_test[i])
        log_data.append(info)

    decisions = np.array(decisions)
    total_right = int(decisions.sum())
    realized_swipe_rate = total_right / n_test * 100
    likes_caught = int(np.sum((decisions == 1) & (y_test == 1)))
    recall = likes_caught / n_likes * 100
    hit_rate = likes_caught / total_right * 100 if total_right > 0 else 0

    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"Total Cards Evaluated: {n_test}")
    print(f"Total Right Swipes: {total_right} / {n_test} ({realized_swipe_rate:.2f}%) [Nominal target: 22.5%]")
    print(f"Actual Likes Caught: {likes_caught} / {n_likes} ({recall:.1f}% Recall)")
    print(f"Precision / Hit Rate: {hit_rate:.1f}% ({likes_caught} likes out of {total_right} right swipes)")

    # Check rolling 100 stats
    rolling_100_sums = [sum(decisions[max(0, i-99):i+1]) for i in range(n_test)]
    max_in_100 = max(rolling_100_sums)
    print(f"Maximum right swipes in any 100-card rolling window: {max_in_100} (Strict limit is 24)")

    # Save summary
    df_sim = pd.DataFrame(log_data)
    out_csv = root / "results" / "pid_controller_simulation_holdout2.csv"
    df_sim.to_csv(out_csv, index=False)
    print(f"Saved step-by-step log to {out_csv}")


if __name__ == "__main__":
    main()
