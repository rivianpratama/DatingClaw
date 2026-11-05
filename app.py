from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Ensure temporary files and model caches run strictly on D: drive to protect C: drive space
if Path("D:/").exists():
    d_temp = Path(r"D:\BumbleTemp")
    d_temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(d_temp)
    os.environ["TEMP"] = str(d_temp)
    os.environ["TMP"] = str(d_temp)
    os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
    os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
    os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from face_similarity.clip_embedding import get_clip_embedding
from face_similarity.embedding import get_face_embedding
from face_similarity.regressor import (
    load_regressor,
    predict_multimodal_rating,
    predict_rating,
)
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store
from face_similarity.experimental_setup import EXPERIMENTAL_SETUPS
from face_similarity.preference import load_preference_model, preference_probability, features_from_prediction

app = FastAPI(title="DatingClaw Image Analysis API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard Regressor/Epoch configurations
BASE_MODELS_CONFIG = {
    "bumble_30k_pure": {
        "name": "Bumble 30k Pure (My Real Swipes)",
        "regressor": "models/rating_regressor_bumble_30k_pure.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_30k_pure.joblib",
        "store": "embeddings/reference_store_bumble_30k_pure.npz",
        "face_weight": 0.60
    },
    "bumble_20k_pure": {
        "name": "Bumble 20k Pure (My Real Swipes)",
        "regressor": "models/rating_regressor_bumble_20k_pure.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_20k_pure.joblib",
        "store": "embeddings/reference_store_bumble_20k_pure.npz",
        "face_weight": 0.60
    },
    "bumble_10k_pure": {
        "name": "Bumble 10k Pure (My Real Swipes)",
        "regressor": "models/rating_regressor_bumble_10k_pure.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_10k_pure.joblib",
        "store": "embeddings/reference_store_bumble_10k_pure.npz",
        "face_weight": 0.60
    },
    "original": {
        "name": "Original Model (7k)",
        "regressor": "models/rating_regressor.joblib",
        "multimodal": "models/rating_regressor_multimodal.joblib",
        "store": "embeddings/reference_store.npz",
        "face_weight": 0.50
    },
    "chandra_7k": {
        "name": "Chandra Labels (7k Method)",
        "regressor": "models/rating_regressor_chandra_7k.joblib",
        "multimodal": "models/rating_regressor_multimodal_chandra_7k.joblib",
        "store": "embeddings/reference_store_chandra_7k.npz",
        "face_weight": 0.50
    },
    "round1": {
        "name": "Round 1 Model (Combined)",
        "regressor": "models/rating_regressor_bumble_combined.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_combined.joblib",
        "store": "embeddings/reference_store_bumble_combined.npz",
        "face_weight": 0.50
    },
    "round2": {
        "name": "Round 2 Model (Combined)",
        "regressor": "models/rating_regressor_bumble_combined_round2.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_combined_round2.joblib",
        "store": "embeddings/reference_store_bumble_combined_round2.npz",
        "face_weight": 0.22
    },
    "round3": {
        "name": "Round 3 Model (Latest Combined)",
        "regressor": "models/rating_regressor_bumble_combined_round3.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_combined_round3.joblib",
        "store": "embeddings/reference_store_bumble_combined_round3.npz",
        "face_weight": 0.60
    },
    "bumble_only": {
        "name": "Bumble Only Model (Round 1)",
        "regressor": "models/rating_regressor_bumble_only.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_only.joblib",
        "store": "embeddings/reference_store_bumble_only.npz",
        "face_weight": 0.50
    },
    "bumble_only_round2": {
        "name": "Bumble Only Model (Round 2)",
        "regressor": "models/rating_regressor_bumble_only_round2.joblib",
        "multimodal": "models/rating_regressor_multimodal_bumble_only_round2.joblib",
        "store": "embeddings/reference_store_bumble_only_round2.npz",
        "face_weight": 0.22
    }
}

TOP_K = 20
SWIPE_THRESHOLD = 55.0
SERVER_PORT = int(os.environ.get("BUMBLECLAW_PORT", "7860"))
DASHBOARD_URL = os.environ.get("BUMBLECLAW_DASHBOARD_URL", "http://localhost:7863/sandbox")


@app.get("/")
def api_status():
    return {
        "service": "DatingClaw Image Analysis API",
        "status": "ok",
        "analyze_endpoint": "POST /analyze",
        "dashboard_url": DASHBOARD_URL,
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@lru_cache(maxsize=16)
def cached_store(path: str):
    if not Path(path).exists():
        return None
    return load_store(path)

@lru_cache(maxsize=32)
def cached_regressor(path: str):
    if not Path(path).exists():
        return None
    return load_regressor(path)

@lru_cache(maxsize=16)
def cached_preference_model(path: str):
    if not Path(path).exists():
        return None
    return load_preference_model(path)

def biased_multimodal_score(
    regressor_rating: float | None,
    multimodal_rating: float | None,
    *,
    face_weight: float,
) -> float | None:
    if regressor_rating is None or multimodal_rating is None:
        return None
    return face_weight * regressor_rating + (1 - face_weight) * multimodal_rating

def comparison_text(score: float | None, base_score: float | None, label: str = "Ridge") -> str:
    if score is None or base_score is None:
        return "no comparison"
    delta = score - base_score
    if abs(delta) < 0.05:
        return f"same as {label}"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{abs(delta):.1f} vs {label}"

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    model_version: str = Form("bumble_10k_pure") # Default is the Pure Bumble 10k model
):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / "upload_image"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Dynamically verify if environment provider resolves correctly
        from face_similarity.embedding import resolve_providers
        try:
            _, active_provider = resolve_providers("auto")
        except Exception:
            active_provider = "cpu"

        # 1. Determine configuration store & metadata settings
        mv_lower = model_version.lower()
        is_experimental = mv_lower in EXPERIMENTAL_SETUPS
        if is_experimental:
            setup = EXPERIMENTAL_SETUPS[mv_lower]
            store_path = setup.store
            k_val = setup.k
            model_display_name = setup.setup_name
        else:
            if mv_lower not in BASE_MODELS_CONFIG:
                mv_lower = "bumble_10k_pure"
            cfg = BASE_MODELS_CONFIG[mv_lower]
            store_path = cfg["store"]
            k_val = TOP_K
            model_display_name = cfg["name"]

        store = cached_store(store_path)
        if store is None:
            raise HTTPException(status_code=500, detail=f"Reference store not found: {store_path}")
        
        # 2. Extract Face Embedding
        try:
            embedding = get_face_embedding(
                str(temp_file_path),
                model_name=store.model_name,
                provider=active_provider,
                det_size=store.det_size,
                det_thresh=store.det_thresh,
                enforce_detection=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(exc)}")

        # 3. Score Nearest Neighbors (KNN)
        result = score_embedding(embedding, store.embeddings, store.ratings, k=k_val)
        
        # 4. Pre-extract CLIP Embedding (using round3 as baseline configuration model metadata)
        clip_embedding = None
        r3_multi = cached_regressor(BASE_MODELS_CONFIG["round3"]["multimodal"])
        if r3_multi is not None:
            try:
                clip_model = r3_multi.metadata.get("clip_model", "openai/clip-vit-base-patch32")
                clip_embedding = get_clip_embedding(str(temp_file_path), model_name=clip_model, provider=active_provider)
            except Exception as e:
                import traceback
                traceback.print_exc()

        # 5. Compute base scoring epochs for comparison loops
        base_scores = {}
        for key, cfg in BASE_MODELS_CONFIG.items():
            ridge_score = None
            multi_score = None
            blended_score = None

            reg = cached_regressor(cfg["regressor"])
            if reg is not None:
                ridge_score = float(predict_rating(reg, embedding))
            
            multi = cached_regressor(cfg["multimodal"])
            if multi is not None and clip_embedding is not None:
                try:
                    multi_score = float(predict_multimodal_rating(
                        multi,
                        face_embedding=embedding,
                        clip_embedding=clip_embedding,
                    ))
                    blended_score = float(biased_multimodal_score(ridge_score, multi_score, face_weight=cfg["face_weight"]))
                except Exception:
                    pass
            
            if blended_score is None:
                blended_score = ridge_score if ridge_score is not None else float(result.rating)

            base_scores[key] = {
                "ridge": ridge_score,
                "multimodal": multi_score,
                "blended": blended_score,
                "face_weight": cfg["face_weight"]
            }

        # 6. Evaluate selected model path details
        preference_probability_val = None
        preference_threshold_val = None
        is_preference_decision = False

        if is_experimental:
            setup = EXPERIMENTAL_SETUPS[model_version]
            
            # Predict ridge & multimodal on the setup's regressors
            setup_ridge = None
            setup_reg = cached_regressor(setup.regressor)
            if setup_reg is not None:
                setup_ridge = float(predict_rating(setup_reg, embedding))
                
            setup_multimodal = None
            setup_multi = cached_regressor(setup.multimodal_regressor)
            if setup_multi is not None and clip_embedding is not None:
                try:
                    setup_multimodal = float(predict_multimodal_rating(
                        setup_multi,
                        face_embedding=embedding,
                        clip_embedding=clip_embedding,
                    ))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            setup_blended = biased_multimodal_score(setup_ridge, setup_multimodal, face_weight=setup.face_weight)
            if setup_blended is None:
                setup_blended = setup_ridge if setup_ridge is not None else float(result.rating)
                
            # Build Prediction model
            from face_similarity.prediction import RatingPrediction
            prediction_obj = RatingPrediction(
                rating=setup_blended,
                method=setup.method,
                face_rating=setup_ridge,
                multimodal_rating=setup_multimodal,
                knn_rating=float(result.rating)
            )

            # Apply MultimodalX prediction adjustments if required
            from face_similarity.multimodalx import METHODS as MULTIMODALX_METHODS
            from face_similarity.multimodalx import prediction as multimodalx_prediction
            old_p_like = None
            if setup.method in MULTIMODALX_METHODS:
                pref_path = setup.blend_preference_model or setup.preference_model
                pref_model = cached_preference_model(pref_path)
                if pref_model is not None:
                    from face_similarity.multimodalx import PREFERENCE_FEATURE_THRESHOLD as MULTIMODALX_PREFERENCE_FEATURE_THRESHOLD
                    from face_similarity.multimodalx import old_p_like_face_weight as multimodalx_old_p_like_face_weight
                    features = features_from_prediction(
                        prediction_obj,
                        threshold=MULTIMODALX_PREFERENCE_FEATURE_THRESHOLD,
                        face_weight=multimodalx_old_p_like_face_weight(setup.method, setup.face_weight),
                        regressor_path=setup.regressor,
                        multimodal_regressor_path=setup.multimodal_regressor
                    )
                    old_p_like = preference_probability(pref_model, features)
                    prediction_obj = multimodalx_prediction(prediction_obj, old_p_like, method=setup.method)
                    setup_blended = prediction_obj.rating

            # Score Preference/Veto Layer
            pref_model = cached_preference_model(setup.preference_model)
            if pref_model is not None:
                features = features_from_prediction(
                    prediction_obj,
                    threshold=setup.threshold,
                    face_weight=setup.face_weight,
                    regressor_path=setup.regressor,
                    multimodal_regressor_path=setup.multimodal_regressor
                )
                if old_p_like is not None:
                    features["old_p_like"] = old_p_like
                preference_probability_val = preference_probability(pref_model, features)
                preference_threshold_val = setup.preference_threshold
            
            primary_ridge = setup_ridge
            primary_multimodal = setup_multimodal
            primary_face_weight = setup.face_weight

            if setup.decision_mode == "preference" and preference_probability_val is not None:
                is_preference_decision = True
                primary_score = preference_probability_val * 100.0  # Scale to 0-100%
                active_threshold = preference_threshold_val * 100.0
                decision = "RIGHT" if preference_probability_val >= preference_threshold_val else "LEFT"
            else:
                primary_score = setup_blended
                active_threshold = setup.threshold
                decision = "RIGHT" if setup_blended >= setup.threshold else "LEFT"
            base_score = setup_blended
        else:
            # Baseline Regressor Config selected
            primary_score = base_scores[model_version]["blended"]
            primary_ridge = base_scores[model_version]["ridge"]
            primary_multimodal = base_scores[model_version]["multimodal"]
            primary_face_weight = BASE_MODELS_CONFIG[model_version]["face_weight"]
            active_threshold = SWIPE_THRESHOLD
            decision = "RIGHT" if primary_score >= SWIPE_THRESHOLD else "LEFT"
            base_score = primary_score

        # Determine formula description
        if is_experimental:
            setup = EXPERIMENTAL_SETUPS[model_version]
            if setup.method == "multimodalx":
                formula_description = "73% Ridge + 20% Multimodal + 7% P(like)"
            elif setup.method == "multimodalx2":
                formula_description = "12% Ridge + 5% Multimodal + 53% P(like) + 30% KNN"
            elif setup.method == "multimodalx5":
                formula_description = "60% Ridge + 5% Multimodal + 15% P(like) + 20% KNN"
            elif setup.method == "multimodalx_original":
                formula_description = "48% Ridge + 48% Multimodal + 4% P(like)"
            else:
                formula_description = f"{int(setup.face_weight * 100)}% Ridge + {int((1 - setup.face_weight) * 100)}% Multimodal"
        else:
            face_weight = BASE_MODELS_CONFIG[model_version]["face_weight"]
            formula_description = f"{int(face_weight * 100)}% Ridge + {int((1 - face_weight) * 100)}% Multimodal"

        threshold_diff = primary_score - active_threshold

        # Comparison strings vs Primary Ridge baseline
        comparison_texts = {
            "face_biased": comparison_text(base_score, primary_ridge, "Ridge"),
            "multimodal": comparison_text(primary_multimodal, primary_ridge, "Ridge"),
            "knn": comparison_text(float(result.rating), primary_ridge, "Ridge")
        }
        for key, sc in base_scores.items():
            comparison_texts[f"{key}_face_biased"] = comparison_text(sc["blended"], primary_ridge, "Ridge")

        nearest_refs = []
        for index, similarity in zip(result.nearest_indices, result.nearest_similarities):
            nearest_refs.append({
                "name": Path(store.paths[index]).name,
                "rating": float(store.ratings[index]),
                "similarity": float(similarity)
            })

        return {
            "success": True,
            "decision": decision,
            "primary_score": primary_score,
            "threshold": active_threshold,
            "threshold_diff": threshold_diff,
            "model_version": mv_lower,
            "model_display_name": model_display_name,
            "formula_description": formula_description,
            "primary_face_weight": primary_face_weight,
            "primary_k": k_val,
            "is_preference_mode": is_preference_decision,
            "base_score": base_score,
            "scores": {
                "face_biased": base_score,
                "ridge": primary_ridge,
                "multimodal": primary_multimodal,
                "knn": float(result.rating),
                "preference_probability": preference_probability_val,
                "preference_threshold": preference_threshold_val,
                **{f"{key}_face_biased": sc["blended"] for key, sc in base_scores.items()}
            },
            "comparison_text": comparison_texts,
            "details": {
                "max_similarity": float(result.max_similarity),
                "mean_similarity": float(result.mean_similarity),
                "references_count": len(store.embeddings)
            },
            "nearest_references": nearest_refs
        }
    except HTTPException as e:
        raise e
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    # Check if running outside of local .venv virtualenv
    is_venv = os.environ.get("VIRTUAL_ENV") or hasattr(sys, "real_prefix") or (sys.base_prefix != sys.prefix)
    if not is_venv:
        print("\n" + "="*80)
        print("WARNING: You are NOT running in the project's local virtual environment (.venv)!")
        print("InsightFace and ONNX Runtime might fail to load DLLs due to dependency mismatches.")
        print("Please run the app using:")
        print("   .\\.venv\\Scripts\\python.exe app.py")
        print("="*80 + "\n")
        raise SystemExit(f"Current Python: {sys.executable}")
        
    print(f"Starting DatingClaw Image Analysis API on port {SERVER_PORT}...")
    uvicorn.run("app:app", host="0.0.0.0", port=SERVER_PORT, reload=False)
