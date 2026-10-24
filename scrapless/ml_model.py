from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["total_part_area_ratio", "part_count", "mean_aspect_ratio", "aspect_ratio_std", "small_part_fraction", "dimension_variability"]

def _synthetic_features(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    return pd.DataFrame({
        "total_part_area_ratio": rng.uniform(0.15, 1.25, n_samples),
        "part_count": rng.integers(4, 180, n_samples),
        "mean_aspect_ratio": rng.uniform(1.0, 4.5, n_samples),
        "aspect_ratio_std": rng.uniform(0.0, 2.0, n_samples),
        "small_part_fraction": rng.uniform(0.0, 0.8, n_samples),
        "dimension_variability": rng.uniform(0.0, 1.0, n_samples),
    })

def _proxy_label(features: pd.DataFrame, rng=None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng(123)
    f = features
    score = (1.5 * (f["total_part_area_ratio"] > 0.78).astype(float) + 0.9 * (f["aspect_ratio_std"] > 1.0).astype(float) +
             0.7 * (f["dimension_variability"] > 0.65).astype(float) + 0.5 * (f["part_count"] > 90).astype(float) +
             0.4 * (f["small_part_fraction"] > 0.55).astype(float) + 0.35 * (f["mean_aspect_ratio"] > 3.0).astype(float) +
             rng.normal(0, 0.55, len(f)))
    return (score >= 1.8).astype(int)

def generate_synthetic_training_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    features = _synthetic_features(n_samples, random_state)
    features["scrap_risk_label"] = _proxy_label(features, np.random.default_rng(random_state + 17))
    return features

@lru_cache(maxsize=1)
def train_scrap_risk_model():
    dataset = generate_synthetic_training_data()
    X, y = dataset[FEATURES], dataset["scrap_risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=140, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluation = {"accuracy": float(accuracy_score(y_test, preds)), "f1": float(f1_score(y_test, preds, zero_division=0)), "test_rows": int(len(y_test)), "training_rows": int(len(y_train)), "data_note": "Synthetic proxy data."}
    return model, evaluation

def build_order_features(order: pd.DataFrame, sheet_width: float, sheet_height: float) -> pd.DataFrame:
    ws, hs = [], []
    for row in order.itertuples(index=False):
        c = int(row.quantity)
        width = getattr(row, "width", None)
        height = getattr(row, "length", None)
        if width is None:
            width = getattr(row, "part_width")
        if height is None:
            height = getattr(row, "part_height")
        ws.extend([float(width)] * c)
        hs.extend([float(height)] * c)
    w_arr, h_arr = np.asarray(ws), np.asarray(hs)
    ratios = np.maximum(w_arr / h_arr, h_arr / w_arr)
    part_area = w_arr * h_arr
    sheet_area = float(sheet_width) * float(sheet_height)
    return pd.DataFrame([{
        "total_part_area_ratio": float(part_area.sum() / sheet_area) if sheet_area else 0.0,
        "part_count": len(ws),
        "mean_aspect_ratio": float(np.mean(ratios)),
        "aspect_ratio_std": float(np.std(ratios)),
        "small_part_fraction": float(np.mean(part_area < sheet_area * 0.04)),
        "dimension_variability": float((np.std(w_arr) / max(np.mean(w_arr), 1e-9) + np.std(h_arr) / max(np.mean(h_arr), 1e-9)) / 2.0)
    }], columns=FEATURES)

def predict_scrap_risk(order: pd.DataFrame, sheet_width: float, sheet_height: float) -> dict:
    model, eval_data = train_scrap_risk_model()
    feats = build_order_features(order, sheet_width, sheet_height)
    prob = float(model.predict_proba(feats)[0, 1])
    imps = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
    return {"risk_label": "High" if prob >= 0.5 else "Lower", "probability": prob, "feature_values": feats.iloc[0].to_dict(), "feature_importances": [{"feature": k, "importance": float(v)} for k, v in imps], "evaluation": eval_data}
