from dataclasses import dataclass
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {"part_id", "length", "width", "quantity"}

@dataclass
class OrderValidation:
    valid: bool
    message: str
    order: pd.DataFrame | None = None

def validate_order_frame(frame: pd.DataFrame) -> OrderValidation:
    if frame is None or frame.empty:
        return OrderValidation(False, "The order file is empty.")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "length" not in normalized and "part_height" in normalized:
        normalized["length"] = normalized["part_height"]
    if "width" not in normalized and "part_width" in normalized:
        normalized["width"] = normalized["part_width"]
    if "part_id" not in normalized.columns:
        normalized["part_id"] = [f"PART_{index + 1}" for index in range(len(normalized))]
    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        return OrderValidation(False, "Missing required columns: " + ", ".join(sorted(missing)))

    for column in ("length", "width", "quantity"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized[["length", "width", "quantity"]].isna().any().any():
        return OrderValidation(False, "Dimensions and quantities must contain numeric values.")
    if not np.isfinite(normalized[["length", "width", "quantity"]].to_numpy()).all():
        return OrderValidation(False, "Dimensions and quantities must be finite.")

    if (normalized[["length", "width"]] <= 0).any().any():
        return OrderValidation(False, "Part dimensions must be greater than zero.")

    if (normalized["quantity"] <= 0).any():
        return OrderValidation(False, "Every quantity must be greater than zero.")

    if not (normalized["quantity"] % 1 == 0).all():
        return OrderValidation(False, "Quantities must be whole numbers.")
    if normalized["part_id"].isna().any() or normalized["part_id"].astype(str).str.strip().eq("").any():
        return OrderValidation(False, "Every part identifier must be present.")
    if normalized["part_id"].astype(str).duplicated().any():
        return OrderValidation(False, "Part identifiers must be unique.")

    normalized["quantity"] = normalized["quantity"].astype(int)
    normalized["part_id"] = normalized["part_id"].astype(str)
    normalized = normalized[["part_id", "length", "width", "quantity"] +
                            [c for c in ("priority", "material") if c in normalized.columns]]

    if len(normalized) > 500:
        return OrderValidation(False, "Please use 500 or fewer part types for this prototype.")

    if normalized["quantity"].sum() > 5000:
        return OrderValidation(False, "Please use 5,000 or fewer total parts for this prototype.")

    return OrderValidation(True, "Order validated successfully.", normalized)
