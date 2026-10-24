from dataclasses import dataclass, asdict
import pandas as pd
from .packing import PackingResult

@dataclass
class LayoutMetrics:
    algorithm: str
    sheets_used: int
    parts_requested: int
    parts_placed: int
    parts_unplaced: int
    part_area: float
    stock_area: float
    unused_area: float
    utilization_pct: float
    unused_area_pct: float
    estimated_material_cost: float
    runtime_seconds: float
    cost_per_placed_part: float | None
    validation_valid: bool = True

    def to_dict(self): return asdict(self)

def calculate_metrics(result: PackingResult, unit_cost_per_sheet: float = 0.0,
                      validation: dict | None = None) -> LayoutMetrics:
    part_area = sum(p.width * p.height for p in result.placements)
    stock_area = result.sheet_count * result.sheet_width * result.sheet_height
    unused_area = max(stock_area - part_area, 0.0)
    utilization = (part_area / stock_area * 100.0) if stock_area else 0.0
    unused_pct = (unused_area / stock_area * 100.0) if stock_area else 0.0

    return LayoutMetrics(
        algorithm=result.algorithm, sheets_used=result.sheet_count,
        parts_requested=result.requested_count, parts_placed=len(result.placements),
        parts_unplaced=len(result.unplaced), part_area=part_area, stock_area=stock_area,
        unused_area=unused_area, utilization_pct=utilization, unused_area_pct=unused_pct,
        estimated_material_cost=result.sheet_count * float(unit_cost_per_sheet),
        runtime_seconds=result.runtime_seconds,
        cost_per_placed_part=(result.sheet_count * float(unit_cost_per_sheet) / len(result.placements)
                              if result.placements else None),
        validation_valid=True if validation is None else bool(validation.get("valid", False)),
    )

def compare_metrics(baseline: LayoutMetrics, optimized: LayoutMetrics) -> dict:
    return {
        "sheets_saved": baseline.sheets_used - optimized.sheets_used,
        "estimated_cost_saved": baseline.estimated_material_cost - optimized.estimated_material_cost,
        "utilization_gain_percentage_points": optimized.utilization_pct - baseline.utilization_pct,
        "baseline_sheets": baseline.sheets_used,
        "optimized_sheets": optimized.sheets_used,
        "unused_area_reduction": baseline.unused_area - optimized.unused_area,
        "unplaced_change": optimized.parts_unplaced - baseline.parts_unplaced,
        "runtime_change_seconds": optimized.runtime_seconds - baseline.runtime_seconds,
        "baseline_validation": baseline.validation_valid,
        "optimized_validation": optimized.validation_valid,
    }

def metrics_frame(metrics: list[LayoutMetrics]) -> pd.DataFrame:
    return pd.DataFrame([metric.to_dict() for metric in metrics])
