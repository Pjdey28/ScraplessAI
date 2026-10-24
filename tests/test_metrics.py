import pandas as pd
from scrapless.packing import maxrects_bssf
from scrapless.metrics import calculate_metrics, compare_metrics

def test_metrics_area_cost_and_unplaced():
    order = pd.DataFrame([{"part_id": "A", "length": 50, "width": 50, "quantity": 2}])
    result = maxrects_bssf(order, 100, 100)
    metrics = calculate_metrics(result, 25)
    assert metrics.sheets_used == 1
    assert metrics.part_area == 5000
    assert metrics.stock_area == 10000
    assert metrics.unused_area == 5000
    assert metrics.utilization_pct == 50
    assert metrics.estimated_material_cost == 25
    assert metrics.cost_per_placed_part == 12.5

def test_comparison_includes_changes():
    order = pd.DataFrame([{"part_id": "A", "length": 50, "width": 50, "quantity": 2}])
    result = maxrects_bssf(order, 100, 100)
    result2 = maxrects_bssf(order, 60, 60)
    comparison = compare_metrics(calculate_metrics(result, 25), calculate_metrics(result2, 25))
    assert "unused_area_reduction" in comparison
    assert comparison["optimized_sheets"] >= comparison["baseline_sheets"]
