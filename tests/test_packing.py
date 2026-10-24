import pandas as pd
from scrapless.packing import first_fit_decreasing, maxrects_bssf, validate_layout, PackingResult, Placement, Part

def order(rows):
    return pd.DataFrame(rows)

def test_both_algorithms_place_and_are_deterministic():
    o = order([{"part_id": "A", "length": 40, "width": 30, "quantity": 2},
               {"part_id": "B", "length": 20, "width": 20, "quantity": 3}])
    for algorithm in (first_fit_decreasing, maxrects_bssf):
        a = algorithm(o, 100, 100)
        b = algorithm(o, 100, 100)
        assert validate_layout(a)["valid"]
        assert [p.to_dict() for p in a.placements] == [p.to_dict() for p in b.placements]
        assert len(a.placements) == 5

def test_rotation_and_oversized_parts():
    o = order([{"part_id": "ROT", "length": 80, "width": 120, "quantity": 1}])
    assert len(first_fit_decreasing(o, 100, 130, allow_rotation=False).unplaced) == 1
    rotated = first_fit_decreasing(o, 100, 130, allow_rotation=True)
    assert not rotated.unplaced and rotated.placements[0].rotated
    too_large = maxrects_bssf(order([{"part_id": "X", "length": 200, "width": 200, "quantity": 1}]), 100, 100)
    assert len(too_large.unplaced) == 1

def test_spacing_and_max_sheets():
    o = order([{"part_id": "A", "length": 60, "width": 60, "quantity": 2}])
    result = first_fit_decreasing(o, 100, 100, spacing=5, max_sheets=1)
    assert len(result.unplaced) == 1

def test_validator_catches_overlap_and_bounds():
    result = PackingResult("test", 100, 100, [
        Placement(0, "A", 1, 0, 0, 60, 60),
        Placement(0, "B", 1, 50, 50, 60, 60),
    ], 1, 2, [], 0)
    checked = validate_layout(result)
    assert not checked["valid"]
    assert any("Overlap" in issue for issue in checked["issues"])
    assert any("exceeds" in issue for issue in checked["issues"])
