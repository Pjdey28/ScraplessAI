import pandas as pd

def sample_order() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"part_id": "PANEL_A", "length": 220, "width": 500, "quantity": 6, "priority": "High", "material": "Aluminium"},
            {"part_id": "PANEL_B", "length": 220, "width": 160, "quantity": 8, "priority": "Medium", "material": "Aluminium"},
            {"part_id": "PLATE_C", "length": 500, "width": 200, "quantity": 13, "priority": "Medium", "material": "Aluminium"},
            {"part_id": "FRAME_D", "length": 600, "width": 360, "quantity": 11, "priority": "High", "material": "Aluminium"},
            {"part_id": "BRACKET_E", "length": 260, "width": 280, "quantity": 14, "priority": "Low", "material": "Aluminium"},
            {"part_id": "LONG_RAIL_F", "length": 700, "width": 420, "quantity": 7, "priority": "High", "material": "Aluminium"},
            {"part_id": "SMALL_TAB_G", "length": 220, "width": 120, "quantity": 6, "priority": "Low", "material": "Aluminium"},
        ]
    )
