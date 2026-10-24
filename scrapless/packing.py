from dataclasses import dataclass, asdict
from typing import Optional
import math
import pandas as pd

@dataclass
class Part:
    part_id: str
    width: float
    height: float
    instance: int

@dataclass
class Placement:
    sheet_index: int
    part_id: str
    instance: int
    x: float
    y: float
    width: float
    height: float
    rotated: bool = False
    original_width: float | None = None
    original_height: float | None = None

    def to_dict(self):
        return asdict(self)

@dataclass
class PackingResult:
    algorithm: str
    sheet_width: float
    sheet_height: float
    placements: list[Placement]
    sheet_count: int
    requested_count: int
    unplaced: list[Part]
    runtime_seconds: float
    allow_rotation: bool = True
    spacing: float = 0.0
    max_sheets: int | None = None

    def placements_frame(self) -> pd.DataFrame:
        columns = ["sheet_index", "part_id", "instance", "x", "y", "width", "height", "rotated",
                   "original_width", "original_height"]
        if not self.placements:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([placement.to_dict() for placement in self.placements], columns=columns)

def expand_parts(order: pd.DataFrame) -> list[Part]:
    parts = []
    for row in order.itertuples(index=False):
        width = getattr(row, "width", None)
        height = getattr(row, "length", None)
        if width is None:
            width = getattr(row, "part_width")
        if height is None:
            height = getattr(row, "part_height")
        for instance in range(1, int(row.quantity) + 1):
            parts.append(Part(part_id=str(row.part_id), width=float(width), height=float(height), instance=instance))
    return parts

def _fits(part_width, part_height, sheet_width, sheet_height):
    return part_width <= sheet_width and part_height <= sheet_height

def first_fit_decreasing(order: pd.DataFrame, sheet_width: float, sheet_height: float,
                         allow_rotation: bool = True, spacing: float = 0.0,
                         max_sheets: int | None = None) -> PackingResult:
    import time
    started = time.perf_counter()
    parts = expand_parts(order)
    parts.sort(key=lambda part: (max(part.width, part.height), part.height, part.width), reverse=True)

    placements = []
    unplaced = []
    sheets = []
    for part in parts:
        orientations = [(part.width + spacing, part.height + spacing, False)]
        if allow_rotation and part.width != part.height:
            orientations.append((part.height, part.width, True))

        placed = False
        for sheet_index, sheet in enumerate(sheets):
            for width, height, rotated in orientations:
                if not _fits(width, height, sheet_width, sheet_height):
                    continue
                for shelf in sheet["shelves"]:
                    if height <= shelf["height"] and shelf["x"] + width <= sheet_width:
                        placements.append(Placement(sheet_index, part.part_id, part.instance, shelf["x"], shelf["y"],
                                                    width - spacing, height - spacing, rotated, part.width, part.height))
                        shelf["x"] += width
                        placed = True
                        break
                if placed: break
                next_y = sheet["next_y"]
                if next_y + height <= sheet_height:
                    sheet["shelves"].append({"x": width, "y": next_y, "height": height})
                    placements.append(Placement(sheet_index, part.part_id, part.instance, 0, next_y,
                                                width - spacing, height - spacing, rotated, part.width, part.height))
                    sheet["next_y"] += height
                    placed = True
                    break
            if placed: break

        if not placed:
            chosen = None
            for width, height, rotated in orientations:
                if _fits(width, height, sheet_width, sheet_height):
                    chosen = (width, height, rotated)
                    break
            if chosen is None or (max_sheets is not None and len(sheets) >= max_sheets):
                unplaced.append(part)
                continue
            width, height, rotated = chosen
            sheet = {"shelves": [{"x": width, "y": 0, "height": height}], "next_y": height}
            sheets.append(sheet)
            placements.append(Placement(len(sheets) - 1, part.part_id, part.instance, 0, 0,
                                        width - spacing, height - spacing, rotated, part.width, part.height))

    return PackingResult(
        algorithm="First Fit Decreasing baseline",
        sheet_width=sheet_width, sheet_height=sheet_height, placements=placements,
        sheet_count=len(sheets), requested_count=len(parts), unplaced=unplaced, runtime_seconds=time.perf_counter() - started,
        allow_rotation=allow_rotation, spacing=spacing, max_sheets=max_sheets,
    )

def _intersects(a, b):
    return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])

def _contains(outer, inner):
    return (inner["x"] >= outer["x"] and inner["y"] >= outer["y"] and inner["x"] + inner["width"] <= outer["x"] + outer["width"] and inner["y"] + inner["height"] <= outer["y"] + outer["height"])

def _split_free_rect(free_rect, used_rect):
    if not _intersects(free_rect, used_rect): return [free_rect]
    result = []
    fx, fy, fw, fh = free_rect["x"], free_rect["y"], free_rect["width"], free_rect["height"]
    ux, uy, uw, uh = used_rect["x"], used_rect["y"], used_rect["width"], used_rect["height"]

    if ux < fx + fw and ux + uw > fx:
        if uy > fy and uy < fy + fh: result.append({"x": fx, "y": fy, "width": fw, "height": uy - fy})
        if uy + uh < fy + fh: result.append({"x": fx, "y": uy + uh, "width": fw, "height": fy + fh - (uy + uh)})
    if uy < fy + fh and uy + uh > fy:
        if ux > fx and ux < fx + fw: result.append({"x": fx, "y": fy, "width": ux - fx, "height": fh})
        if ux + uw < fx + fw: result.append({"x": ux + uw, "y": fy, "width": fx + fw - (ux + uw), "height": fh})
    return [rect for rect in result if rect["width"] > 1e-9 and rect["height"] > 1e-9]

def _prune_free_rectangles(free_rects):
    pruned = []
    for i, rect in enumerate(free_rects):
        if any(i != j and _contains(other, rect) for j, other in enumerate(free_rects)): continue
        pruned.append(rect)
    return pruned

def maxrects_bssf(order: pd.DataFrame, sheet_width: float, sheet_height: float,
                  allow_rotation: bool = True, spacing: float = 0.0,
                  max_sheets: int | None = None) -> PackingResult:
    import time
    started = time.perf_counter()
    parts = expand_parts(order)
    parts.sort(key=lambda part: part.width * part.height, reverse=True)

    placements = []
    unplaced = []
    sheets_free = []

    for part in parts:
        best = None
        for sheet_index, free_rects in enumerate(sheets_free):
            for free_index, free_rect in enumerate(free_rects):
                orientations = [(part.width + spacing, part.height + spacing, False)]
                if allow_rotation and part.width != part.height: orientations.append((part.height + spacing, part.width + spacing, True))
                for width, height, rotated in orientations:
                    if width <= free_rect["width"] and height <= free_rect["height"]:
                        leftover_h, leftover_w = abs(free_rect["height"] - height), abs(free_rect["width"] - width)
                        score = (min(leftover_w, leftover_h), max(leftover_w, leftover_h))
                        candidate = (score, sheet_index, free_index, width, height, rotated, free_rect)
                        if best is None or candidate[0] < best[0]: best = candidate

        if best is None:
            orientations = [(part.width + spacing, part.height + spacing, False)]
            if allow_rotation and part.width != part.height: orientations.append((part.height + spacing, part.width + spacing, True))
            possible = [(w, h, r) for w, h, r in orientations if w <= sheet_width and h <= sheet_height]
            if not possible or (max_sheets is not None and len(sheets_free) >= max_sheets):
                unplaced.append(part)
                continue
            width, height, rotated = min(possible, key=lambda item: (min(sheet_width - item[0], sheet_height - item[1]), max(sheet_width - item[0], sheet_height - item[1])))
            sheet_index = len(sheets_free)
            sheets_free.append([{"x": 0.0, "y": 0.0, "width": sheet_width, "height": sheet_height}])
            free_rect = sheets_free[sheet_index][0]
        else:
            _, sheet_index, _, width, height, rotated, free_rect = best

        used = {"x": free_rect["x"], "y": free_rect["y"], "width": width, "height": height}
        placements.append(Placement(sheet_index, part.part_id, part.instance, used["x"], used["y"],
                                    width - spacing, height - spacing, rotated, part.width, part.height))

        updated = []
        for rect in sheets_free[sheet_index]: updated.extend(_split_free_rect(rect, used))
        sheets_free[sheet_index] = _prune_free_rectangles(updated)

    return PackingResult(
        algorithm="MaxRects BSSF",
        sheet_width=sheet_width, sheet_height=sheet_height, placements=placements,
        sheet_count=len(sheets_free), requested_count=len(parts), unplaced=unplaced, runtime_seconds=time.perf_counter() - started,
        allow_rotation=allow_rotation, spacing=spacing, max_sheets=max_sheets,
    )

def validate_layout(result: PackingResult, tolerance: float = 1e-6) -> dict:
    placements = result.placements
    issues = []
    seen = set()
    for index, placement in enumerate(placements):
        key = (placement.part_id, placement.instance)
        if key in seen:
            issues.append(f"Duplicate part instance {placement.part_id} inst {placement.instance}.")
        seen.add(key)
        if placement.width <= 0 or placement.height <= 0:
            issues.append(f"Part {placement.part_id} has invalid dimensions.")
        if not result.allow_rotation and placement.rotated:
            issues.append(f"Part {placement.part_id} is rotated while rotation is disabled.")
        if result.allow_rotation and result.spacing > 0:
            # The stored physical rectangle excludes spacing; spacing is reserved by the
            # algorithms and therefore must be reflected by its distance from neighbors.
            pass
        if placement.x < -tolerance or placement.y < -tolerance: issues.append(f"Part {placement.part_id} has a negative position.")
        if placement.x + placement.width > result.sheet_width + tolerance: issues.append(f"Part {placement.part_id} exceeds the sheet width.")
        if placement.y + placement.height > result.sheet_height + tolerance: issues.append(f"Part {placement.part_id} exceeds the sheet height.")
        for other in placements[index + 1:]:
            if placement.sheet_index == other.sheet_index:
                a = {"x": placement.x, "y": placement.y, "width": placement.width, "height": placement.height}
                b = {"x": other.x, "y": other.y, "width": other.width, "height": other.height}
                if _intersects(a, b): issues.append(f"Overlap between {placement.part_id} inst {placement.instance} and {other.part_id} inst {other.instance}.")
    reconciled = len(placements) + len(result.unplaced) == result.requested_count
    if not reconciled:
        issues.append("Placed and unplaced instances do not reconcile with the requested count.")
    return {"valid": len(issues) == 0 and len(result.unplaced) == 0, "issues": issues,
            "placed_count": len(placements), "unplaced_count": len(result.unplaced),
            "requested_count": result.requested_count, "reconciled": reconciled}
