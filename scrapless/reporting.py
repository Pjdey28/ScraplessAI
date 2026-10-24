import pandas as pd

def to_csv_bytes(frame: pd.DataFrame) -> bytes: return frame.to_csv(index=False).encode("utf-8")
