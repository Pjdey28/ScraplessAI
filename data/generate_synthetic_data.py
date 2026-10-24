from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrapless.ml_model import generate_synthetic_training_data

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic ScrapLess risk data.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of generated rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("synthetic_scrap_risk_dataset.csv"))
    args = parser.parse_args(argv)
    if args.rows < 10:
        parser.error("--rows must be at least 10")
    dataset = generate_synthetic_training_data(n_samples=args.rows, random_state=args.seed)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    print(f"Wrote {len(dataset):,} synthetic rows to {output}")

if __name__ == "__main__":
    main()
