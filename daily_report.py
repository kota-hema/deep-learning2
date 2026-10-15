from __future__ import annotations

import argparse
from pathlib import Path

from sales_report import DATA_DIR, daily_metrics, generate_daily_report, load_sales_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a daily sales report from files in data/.")
    parser.add_argument("--date", help="Sales date in YYYY-MM-DD format (default: latest available).")
    parser.add_argument("--provider", choices=("Ollama", "OpenRouter"), default="Ollama")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reports")
    args = parser.parse_args()

    data, files = load_sales_data(DATA_DIR)
    report_date = args.date or data["_report_date"].max().date().isoformat()
    metrics = daily_metrics(data, report_date)
    report = generate_daily_report(metrics, args.provider)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"sales-report-{metrics['date']}.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Generated {output_path} from {len(files)} file(s).")


if __name__ == "__main__":
    main()