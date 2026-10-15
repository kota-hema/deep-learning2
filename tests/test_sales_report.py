import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sales_report import daily_metrics, load_sales_data


class SalesReportTests(unittest.TestCase):
    def test_loads_kaggle_sales_and_calculates_daily_metrics(self):
        data_dir = Path(__file__).parent.parent / "data"
        data, files = load_sales_data(data_dir)

        self.assertIn("train.csv", files)
        self.assertEqual(len(data), 9800)
        self.assertEqual(
            data.loc[data["Order Date"] == "08/11/2017", "_report_date"].iloc[0],
            pd.Timestamp("2017-08-11"),
        )
        report_date = data["_report_date"].max()
        metrics = daily_metrics(data, report_date)

        self.assertGreater(metrics["sales_total"], 0)
        self.assertGreater(metrics["orders"], 0)
        self.assertIn("category", metrics["breakdowns"])

    def test_combines_csv_and_excel_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            pd.DataFrame({"Date": ["2025-01-01"], "Revenue": ["$12.50"]}).to_csv(
                data_dir / "one.csv", index=False
            )
            pd.DataFrame({"Order Date": ["2025-01-01"], "Sales": [7.5]}).to_excel(
                data_dir / "two.xlsx", index=False
            )

            data, files = load_sales_data(data_dir)
            metrics = daily_metrics(data, "2025-01-01")

            self.assertEqual(set(files), {"one.csv", "two.xlsx"})
            self.assertEqual(metrics["sales_total"], 20.0)
            self.assertEqual(metrics["orders"], 2)


if __name__ == "__main__":
    unittest.main()