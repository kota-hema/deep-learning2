import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sales_report import (
    _invoke_report_chain,
    daily_metrics,
    generate_metrics_report,
    load_sales_data,
    load_uploaded_sales_data,
)


class UploadedBytes(io.BytesIO):
    def __init__(self, name, contents):
        super().__init__(contents)
        self.name = name


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

    def test_loads_uploaded_csv_and_excel_files_without_saving(self):
        csv_file = UploadedBytes(
            "external.csv",
            b"Date,Revenue\n2025-01-01,12.50\n",
        )
        excel_contents = io.BytesIO()
        pd.DataFrame({"Order Date": ["2025-01-01"], "Sales": [7.5]}).to_excel(
            excel_contents, index=False
        )
        excel_file = UploadedBytes("external.xlsx", excel_contents.getvalue())

        data, files = load_uploaded_sales_data([csv_file, excel_file])
        metrics = daily_metrics(data, "2025-01-01")

        self.assertEqual(set(files), {"external.csv", "external.xlsx"})
        self.assertEqual(metrics["sales_total"], 20.0)
        self.assertEqual(metrics["orders"], 2)

    def test_ollama_connection_refused_has_setup_guidance(self):
        class FailingChain:
            def invoke(self, _input):
                raise ConnectionRefusedError(111, "Connection refused")

        with self.assertRaisesRegex(RuntimeError, "OLLAMA_BASE_URL"):
            _invoke_report_chain(FailingChain(), {}, "Ollama")

    def test_ollama_connection_error_is_retried_once(self):
        class FailsOnceChain:
            calls = 0

            def invoke(self, _input):
                self.calls += 1
                if self.calls == 1:
                    raise OSError(99, "Cannot assign requested address")
                return "report response"

        chain = FailsOnceChain()

        response = _invoke_report_chain(chain, {}, "Ollama")

        self.assertEqual(response, "report response")
        self.assertEqual(chain.calls, 2)

    def test_metrics_report_is_available_without_an_llm(self):
        report = generate_metrics_report({
            "date": "2025-01-01",
            "sales_total": 20.0,
            "orders": 2,
            "average_order_value": 10.0,
            "change_percent": 25.0,
            "breakdowns": {"category": [{"name": "Furniture", "sales": 20.0}]},
        })

        self.assertIn("Daily Sales Report: 2025-01-01", report)
        self.assertIn("Sales increased 25.0%", report)
        self.assertIn("Top category by sales: Furniture", report)
        self.assertIn("AI narrative unavailable", report)


if __name__ == "__main__":
    unittest.main()