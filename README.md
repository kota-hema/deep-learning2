# Daily Sales Report

A Streamlit dashboard that combines sales CSV/Excel files from `data/`, calculates daily KPIs, and uses LangChain with either a local Ollama model or OpenRouter to write a concise sales narrative. A CLI report generator can be scheduled to run once per day.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set provider values in the root `.env` file. The project includes a blank, git-ignored `.env` and a reference in `.env.example`; add your OpenRouter key to `OPENROUTER_API_KEY` there. Ollama users should install Ollama, pull a model such as `llama3.2` (`ollama pull llama3.2`), and keep the Ollama service running. No API key is needed for Ollama.

## Run the dashboard

```bash
streamlit run app.py
```

Put additional `.csv`, `.xlsx`, or `.xlsm` sales files in `data/`. The loader combines files that contain a recognizable date column (for example `Order Date`) and sales amount column (for example `Sales` or `Revenue`). The dashboard shows the latest available date by default; choose another date in the sidebar to review it.

## Daily automation

Generate a report for the latest available sales date:

```bash
python daily_report.py --provider Ollama
```

If the selected LLM is unavailable or its credentials are missing, the dashboard and CLI still produce a metrics-only report and indicate that the AI narrative could not be generated.

For OpenRouter, use `--provider OpenRouter` after setting `OPENROUTER_API_KEY` in `.env`. Reports are written to `reports/`. To run automatically, add a cron entry using absolute paths for this workspace and its virtual environment, for example:

```cron
0 7 * * * cd /path/to/deep-learning2 && /path/to/deep-learning2/.venv/bin/python daily_report.py --provider Ollama >> /path/to/deep-learning2/reports/cron.log 2>&1
```

The schedule runs at 07:00 in the machine's local timezone and reports the latest date found in the input data. It does not fetch new sales from an external business system; add that day's export to `data/` before the scheduled run.

## Kaggle starter data

`data/train.csv` is the public [Superstore Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) by Rohit Sahoo, downloaded from Kaggle. Kaggle lists this dataset under GPL-2. Replace or supplement it with your own exports in `data/` as needed.

## Tests

```bash
python -m unittest discover -s tests
```