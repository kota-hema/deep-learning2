from __future__ import annotations

import streamlit as st

from sales_report import DATA_DIR, daily_metrics, daily_sales_series, generate_daily_report, load_sales_data

st.set_page_config(page_title="Daily Sales Report", page_icon="📈", layout="wide")
st.title("Daily Sales Report")

try:
    sales_data, source_files = load_sales_data()
except (ValueError, OSError, ImportError) as error:
    st.error(str(error))
    st.stop()

st.caption(f"Loaded {len(source_files)} file(s) from `{DATA_DIR.relative_to(DATA_DIR.parent)}/`: {', '.join(source_files)}")
available_dates = sorted(sales_data["_report_date"].dropna().dt.date.unique())
if not available_dates:
    st.error("No valid sales dates were found in the data files.")
    st.stop()

with st.sidebar:
    st.header("Report settings")
    report_date = st.selectbox(
        "Sales date",
        options=available_dates,
        index=len(available_dates) - 1,
        format_func=lambda value: value.strftime("%B %d, %Y"),
    )
    provider = st.radio("Language model", ["Ollama", "OpenRouter"])
    st.caption("Ollama must be running locally. OpenRouter uses the key in `.env`.")

metrics = daily_metrics(sales_data, report_date)
first, second, third, fourth = st.columns(4)
first.metric("Sales", f"${metrics['sales_total']:,.2f}")
second.metric("Orders", f"{metrics['orders']:,}")
third.metric("Average order value", f"${metrics['average_order_value']:,.2f}")
fourth.metric(
    "Vs. previous day",
    f"{metrics['change_percent']:+.1f}%" if metrics["change_percent"] is not None else "N/A",
)

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Sales over time")
    st.line_chart(daily_sales_series(sales_data), y_label="Sales ($)")
with right:
    st.subheader("Daily breakdown")
    for label, items in metrics["breakdowns"].items():
        with st.expander(f"Top {label}", expanded=(label == "category")):
            if items:
                st.dataframe(items, hide_index=True, width="stretch")
            else:
                st.caption("No values available.")

st.subheader("Narrative report")
if st.button("Generate report", type="primary"):
    with st.spinner(f"Generating report with {provider}…"):
        try:
            st.session_state["daily_report"] = generate_daily_report(metrics, provider)
            st.session_state["report_error"] = None
        except Exception as error:
            st.session_state["report_error"] = str(error)

if st.session_state.get("report_error"):
    st.error(st.session_state["report_error"])
if st.session_state.get("daily_report"):
    st.markdown(st.session_state["daily_report"])
    st.download_button(
        "Download report as Markdown",
        data=st.session_state["daily_report"],
        file_name=f"sales-report-{report_date.isoformat()}.md",
        mime="text/markdown",
    )