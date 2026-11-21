"""
Streamlit dashboard for the NYC street-flooding digital twin prototype.
Run from the project root:
    python -m pip install streamlit pandas
    streamlit run dashboard/app.py

The dashboard only reads result files; it does not train models.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="NYC Flood Risk Digital Twin", layout="wide")
st.title("NYC street-flooding risk: digital twin prototype")
st.warning(
    "Replay of historical rainfall, no live sensors. "
    "Flood labels are 311 complaints, a proxy for flooding, not measured flooding. "
    "One rainfall point (Central Park) is used for all boroughs."
)


@st.cache_data
def get(name):
    return dd.load(name)


def not_ready(what):
    st.info(f"{what} is not available yet. It appears once the model scripts have been run "
            "and their result files are saved in the results/ folder.")


try:
    base, metrics, risk, whatif = get("borough_day"), get("metrics"), get("risk"), get("whatif")
except ValueError as err:
    st.error(str(err))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Data overview", "Model results", "Twin: risk by borough", "What-if rainfall"])

with tab1:
    if base is None:
        not_ready("data/processed/borough_day.csv")
    else:
        s = dd.split_summary(base)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Borough-day rows", f"{s['rows']:,}")
        c2.metric("Days", f"{s['days']:,}")
        c3.metric("Flood days, training (2024-2025)", f"{s['train_flood_share']:.1%}")
        c4.metric("Flood days, test (2026)", f"{s['test_flood_share']:.1%}")
        st.caption(f"{s['date_min']} to {s['date_max']}. A flood day is a borough-day with at least "
                   f"{dd.THRESHOLD} street or highway flooding complaints.")
        st.subheader("Share of days that are flood days, by borough")
        st.bar_chart(dd.flood_share_by_borough(base))
        st.subheader("Daily rainfall (mm)")
        rain = base.drop_duplicates("date").set_index("date")["precip_mm"]
        st.line_chart(rain)

with tab2:
    if metrics is None:
        not_ready("Model results")
    else:
        st.write("Evaluated on the 2026 test period. Accuracy is not shown because only about 10% "
                 "of borough-days are flood days.")
        st.dataframe(metrics.round(3), hide_index=True)
        st.bar_chart(metrics.set_index("model")[["precision", "recall", "f1"]])

with tab3:
    if risk is None:
        not_ready("The risk table")
    else:
        dates = sorted(risk["date"].dt.date.unique())
        chosen = st.select_slider("Date", options=dates, value=dates[-1])
        st.subheader(f"Predicted flood risk on {chosen}")
        st.bar_chart(dd.risk_on_date(risk, chosen))
        if base is not None:
            day = base[base["date"] == pd.Timestamp(chosen)].iloc[0]
            st.caption(f"Rainfall that day: {day['precip_mm']:.1f} mm; peak hour {day['max_hourly_mm']:.1f} mm.")
        borough = st.selectbox("Risk over time for", sorted(risk["borough"].unique()))
        series = risk[risk["borough"] == borough].set_index("date")["risk"]
        st.line_chart(series)
        st.caption("Risk is the model's predicted probability of a flood day (at least "
                   f"{dd.THRESHOLD} complaints). It is not a measured flood.")

with tab4:
    if whatif is None:
        not_ready("The what-if results")
    else:
        factors = sorted(whatif["factor"].unique())
        factor = st.select_slider("Rainfall multiplier", options=factors,
                                  format_func=lambda f: f"x{f:g} ({(f - 1):+.0%})")
        table = dd.whatif_for_factor(whatif, factor)
        st.bar_chart(table[["risk_baseline", "risk_scenario"]])
        st.dataframe(table.round(3))
        if factor > 1:
            st.warning("Scaled rainfall can go beyond what was observed in 2024-2026 "
                       "(maximum 65.6 mm per day, 16.1 mm per hour). Beyond that range the model "
                       "is extrapolating, so treat these as model outputs, not predictions.")