"""
Simple checks for the project. Run from the project root:
    python -m pip install pytest pandas
    python -m pytest tests -v
Tests that need a file which does not exist yet are skipped, not failed.
"""
from pathlib import Path

import pandas as pd
import pytest

import dashboard_data as dd

ROOT = Path(__file__).resolve().parents[1]
BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}


def need(name):
    df = dd.load(name, ROOT)
    if df is None:
        pytest.skip(f"{dd.FILES[name][0]} not available yet")
    return df


def test_processed_table_shape():
    df = need("borough_day")
    assert set(df["borough"].unique()) == BOROUGHS
    assert len(df) == df["date"].nunique() * 5          # one row per borough per day
    assert not df.duplicated(["borough", "date"]).any()
    assert df.isna().sum().sum() == 0


def test_label_rule_matches_documented_share():
    df = need("borough_day")
    s = dd.split_summary(df)
    assert 0.09 < s["train_flood_share"] < 0.13        # about 10.8% documented


def test_train_test_split_has_no_overlap():
    df = need("borough_day")
    train = df[df["date"] < pd.Timestamp(dd.TRAIN_END)]
    test = df[df["date"] >= pd.Timestamp(dd.TRAIN_END)]
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(df)


def test_label_does_not_change_with_other_columns():
    df = pd.DataFrame({"flood_count": [0, 3, 4, 9], "borough": ["BRONX"] * 4,
                       "date": pd.to_datetime(["2024-01-01"] * 4)})
    assert dd.with_label(df)["is_flood_day"].tolist() == [0, 0, 1, 1]


def test_metrics_file_follows_contract():
    df = need("metrics")
    assert df[["precision", "recall", "f1", "roc_auc"]].apply(lambda c: c.between(0, 1)).all().all()


def test_risk_table_follows_contract():
    df = need("risk")
    assert df["risk"].between(0, 1).all()
    assert set(df["borough"].unique()) == BOROUGHS


def test_whatif_factor_one_equals_baseline():
    df = need("whatif")
    one = df[df["factor"] == 1.0]
    if one.empty:
        pytest.skip("no factor 1.0 rows in whatif_results.csv")
    assert (one["risk_baseline"] - one["risk_scenario"]).abs().max() < 1e-9


def test_whatif_more_rain_does_not_lower_average_risk():
    df = need("whatif")
    avg = df.groupby("factor")["risk_scenario"].mean()
    assert avg.is_monotonic_increasing


def test_best_model_file_loads():
    files = list((ROOT / "models").glob("*.joblib"))
    if not files:
        pytest.skip("no model file in models/ yet")
    import joblib
    assert joblib.load(files[0]) is not None


def test_dashboard_reports_missing_columns(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "risk_table.csv").write_text("date,borough\n2026-01-01,BRONX\n")
    with pytest.raises(ValueError, match="missing columns"):
        dd.load("risk", tmp_path)