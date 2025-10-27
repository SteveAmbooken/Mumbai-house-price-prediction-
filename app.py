# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Maharashtra Housing Dashboard", layout="wide")

# Paths
DATA_PATH = Path("Mumbai House Prices.csv")
MODEL_PATH = Path("models/price_model.pkl")
META_PATH = Path("models/metadata.pkl")

# Load model, metadata, raw data
pipe = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
df_raw = pd.read_csv(DATA_PATH)

# Helpers
def to_inr(row):
    val = row["price"]
    unit = str(row["price_unit"]).strip().lower()
    if unit.startswith("cr"):
        return val * 1e7
    if unit.startswith("l"):
        return val * 1e5
    return val

# Prepare data
df_raw["price_inr"] = df_raw.apply(to_inr, axis=1)

# Sidebar
st.sidebar.title("Maharashtra Housing Dashboard")
mode = st.sidebar.radio("Select Mode", ["Price Prediction", "Data Visualization"])

# ============= Price Prediction =============
if mode == "Price Prediction":
    st.markdown("## Maharashtra's House Price Prediction")

    c1, c2 = st.columns([1, 1])

    with c1:
        area = st.slider(
            "Enter carpet area in sqft",
            min_value=int(meta["area_min"]),
            max_value=int(meta["area_max"]),
            value=int((meta["area_min"] + meta["area_max"]) // 2),
        )
        bhk = st.slider(
            "Choose number of BHK",
            min_value=int(meta["bhk_min"]),
            max_value=int(min(meta["bhk_max"], 8)),
            value=2,
        )

    with c2:
        region = st.selectbox("Region", options=meta["regions"])
        ptype = st.selectbox("Property Type", options=meta["types"])
        status = st.selectbox("Construction Status", options=meta["statuses"])
        age = st.number_input(
            "Age (years, use 0 if New)", min_value=0.0, max_value=60.0, value=0.0, step=0.5
        )

    if st.button("Predict House Price"):
        X_pred = {
            "area": area,
            "bhk": bhk,
            "type": ptype,
            "region": region,
            "status": status,
            "age": age,
        }
        X_pred_row = pd.DataFrame([X_pred])

        # Ensure helper columns are present if the pipeline expects them
        if "age_cat" not in X_pred_row.columns:
            X_pred_row["age_cat"] = "New" if float(age) == 0 else str(int(age))
        if "age_num" not in X_pred_row.columns:
            X_pred_row["age_num"] = float(age) if float(age) > 0 else 0.0

        # Predict (pipeline may be trained on log target)
        pred_log = pipe.predict(X_pred_row)
        y_hat = float(np.expm1(pred_log[0]) if meta.get("uses_log_target", False) else pred_log[0])

        st.success(f"Estimated Price for {bhk} BHK ({area} sqft) in {region}: ₹{y_hat:,.2f}")

# ============= Data Visualization =============
else:
    st.markdown("## Maharashtra Housing Data Insights")
    st.sidebar.subheader("Filter Options")

    # Sidebar filters
    bhk_values = sorted(df_raw["bhk"].dropna().unique().tolist())
    sel_bhk = st.sidebar.multiselect("Select Number of Bedrooms", bhk_values, default=bhk_values[:4])

    a_min, a_max = int(df_raw["area"].min()), int(df_raw["area"].max())
    sel_area = st.sidebar.slider(
        "Select Area (sqft) Range", min_value=a_min, max_value=a_max, value=(a_min, a_max)
    )

    region_values = sorted(df_raw["region"].dropna().astype(str).unique().tolist())
    sel_regions = st.sidebar.multiselect("Select Regions", region_values)

    # Apply filters to a working copy
    df = df_raw.copy()
    if sel_bhk:
        df = df[df["bhk"].isin(sel_bhk)]
    if sel_area:
        df = df[(df["area"] >= sel_area[0]) & (df["area"] <= sel_area[1])]
    if sel_regions:
        df = df[df["region"].isin(sel_regions)]

    # Data preview
    st.markdown("### Filtered Data Preview")
    show_cols = ["region", "price_inr", "area", "status", "type", "bhk", "age"]
    st.dataframe(df[show_cols].head(20), use_container_width=True)

    # Visualization selector
    st.markdown("### Select Visualization Type")
    viz = st.selectbox(
        "Select Visualization Type",
        [
            "Price Distribution",
            "Top 10 Expensive Regions",
            "Area vs Price Scatter",
            "Bedrooms vs Average Price",
            "Correlation Heatmap",
        ],
        index=0,
    )

    if not len(df):
        st.info("No data after filters.")
    else:
        if viz == "Price Distribution":
            fig = px.histogram(
                df, x="price_inr", nbins=60, title="Distribution of House Prices (Filtered View)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz == "Top 10 Expensive Regions":
            top = (
                df.groupby("region", as_index=False)["price_inr"]
                  .median()
                  .sort_values("price_inr", ascending=False)
                  .head(10)
            )
            fig = px.bar(
                top, x="region", y="price_inr",
                title="Top 10 Expensive Regions (Median Price)",
                color="price_inr", color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz == "Area vs Price Scatter":
            fig = px.scatter(
                df,
                x="area",
                y="price_inr",
                color="bhk" if "bhk" in df.columns else None,
                hover_data=["region", "status", "type"] if set(["region","status","type"]).issubset(df.columns) else None,
                trendline="lowess",
                title="Area vs Price (Filtered View)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz == "Bedrooms vs Average Price":
            by_bhk = (
                df.groupby("bhk", as_index=False)["price_inr"]
                  .mean()
                  .sort_values("bhk")
            )
            fig = px.bar(
                by_bhk, x="bhk", y="price_inr",
                title="Average Price by Bedrooms (Filtered View)",
                text_auto=".2s", color="price_inr", color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz == "Correlation Heatmap":
            # Select numeric columns available
            num_cols = []
            for c in ["price_inr", "area", "bhk", "age"]:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    num_cols.append(c)
            if len(num_cols) < 2:
                st.info("Not enough numeric columns for correlation.")
            else:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", origin="lower",
                    title="Correlation Heatmap (Filtered View)"
                )
                st.plotly_chart(fig, use_container_width=True)
