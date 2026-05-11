"""
Streamlit dashboard for Indian stock market predictions.
Run with: streamlit run src/dashboard/app.py
"""

import sys
import json
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prediction.predict import StockPredictor
from ml_pipeline.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stock registry ────────────────────────────────────────────────────────────
STOCKS = {
    "LT_NS": "Larsen & Turbo",
    "ADANIENT_NS": "Adani Enterprises",
    "COALINDIA_NS": "Coal India",
    "BBOX_NS": "Black Box",
    "TATACOMM_NS": "Tata Communications",
    "INFY_NS": "Infosys",
}

MODEL_DIR = Path("data/models")
PRICE_DIR = Path("data/prices")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("AI-powered predictions for NSE stocks")

view_mode = st.sidebar.radio("View Mode", ["Individual Stock", "Portfolio View"])

if view_mode == "Individual Stock":
    selected_key = st.sidebar.selectbox(
        "Select Stock",
        list(STOCKS.keys()),
        format_func=lambda k: f"{STOCKS[k]} ({k.replace('_', '.')})",
    )
    stocks_to_show = [selected_key]
else:
    stocks_to_show = list(STOCKS.keys())

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# ── Helper: load metadata ─────────────────────────────────────────────────────

def load_metadata(safe_symbol: str) -> dict:
    path = MODEL_DIR / f"{safe_symbol}_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Helper: load predictions (cached) ────────────────────────────────────────

@st.cache_data(ttl=300)
def get_predictions(symbols: tuple) -> dict:
    """Cached prediction loader — reruns at most every 5 minutes."""
    engineer = FeatureEngineer()
    predictor = StockPredictor()
    results = {}

    for sym in symbols:
        try:
            X, _, _ = engineer.combine_all_features(sym)
            pred = predictor.predict_stock(sym, X.tail(1))
            results[sym] = pred
        except FileNotFoundError:
            results[sym] = {
                "stock_symbol": sym,
                "prediction": "NO MODEL",
                "probability_up": 0.5,
                "probability_down": 0.5,
                "confidence": 0.0,
                "error": "Model or data not found. Run run_pipeline.py first.",
            }
        except Exception as e:
            results[sym] = {
                "stock_symbol": sym,
                "prediction": "ERROR",
                "probability_up": 0.5,
                "probability_down": 0.5,
                "confidence": 0.0,
                "error": str(e),
            }
    return results


# ── Helper: colour for prediction ────────────────────────────────────────────

def pred_colour(p: str) -> str:
    return {"UP": "🟢", "DOWN": "🔴", "NEUTRAL": "⚪"}.get(p, "⚫")


# ── Helper: confidence gauge ──────────────────────────────────────────────────

def confidence_gauge(value: float, title: str = "Confidence") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#ffcccc"},
                    {"range": [50, 70], "color": "#fff0cc"},
                    {"range": [70, 100], "color": "#ccffcc"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Helper: price chart ───────────────────────────────────────────────────────

def price_chart(safe_symbol: str, days: int = 90) -> go.Figure | None:
    candidates = [
        PRICE_DIR / f"{safe_symbol}_prices.csv",
        PRICE_DIR / f"{safe_symbol.replace('_', '.')}_prices.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.tail(days)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            if "sma_20" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["sma_20"],
                        mode="lines",
                        name="SMA 20",
                        line=dict(color="orange", width=1, dash="dash"),
                    )
                )
            fig.update_layout(
                title=f"{days}-Day Price History",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                height=350,
                margin=dict(t=50, b=30),
            )
            return fig
    return None


# ── Main content ──────────────────────────────────────────────────────────────
st.title("📈 Stock Market Prediction Dashboard")
st.caption("ML predictions for NSE-listed Indian stocks  |  Data: yfinance  |  Model: XGBoost")

preds = get_predictions(tuple(stocks_to_show))

# ─────────────────────── INDIVIDUAL STOCK VIEW ────────────────────────────────
if view_mode == "Individual Stock":
    sym = selected_key
    pred = preds.get(sym, {})
    meta = load_metadata(sym)

    st.header(f"{STOCKS[sym]}  ({sym.replace('_', '.')})")

    if "error" in pred and pred["prediction"] in ("NO MODEL", "ERROR"):
        st.error(pred.get("error", "Unknown error"))
        st.info("Run `python run_pipeline.py` to fetch data and train models.")
    else:
        direction = pred.get("prediction", "NEUTRAL")
        conf = pred.get("confidence", 0.0)
        p_up = pred.get("probability_up", 0.5)
        p_down = pred.get("probability_down", 0.5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", f"{pred_colour(direction)} {direction}")
        col2.metric("Confidence", f"{conf:.1%}")
        col3.metric("P(UP)", f"{p_up:.1%}")
        col4.metric("P(DOWN)", f"{p_down:.1%}")

        # Gauge + prob bar side by side
        g_col, b_col = st.columns([1, 1])
        with g_col:
            st.plotly_chart(confidence_gauge(conf), width='stretch')
        with b_col:
            st.subheader("Direction Probability")
            prob_fig = go.Figure(
                go.Bar(
                    x=["UP ▲", "DOWN ▼"],
                    y=[p_up * 100, p_down * 100],
                    marker_color=["green", "red"],
                    text=[f"{p_up:.1%}", f"{p_down:.1%}"],
                    textposition="outside",
                )
            )
            prob_fig.update_layout(
                yaxis=dict(range=[0, 110], title="Probability (%)"),
                height=250,
                margin=dict(t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(prob_fig, width='stretch')

        # Price history
        st.subheader("Historical Prices")
        fig = price_chart(sym)
        if fig:
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Price data not found. Run step 1 of the pipeline.")

        # Model metrics
        if meta.get("metrics"):
            st.subheader("Model Performance")
            m = meta["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
            mc2.metric("Precision", f"{m.get('precision', 0):.1%}")
            mc3.metric("Recall", f"{m.get('recall', 0):.1%}")
            mc4.metric("F1 Score", f"{m.get('f1_score', 0):.1%}")

        # Feature importance
        if meta.get("feature_names"):
            model_path = MODEL_DIR / f"{sym}_metadata.json"
            imp_path = MODEL_DIR / f"xgboost_{sym}.pkl"
            if imp_path.exists():
                import joblib
                model_obj = joblib.load(imp_path)
                feat_names = meta["feature_names"]
                importances = model_obj.feature_importances_
                imp_df = (
                    pd.DataFrame({"Feature": feat_names, "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(15)
                )
                st.subheader("Top 15 Feature Importances")
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="blues",
                )
                fig_imp.update_layout(height=400, margin=dict(t=20))
                st.plotly_chart(fig_imp, width='stretch')

# ─────────────────────── PORTFOLIO VIEW ───────────────────────────────────────
else:
    st.header("Portfolio Overview")

    # Aggregate portfolio score (confidence-weighted)
    total_score, total_weight = 0.0, 0.0
    direction_map = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}
    for pred in preds.values():
        d = direction_map.get(pred.get("prediction", "NEUTRAL"), 0.0)
        w = pred.get("confidence", 0.0)
        total_score += d * w
        total_weight += w

    portfolio_score = total_score / total_weight if total_weight > 0 else 0.0
    if portfolio_score > 0.05:
        overall = "UP"
    elif portfolio_score < -0.05:
        overall = "DOWN"
    else:
        overall = "NEUTRAL"

    c1, c2 = st.columns(2)
    c1.metric(
        "Overall Portfolio Signal",
        f"{pred_colour(overall)} {overall}",
        help="Confidence-weighted aggregate of all stock predictions",
    )
    c2.metric("Aggregate Score", f"{portfolio_score:+.3f}", help="-1 (all DOWN) to +1 (all UP)")

    # Individual stock table
    st.subheader("Individual Stock Predictions")
    rows = []
    for sym, pred in preds.items():
        rows.append({
            "Stock": STOCKS[sym],
            "Symbol": sym.replace("_", "."),
            "Signal": f"{pred_colour(pred['prediction'])} {pred['prediction']}",
            "P(UP)": f"{pred['probability_up']:.1%}",
            "P(DOWN)": f"{pred['probability_down']:.1%}",
            "Confidence": f"{pred['confidence']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # Pie chart
    up_count = sum(1 for p in preds.values() if p["prediction"] == "UP")
    down_count = sum(1 for p in preds.values() if p["prediction"] == "DOWN")
    neutral_count = len(preds) - up_count - down_count

    labels, values, colours = [], [], []
    for label, cnt, col in [("UP", up_count, "green"), ("DOWN", down_count, "red"),
                              ("NEUTRAL", neutral_count, "gray")]:
        if cnt > 0:
            labels.append(label)
            values.append(cnt)
            colours.append(col)

    if values:
        pie_col, bar_col = st.columns(2)
        with pie_col:
            st.subheader("Signal Distribution")
            pie_fig = go.Figure(
                go.Pie(
                    labels=labels, values=values,
                    hole=0.35, marker_colors=colours,
                )
            )
            pie_fig.update_layout(height=350)
            st.plotly_chart(pie_fig, width='stretch')

        with bar_col:
            st.subheader("Confidence by Stock")
            conf_data = {STOCKS[s]: preds[s]["confidence"] for s in stocks_to_show}
            bar_fig = px.bar(
                x=list(conf_data.keys()),
                y=[v * 100 for v in conf_data.values()],
                labels={"x": "Stock", "y": "Confidence (%)"},
                color=list(conf_data.values()),
                color_continuous_scale="blues",
            )
            bar_fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(bar_fig, width='stretch')

    # Model metrics summary table
    metric_rows = []
    for sym in STOCKS:
        meta = load_metadata(sym)
        m = meta.get("metrics", {})
        if m:
            metric_rows.append({
                "Stock": STOCKS[sym],
                "Accuracy": f"{m.get('accuracy', 0):.1%}",
                "Precision": f"{m.get('precision', 0):.1%}",
                "Recall": f"{m.get('recall', 0):.1%}",
                "F1": f"{m.get('f1_score', 0):.1%}",
            })
    if metric_rows:
        st.subheader("Model Performance Summary")
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center'>"
    "⚠️ <b>Disclaimer:</b> Predictions are for educational purposes only. "
    "Not financial advice. Always do your own research before investing."
    "</div>",
    unsafe_allow_html=True,
)
