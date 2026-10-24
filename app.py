from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scrapless.sample_data import sample_order
from scrapless.schemas import validate_order_frame
from scrapless.packing import first_fit_decreasing, maxrects_bssf, validate_layout
from scrapless.metrics import calculate_metrics, compare_metrics, metrics_frame
from scrapless.ml_model import predict_scrap_risk
from scrapless.reporting import to_csv_bytes
from scrapless.vision import analyze_material_image

st.set_page_config(page_title="ScrapLess AI", page_icon="♻", layout="wide")
st.title("ScrapLess AI")
st.caption("Turn material waste into measurable savings. A deterministic cutting and analytics prototype.")

with st.sidebar:
    st.header("Production setup")
    unit = st.selectbox("Dimension unit", ["mm", "cm", "inches"])
    sheet_width = st.number_input("Stock sheet width", min_value=1.0, value=2440.0, step=10.0)
    sheet_height = st.number_input("Stock sheet length", min_value=1.0, value=1220.0, step=10.0)
    sheet_cost = st.number_input("Cost per stock sheet", min_value=0.0, value=1200.0, step=50.0)
    currency = st.text_input("Currency label", "INR")
    spacing = st.number_input("Kerf / spacing allowance", min_value=0.0, value=0.0, step=0.5,
                               help="Reserved on the right and top of each placed part; not subtracted from part area.")
    allow_rotation = st.toggle("Allow 90-degree rotation", value=True)
    max_sheets = st.number_input("Maximum sheets to evaluate", min_value=1, max_value=1000, value=100, step=1)
    risk_threshold = st.slider("Illustrative high-scrap threshold", 0.05, 0.95, 0.50, 0.05)

tab_order, tab_base, tab_opt, tab_compare, tab_ai, tab_vision = st.tabs(
    ["Order Input", "Baseline Layout", "Optimized Layout", "Comparison", "AI Scrap Risk", "Computer Vision"]
)

with tab_order:
    uploaded = st.file_uploader("Upload order CSV", type=["csv"])
    raw = pd.read_csv(uploaded) if uploaded else sample_order()
    edited = st.data_editor(raw, num_rows="dynamic", use_container_width=True)
    st.download_button("Download sample CSV", sample_order().to_csv(index=False).encode(), "sample_orders.csv", "text/csv")
    validation = validate_order_frame(edited)
    if validation.valid:
        total = int(validation.order["quantity"].sum())
        area = float((validation.order["length"] * validation.order["width"] * validation.order["quantity"]).sum())
        st.success(validation.message)
        st.write(f"**{total:,}** part instances | **{area:,.0f} {unit}²** requested area")
    else:
        st.error(validation.message)
    run = st.button("Run Analysis", type="primary", disabled=not validation.valid)
    if run:
        order = validation.order
        ffd = first_fit_decreasing(order, sheet_width, sheet_height, allow_rotation, spacing, int(max_sheets))
        mx = maxrects_bssf(order, sheet_width, sheet_height, allow_rotation, spacing, int(max_sheets))
        vals = [validate_layout(ffd), validate_layout(mx)]
        mets = [calculate_metrics(r, sheet_cost, v) for r, v in zip([ffd, mx], vals)]
        risk = predict_scrap_risk(order, sheet_width, sheet_height)
        comparison = compare_metrics(mets[0], mets[1])
        st.session_state["analysis"] = {"order": order, "results": [ffd, mx], "metrics": mets,
                                        "validations": vals, "risk": risk, "comparison": comparison,
                                        "currency": currency}

def layout_chart(result, sheet_choice):
    rows = result.placements_frame()
    rows = rows[rows.sheet_index == sheet_choice].reset_index(drop=True)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=result.sheet_width, y1=result.sheet_height,
                  line=dict(color="#1e69b4", width=3), fillcolor="white")
    colors = ["#20a39e", "#4d91d1", "#e89a3c", "#7a6fd0", "#7caa62"]
    for i, row in rows.iterrows():
        fig.add_shape(type="rect", x0=row.x, y0=row.y, x1=row.x + row.width, y1=row.y + row.height,
                      fillcolor=colors[i % len(colors)], line=dict(color="white"))
        fig.add_annotation(x=row.x + row.width / 2, y=row.y + row.height / 2,
                           text=f"{row.part_id}-{row.instance}", showarrow=False)
    fig.update_layout(height=520, title=f"{result.algorithm} — Sheet {sheet_choice + 1}",
                      xaxis_title="Width", yaxis_title="Length", hovermode="closest")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig

def show_layout(result):
    if not result.sheet_count:
        st.warning("No sheets were opened; all parts are unplaced.")
        return
    sheet = st.selectbox("Sheet to inspect", range(result.sheet_count), format_func=lambda x: str(x + 1),
                         key=f"sheet-{result.algorithm}")
    st.plotly_chart(layout_chart(result, sheet), use_container_width=True)
    st.dataframe(result.placements_frame()[lambda f: f.sheet_index == sheet], use_container_width=True)
    if result.unplaced:
        st.warning("Unplaced instances")
        st.dataframe(pd.DataFrame([p.__dict__ for p in result.unplaced]), use_container_width=True)

analysis = st.session_state.get("analysis")
if analysis:
    baseline, optimized = analysis["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline sheets", baseline.sheets_used)
    c2.metric("Optimized sheets", optimized.sheets_used)
    c3.metric("Optimized utilization", f"{optimized.utilization_pct:.1f}%")
    c4.metric("Cost difference", f"{analysis['currency']} {analysis['comparison']['estimated_cost_saved']:,.2f}")
    c5.metric("ML scrap risk", analysis["risk"]["risk_label"])
    with tab_base:
        show_layout(analysis["results"][0])
    with tab_opt:
        show_layout(analysis["results"][1])
    with tab_compare:
        st.dataframe(metrics_frame(analysis["metrics"]), use_container_width=True)
        st.bar_chart(pd.DataFrame({"Sheets": [baseline.sheets_used, optimized.sheets_used],
                                   "Utilization (%)": [baseline.utilization_pct, optimized.utilization_pct]},
                                  index=["Baseline", "Optimized"]))
        if any(not v["valid"] for v in analysis["validations"]) or any(m.parts_unplaced for m in analysis["metrics"]):
            st.warning("Review unplaced parts or validation issues before interpreting savings.")
        comparison_rows = pd.DataFrame([
            {"Measure": "Sheets used", "Baseline": baseline.sheets_used, "Optimized": optimized.sheets_used,
             "Change": analysis["comparison"]["sheets_saved"]},
            {"Measure": "Material utilization (%)", "Baseline": round(baseline.utilization_pct, 2),
             "Optimized": round(optimized.utilization_pct, 2),
             "Change": round(analysis["comparison"]["utilization_gain_percentage_points"], 2)},
            {"Measure": "Unused area", "Baseline": round(baseline.unused_area, 2),
             "Optimized": round(optimized.unused_area, 2),
             "Change": round(analysis["comparison"]["unused_area_reduction"], 2)},
            {"Measure": f"Estimated material cost ({analysis['currency']})",
             "Baseline": round(baseline.estimated_material_cost, 2),
             "Optimized": round(optimized.estimated_material_cost, 2),
             "Change": round(analysis["comparison"]["estimated_cost_saved"], 2)},
            {"Measure": "Unplaced parts", "Baseline": baseline.parts_unplaced,
             "Optimized": optimized.parts_unplaced,
             "Change": analysis["comparison"]["unplaced_change"]},
            {"Measure": "Runtime (seconds)", "Baseline": round(baseline.runtime_seconds, 6),
             "Optimized": round(optimized.runtime_seconds, 6),
             "Change": round(analysis["comparison"]["runtime_change_seconds"], 6)},
        ])
        st.subheader("Calculated comparison")
        st.dataframe(comparison_rows, hide_index=True, use_container_width=True)
        st.download_button("Download placements CSV",
                           to_csv_bytes(pd.concat([r.placements_frame().assign(algorithm=r.algorithm)
                                                   for r in analysis["results"]])), "placements.csv")
    with tab_ai:
        risk = analysis["risk"]
        st.warning("This is a model score learned from synthetic proxy labels, not a factory-certified probability.")
        st.metric("Predicted risk", risk["risk_label"], f"{risk['probability']:.1%} score")
        evaluation = risk["evaluation"]
        st.subheader("Model evaluation")
        eval_cols = st.columns(4)
        eval_cols[0].metric("Accuracy", f"{evaluation['accuracy']:.1%}")
        eval_cols[1].metric("F1 score", f"{evaluation['f1']:.1%}")
        eval_cols[2].metric("Test rows", f"{evaluation['test_rows']:,}")
        eval_cols[3].metric("Training rows", f"{evaluation['training_rows']:,}")
        st.caption("Evaluation uses a held-out synthetic test split. It measures consistency with the synthetic proxy labels, not factory performance.")
        st.dataframe(pd.DataFrame([
            {"Metric": "Accuracy", "Value": f"{evaluation['accuracy']:.1%}"},
            {"Metric": "F1 score", "Value": f"{evaluation['f1']:.1%}"},
            {"Metric": "Training rows", "Value": f"{evaluation['training_rows']:,}"},
            {"Metric": "Test rows", "Value": f"{evaluation['test_rows']:,}"},
            {"Metric": "Data basis", "Value": "Synthetic proxy labels"},
        ]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(risk["feature_importances"]), use_container_width=True)
    with tab_vision:
        st.info("Experimental contour detection only; pixels are not converted to millimetres.")

with tab_vision:
    image_file = st.file_uploader("Upload a material image", type=["png", "jpg", "jpeg"], key="vision-file")
    min_area = st.number_input("Minimum contour area (pixels)", min_value=10.0, value=500.0)
    if image_file:
        from PIL import Image
        try:
            vision = analyze_material_image(Image.open(BytesIO(image_file.getvalue())), min_area=min_area)
            st.image([vision.image, vision.overlay], caption=["Original", "Detected candidates"])
            st.write(vision.note)
            st.dataframe(pd.DataFrame(vision.regions), use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))
