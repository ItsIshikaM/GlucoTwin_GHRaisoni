import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="GlucoTwin Clinical Cockpit", layout="wide")

st.markdown("""
<div style="background-color: #0F172A; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
    <h2 style="color: #38BDF8; margin: 0;">🧬 GlucoTwin: Clinical Digital Twin Cockpit</h2>
    <p style="color: #94A3B8; margin: 0; font-size: 14px;">Reimagining & Reforming Healthcare Summit 2026 | Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Static EHR Selection
st.sidebar.header("📋 Electronic Health Record")
patients = {
    "PAT_012 (Uncontrolled T2DM)": {"age": 58, "bmi": 29.4, "hba1c": 8.7, "med": "Metformin 1000mg + Glimepiride", "base_g": 165},
    "PAT_045 (Moderate Risk)": {"age": 49, "bmi": 26.1, "hba1c": 7.4, "med": "Metformin 500mg BD", "base_g": 140},
    "PAT_089 (Well Controlled)": {"age": 42, "bmi": 23.8, "hba1c": 6.5, "med": "Lifestyle Only", "base_g": 115}
}

selected = st.sidebar.selectbox("Active Patient Twin", list(patients.keys()))
profile = patients[selected]

st.sidebar.markdown(f"""
- **Age:** {profile['age']} yrs
- **BMI:** {profile['bmi']} kg/m²
- **Baseline HbA1c:** {profile['hba1c']}%
- **Current Regimen:** {profile['med']}
""")

st.sidebar.markdown("---")
st.sidebar.header("🧪 Counterfactual 'What-If' Simulation")
st.sidebar.write("Simulate patient behavioral inputs:")
sim_carbs = st.sidebar.slider("Carbohydrate Intake (g)", 0, 120, 60, step=5)
sim_walk = st.sidebar.slider("Postprandial Brisk Walk (mins)", 0, 60, 15, step=5)

# Main Dashboard: Key Metrics
col1, col2, col3, col4 = st.columns(4)
current_glucose = profile["base_g"] + 10
col1.metric("Current CGM Glucose", f"{current_glucose} mg/dL", delta="+5 mg/dL", delta_color="inverse")
col2.metric("Mean HRV (RMSSD)", "36 ms", delta="-3 ms", delta_color="inverse")
col3.metric("Rolling 24h Steps", "6,200", delta="+600")
col4.metric("Twin Confidence", "96.2%", delta="Nominal")

st.markdown("---")

# Trajectory & Risk Alerts
c1, c2 = st.columns([3, 1])

with c1:
    st.subheader("2-Hour Forward Horizon Trajectory")
    
    time_past = [f"T-{i*5}m" for i in range(12, 0, -1)] + ["T-Now"]
    past_series = [current_glucose - (12 - i) * 2 + int(np.random.normal(0, 1.5)) for i in range(13)]
    
    time_future = ["T-Now"] + [f"T+{i*5}m" for i in range(1, 13)]
    
    # Mathematical response model of the twin
    delta = (sim_carbs * 0.9) - (sim_walk * 1.3)
    future_series = [past_series[-1] + (delta * (1 - np.exp(-0.25 * i))) for i in range(13)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_past, y=past_series, mode="lines+markers", name="Observed CGM", line=dict(color="#38BDF8", width=3)))
    fig.add_trace(go.Scatter(x=time_future, y=future_series, mode="lines+markers", name="Twin 2-Hr Forecast", line=dict(color="#FB923C", width=3, dash="dash")))
    
    fig.add_hline(y=180, line_dash="dot", line_color="#EF4444", annotation_text="Hyperglycemic Threshold (180 mg/dL)")
    fig.add_hline(y=70, line_dash="dot", line_color="#FACC15", annotation_text="Hypoglycemic Threshold (70 mg/dL)")
    
    fig.update_layout(height=420, xaxis_title="Timeline (5-min intervals)", yaxis_title="Glucose (mg/dL)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Clinical Alert Engine")
    terminal_g = future_series[-1]
    
    if terminal_g > 180:
        st.error(f"⚠️ **High Hyperglycemia Risk**\n\nForecasted: **{terminal_g:.1f} mg/dL**")
        st.warning("• Automated Nudge: 15–20 min light walk recommended immediately.")
    elif terminal_g < 70:
        st.error(f"🛑 **Hypoglycemia Alert**\n\nForecasted: **{terminal_g:.1f} mg/dL**")
        st.info("• Ingest 15g fast-acting oral glucose.")
    else:
        st.success(f"✅ **Target Range (Euglycemic)**\n\nForecasted: **{terminal_g:.1f} mg/dL**")

st.caption("🔒 Sandbox Mode: Compliant with India DPDP Act 2023. Synthetic data only.")