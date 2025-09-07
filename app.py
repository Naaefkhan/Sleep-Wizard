import streamlit as st
import joblib, json
import pandas as pd
import numpy as np
import plotly.express as px

# Load model & features
pipe = joblib.load("sleep_disruption_model.pkl")
with open("feature_list.json") as f:
    feature_list = json.load(f)
features = json.load(open("feature_list.json"))

# Page settings
st.set_page_config(page_title="Sleep Disruption Risk Predictor", page_icon="😴", layout="wide")

# Title
st.title("😴 Sleep Disruption Risk Predictor")
st.markdown("### Adjust your nightly habits and see how they affect your sleep quality.")

# Sidebar for user inputs
st.sidebar.header("📊 Enter Your Habits")

sample = {}
for f in features:
    if f in ["dnd_enabled_night","caffeine_after_18","exam_period_flag"]:
        sample[f] = st.sidebar.selectbox(
            f.replace("_"," ").title(),
            [0,1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )
    elif f == "day_of_week":
        sample[f] = st.sidebar.selectbox(
            "Day of Week",
            [0,1,2,3,4,5,6],
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )
    elif "22_24" in f:  # Late night screen time
        t = st.sidebar.slider("📱 Screen Time (10 PM – 12 AM)", 0, 120, 30, step=5)
        sample[f] = t
    elif "18" in f:  # Caffeine after 6 PM
        t = st.sidebar.slider("☕ Caffeine after 6 PM? (Yes=1, No=0)", 0, 1, 0)
        sample[f] = t
    elif "brightness" in f:
        sample[f] = st.sidebar.slider("💡 Screen Brightness", 0.0, 1.0, 0.5, 0.1)
    else:
        # Generic input with hours & minutes
        hours = st.sidebar.number_input(f"{f.replace('_',' ').title()} (Hours)", min_value=0, max_value=12, value=1)
        minutes = st.sidebar.number_input(f"{f.replace('_',' ').title()} (Minutes)", min_value=0, max_value=59, value=0)
        sample[f] = hours*60 + minutes  # convert to minutes total

# Convert to DataFrame
sample_df = pd.DataFrame([sample])

# Prediction
risk = pipe.predict_proba(sample_df)[0][1]
score = int((1 - risk) * 100)

# --- MAIN RESULTS ---
st.subheader("📌 Predictions")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probability of Poor Sleep", value=f"{risk:.2f}")

with col2:
    st.metric(label="🌙 Sleep Hygiene Score", value=f"{score}/100")

# Risk progress bar
st.progress(int(risk*100))

# --- FEATURE IMPORTANCE (Plotly) ---
st.subheader("📊 Feature Importance")
try:
    clf = pipe.named_steps["classifier"]

    if hasattr(clf, "coef_"):  # Logistic Regression
        importances = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):  # RandomForest, etc.
        importances = clf.feature_importances_
    else:
        raise ValueError("Model type not supported for importance.")

    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", key=abs, ascending=False)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="RdBu",
        title="Which habits impact sleep the most?",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Feature importance not available: {e}")

# --- SIMULATED SLEEP DISTRIBUTION (Plotly) ---
st.subheader("📊 Sleep Quality Distribution (Simulated)")
simulated = np.random.choice(["Good Sleep","Poor Sleep"], size=200, p=[1-risk, risk])
simulated_df = pd.DataFrame({"Sleep Quality": simulated})

fig2 = px.histogram(
    simulated_df,
    x="Sleep Quality",
    color="Sleep Quality",
    color_discrete_map={"Good Sleep": "green", "Poor Sleep": "red"},
    title="Distribution of Sleep Outcomes",
    text_auto=True
)
st.plotly_chart(fig2, use_container_width=True)

# --- TIP BOX ---
st.markdown("---")
if score >= 70:
    st.success("✅ Great sleep hygiene! Keep it up 🌙")
elif score >= 40:
    st.warning("⚠️ Moderate risk. Try reducing screen/caffeine in the evening.")
else:
    st.error("❌ High risk! Limit late-night screens & caffeine to improve sleep.")


