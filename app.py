import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = "models/cyber_threat_model.joblib"
DATA_PATH = "data/cyber_threat_dataset.csv"

st.set_page_config(page_title="AI Cyber Threat Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ AI-Based Cyber Threat Detection")
st.caption("Machine-learning framework for network traffic threat classification")
st.divider()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_data
def load_demo():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

model = load_model()
demo = load_demo()

if model is None:
    st.error("Model not found. Run: python train_model.py")
    st.stop()

required = [
    "duration","protocol","src_bytes","dst_bytes","packets","src_port",
    "dst_port","failed_logins","connection_rate","login_attempts","payload_size"
]

with st.sidebar:
    st.header("⚙️ Data Source")
    uploaded = st.file_uploader("Upload network CSV", type=["csv"])
    use_demo = st.checkbox("Use built-in demo dataset", value=uploaded is None)
    st.markdown("**Required feature columns**")
    st.code("\n".join(required))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_demo and demo is not None:
    df = demo.copy()
else:
    st.warning("Upload a CSV or select the built-in demo dataset.")
    st.stop()

missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

X = df[required].copy()
pred = model.predict(X)
proba = model.predict_proba(X)
confidence = np.max(proba, axis=1) * 100

result = df.copy()
result["prediction"] = pred
result["confidence_%"] = np.round(confidence, 2)
result["risk"] = np.where(result["prediction"].eq("Normal"), "Low", "High")

total = len(result)
threats = int((result["prediction"] != "Normal").sum())
normal = total - threats

a,b,c,d = st.columns(4)
a.metric("Total Records", f"{total:,}")
b.metric("Threats Detected", f"{threats:,}")
c.metric("Normal Traffic", f"{normal:,}")
d.metric("Threat Rate", f"{(threats/total*100 if total else 0):.1f}%")

left,right = st.columns(2)
counts = result["prediction"].value_counts().reset_index()
counts.columns = ["Threat Type","Count"]

with left:
    st.plotly_chart(px.bar(counts, x="Threat Type", y="Count", title="Traffic Classification"),
                    use_container_width=True)
with right:
    st.plotly_chart(px.pie(counts, names="Threat Type", values="Count",
                            title="Threat Distribution", hole=.4),
                    use_container_width=True)

st.subheader("🚨 Threat Summary")
summary = result[result["prediction"] != "Normal"]["prediction"].value_counts().reset_index()
summary.columns = ["Threat Type","Detected"]
if len(summary):
    st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.success("No threats detected.")

st.subheader("🔎 Prediction Results")
display_cols = required + ["prediction","confidence_%","risk"]
st.dataframe(result[display_cols].head(500), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Prediction Report",
    result.to_csv(index=False).encode("utf-8"),
    "cyber_threat_predictions.csv",
    "text/csv"
)

if "label" in df.columns:
    st.divider()
    st.subheader("📊 Model Evaluation")
    y_true, y_pred = df["label"].astype(str), result["prediction"].astype(str)
    metrics = [
        accuracy_score(y_true,y_pred),
        precision_score(y_true,y_pred,average="weighted",zero_division=0),
        recall_score(y_true,y_pred,average="weighted",zero_division=0),
        f1_score(y_true,y_pred,average="weighted",zero_division=0)
    ]
    m1,m2,m3,m4 = st.columns(4)
    for col,val,name in zip([m1,m2,m3,m4],metrics,["Accuracy","Precision","Recall","F1 Score"]):
        col.metric(name, f"{val*100:.2f}%")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                    labels=dict(x="Predicted",y="Actual",color="Count"),
                    title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Academic prototype: structured network-flow classification. Not a production IDS.")
