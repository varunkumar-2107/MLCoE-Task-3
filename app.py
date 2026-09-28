import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ART = Path("MOdels")

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚧", layout="wide")


@st.cache_resource
def load_artifacts():
    ohe = joblib.load(ART / "ohe.joblib")
    scaler = joblib.load(ART / "scaler.joblib")
    pca = joblib.load(ART / "pca.joblib")
    model = joblib.load(ART / "model.joblib")
    meta = json.load(open(ART / "meta.json"))
    return ohe, scaler, pca, model, meta


ohe, scaler, pca, model, meta = load_artifacts()

FEATURE_COLUMNS = meta["feature_columns"]      # exact column order used for StandardScaler/PCA
NUMERIC = meta["numeric"]                      # {col: {min, max, median, is_int}}
OHE_COLS = list(ohe.feature_names_in_)         # same list used in the notebook

# Same mappings as your OrdinalEncoders (index = encoded value)
YES_NO = ["No", "Yes"]
YES_NO_COLS = [
    "alcohol_involved", "median_present", "traffic_signal_present",
    "hospitalization_required", "police_called", "ambulance_called",
    "emergency_services_called", "fatigue_involved", "mobile_phone_used",
    "distraction_involved",
]
LIGHTING = ["Streetlight Absent", "Streetlight Present"]
VEH_COND = ["Poor", "Average", "Good"]
SEVERITY = ["Minor", "Moderate", "Severe", "Fatal"]


def label(c):
    return c.replace("_", " ").title()


st.title("🚧 Road Accident Severity Predictor")
st.caption("Fill in the accident details and click Predict.")

raw = {}

# ---------- numeric inputs (built automatically from training data ranges) ----------
st.subheader("Numeric details")
cols = st.columns(3)
for i, (c, s) in enumerate(NUMERIC.items()):
    with cols[i % 3]:
        if s["is_int"]:
            raw[c] = st.number_input(label(c), min_value=int(s["min"]), max_value=int(s["max"]),
                                     value=int(s["median"]), step=1, key=c)
        else:
            raw[c] = st.number_input(label(c), min_value=float(s["min"]), max_value=float(s["max"]),
                                     value=float(s["median"]), key=c)

# ---------- categorical (one-hot) inputs ----------
st.subheader("Categorical details")
cols = st.columns(3)
for i, (c, cats) in enumerate(zip(OHE_COLS, ohe.categories_)):
    with cols[i % 3]:
        raw[c] = st.selectbox(label(c), list(cats), key=c)

# ---------- yes/no + ordinal inputs ----------
st.subheader("Yes / No and condition details")
cols = st.columns(3)
for i, c in enumerate(YES_NO_COLS):
    with cols[i % 3]:
        raw[c] = st.selectbox(label(c), YES_NO, key=c)
with cols[0]:
    raw["road_lighting"] = st.selectbox("Road Lighting", LIGHTING)
with cols[1]:
    raw["vehicle_condition"] = st.selectbox("Vehicle Condition", VEH_COND)


def preprocess(raw: dict) -> pd.DataFrame:
    """Replicates Feature_engineering + Model notebooks, in the same order."""
    row = dict(raw)

    # 1) ordinal encoding
    for c in YES_NO_COLS:
        row[c] = YES_NO.index(row[c])
    row["road_lighting"] = LIGHTING.index(row["road_lighting"])
    row["vehicle_condition"] = VEH_COND.index(row["vehicle_condition"])

    # 2) one-hot encoding with the FITTED encoder
    ohe_in = pd.DataFrame([{c: row[c] for c in OHE_COLS}])
    enc = ohe.transform(ohe_in)
    enc = enc.toarray() if hasattr(enc, "toarray") else enc
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(OHE_COLS))

    base = pd.DataFrame([{k: v for k, v in row.items() if k not in OHE_COLS}])
    full = pd.concat([base, enc_df], axis=1)

    # 3) exact training column order (missing -> 0)
    full = full.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # 4) scale -> PCA
    scaled = scaler.transform(full)
    return pca.transform(scaled)


if st.button("Predict", type="primary"):
    x = preprocess(raw)
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]

    st.success(f"Predicted severity: **{SEVERITY[pred]}**")
    st.bar_chart(pd.DataFrame({"Probability": proba}, index=[SEVERITY[int(k)] for k in model.classes_]))
