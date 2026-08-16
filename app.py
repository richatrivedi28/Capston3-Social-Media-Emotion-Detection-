import os
import glob
import pickle
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Emotion Predictor", page_icon="🙂", layout="centered")
st.title("🌟📱 Social Media — Dominant Emotion Predictor")
st.info("Two ways to predict: 1) Use the sidebar inputs and click Predict (single row). 2) Upload a CSV and download predictions (Multiple Rows at a time).")


def load_model(fname=None):
    """Load a specific .pkl model if fname provided, otherwise try common candidates."""
    candidates = []
    if fname:
        candidates = [fname]
    else:
        candidates = [
            "optimized_random_forest_model.pkl",
            "optimized_xgboost_model.pkl",
        
        ]
    for f in candidates:
        if not os.path.exists(f):
            continue
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh), f
        except Exception:
            pass
    return None, None


def find_models():
    # prefer known candidates but include any other .pkl in folder
    known = ["optimized_random_forest_model.pkl", "optimized_xgboost_model.pkl"]
    found = [p for p in known if os.path.exists(p)]
    for p in glob.glob("*.pkl"):
        if p not in found:
            found.append(p)
    return found


# model selection in sidebar
available_models = find_models()
selected_model = None
if available_models:
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    MODEL, model_file = load_model(selected_model)
else:
    MODEL, model_file = None, None

model_name = None
if MODEL is None:
    st.warning("Model not found. Place optimized_random_forest_model.pkl or optimized_xgboost_model.pkl in this folder.")
else:
    if model_file:
        mlf = model_file.lower()
        if "random_forest" in mlf:
            model_name = "Random Forest"
        elif "xgboost" in mlf:
            model_name = "XGBoost"
        elif "compat" in mlf:
            model_name = "Compatibility model"
        else:
            model_name = os.path.splitext(model_file)[0]
        st.sidebar.success(f"Loaded model: {model_name} ({model_file})")


def build_input_df(age, gender, platform, daily_minutes, posts_per_day, likes_per_day, comments_per_day, messages_per_day):
    d = {
        "Daily_Usage_Time (minutes)": [daily_minutes],
        "Posts_Per_Day": [posts_per_day],
        "Likes_Received_Per_Day": [likes_per_day],
        "Comments_Received_Per_Day": [comments_per_day],
        "Messages_Sent_Per_Day": [messages_per_day],
        "Age": [age],
        "Gender": [gender],
        "Platform": [platform],
    }
    return pd.DataFrame(d)


with st.sidebar.form("input_form"):
    st.header("Single Prediction")
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Unknown"])
    platform = st.selectbox("Platform", ["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Whatsapp", "Telegram"])
    daily_minutes = st.number_input("Daily Usage Time (minutes)", min_value=0, max_value=1440, value=90)
    posts_per_day = st.number_input("Posts Per Day", min_value=0, max_value=100, value=2)
    likes_per_day = st.number_input("Likes Received Per Day", min_value=0, max_value=10000, value=40)
    comments_per_day = st.number_input("Comments Received Per Day", min_value=0, max_value=1000, value=15)
    messages_per_day = st.number_input("Messages Sent Per Day", min_value=0, max_value=1000, value=20)
    submit_single = st.form_submit_button("🔮 Predict")

if submit_single:
    if MODEL is None:
        st.error("Model not available.")
    else:
        df_single = build_input_df(age, gender, platform, daily_minutes, posts_per_day, likes_per_day, comments_per_day, messages_per_day)
        try:
            pred = MODEL.predict(df_single)
            label_map = ["Anger", "Anxiety", "Boredom", "Happiness", "Neutral", "Sadness"]
            try:
                pred_int = int(pred[0])
                pred_label = label_map[pred_int] if 0 <= pred_int < len(label_map) else str(pred_int)
            except Exception:
                pred_label = str(pred[0])
            st.success(f"Predicted dominant emotion: {pred_label}")
            if hasattr(MODEL, "predict_proba"):
                try:
                    probs = MODEL.predict_proba(df_single)[0]
                    probs_df = pd.DataFrame({"emotion": label_map[: len(probs)], "probability": probs})
                    st.table(probs_df)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.header("Batch predict from CSV")
uploaded_file = st.file_uploader("Upload CSV (header required)", type=["csv"])
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df_upload = None

    if df_upload is not None:
        st.write("Preview:")
        st.dataframe(df_upload.head())

        # drop target column if present
        if 'Dominant_Emotion' in df_upload.columns:
            df_upload = df_upload.drop(columns=['Dominant_Emotion'])

        # required columns
        required_cols = [
            "Daily_Usage_Time (minutes)",
            "Posts_Per_Day",
            "Likes_Received_Per_Day",
            "Comments_Received_Per_Day",
            "Messages_Sent_Per_Day",
            "Age",
            "Gender",
            "Platform",
        ]
        missing = [c for c in required_cols if c not in df_upload.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # coerce numeric columns
            numeric_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day", "Age"]
            for col in numeric_cols:
                df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce')
                df_upload[col].fillna(0, inplace=True)

            if MODEL is None:
                st.error("Model not available.")
            else:
                try:
                    preds = MODEL.predict(df_upload)
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                else:
                    label_map = ["Anger", "Anxiety", "Boredom", "Happiness", "Neutral", "Sadness"]
                    try:
                        pred_labels = [label_map[int(p)] if (isinstance(p, (int, float)) or str(p).isdigit()) else str(p) for p in preds]
                    except Exception:
                        pred_labels = [str(p) for p in preds]

                    out = df_upload.copy()
                    out["predicted_emotion"] = pred_labels
                    st.success(f"Predicted {len(out)} rows successfully")
                    st.dataframe(out.head())
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")


st.markdown("---")

