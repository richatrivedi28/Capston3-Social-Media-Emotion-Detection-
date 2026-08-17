import glob
import os
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Emotion Predictor",
    page_icon="🙂",
    layout="centered",
)

st.title("🌟📱 Social Media — Dominant Emotion Predictor")
st.info(
    "1.Use the sidebar for a single row prediction 2. upload a CSV for batch predictions(Multiple Rows)."
)

# These are the features expected by the saved training pipelines.
FEATURE_COLUMNS = [
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day",
    "Age",
    "Gender",
    "Platform",
]

NUMERIC_COLUMNS = [
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day",
    "Age",
]

# The training notebook used LabelEncoder for these six emotion classes.
EMOTION_LABELS = [
    "Anger",
    "Anxiety",
    "Boredom",
    "Happiness",
    "Neutral",
    "Sadness",
]

MODEL_FILES = [
    "optimized_random_forest_model.pkl",
    "optimized_xgboost_model.pkl",
]


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """Load one saved sklearn/XGBoost pipeline."""
    try:
        with open(model_path, "rb") as fh:
            return pickle.load(fh), None
    except Exception as exc:
        return None, (
            f"Could not load `{os.path.basename(model_path)}`: "
            f"{type(exc).__name__}: {exc}"
        )


def find_models():
    """Return known model files that exist in the app directory."""
    return [path for path in MODEL_FILES if os.path.isfile(path)]


def prediction_to_label(prediction):
    """Convert the model's numeric prediction to the trained emotion label."""
    try:
        value = int(prediction)
        if 0 <= value < len(EMOTION_LABELS):
            return EMOTION_LABELS[value]
    except (TypeError, ValueError):
        pass
    return str(prediction)


def build_input_df(
    age,
    gender,
    platform,
    daily_minutes,
    posts_per_day,
    likes_per_day,
    comments_per_day,
    messages_per_day,
):
    """Build a single-row DataFrame with the exact training feature names."""
    return pd.DataFrame(
        {
            "Daily_Usage_Time (minutes)": [daily_minutes],
            "Posts_Per_Day": [posts_per_day],
            "Likes_Received_Per_Day": [likes_per_day],
            "Comments_Received_Per_Day": [comments_per_day],
            "Messages_Sent_Per_Day": [messages_per_day],
            "Age": [age],
            "Gender": [gender],
            "Platform": [platform],
        }
    )[FEATURE_COLUMNS]


def validate_and_prepare_batch(df):
    """Validate and prepare an uploaded CSV for prediction."""
    df = df.copy()

    # Target is optional for evaluation/test CSVs; never send it to the model.
    if "Dominant_Emotion" in df.columns:
        df = df.drop(columns=["Dominant_Emotion"])

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"

    # Keep only model features, in the exact training order.
    df = df[FEATURE_COLUMNS].copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve the app's original behavior for invalid numeric values,
    # but avoid chained-assignment warnings.
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0)

    return df, None


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
available_models = find_models()

if not available_models:
    st.error(
        "No model files were found. Upload "
        "`optimized_random_forest_model.pkl` and/or "
        "`optimized_xgboost_model.pkl` with this app."
    )
    MODEL = None
    model_file = None
else:
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
    )
    MODEL, load_error = load_model(selected_model)
    model_file = selected_model

    if MODEL is None:
        st.sidebar.error(load_error)
        st.error(
            "The selected model could not be loaded. "
            "Check that the model was serialized with a compatible "
            "scikit-learn/XGBoost environment."
        )
    else:
        if "random_forest" in model_file.lower():
            model_name = "Random Forest"
        elif "xgboost" in model_file.lower():
            model_name = "XGBoost"
        else:
            model_name = os.path.splitext(model_file)[0]

        st.sidebar.success(f"Loaded model: {model_name}")


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------
with st.sidebar.form("input_form"):
    st.header("Single Prediction")

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=30,
        step=1,
    )
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Non-binary", "Unknown"],
    )
    platform = st.selectbox(
        "Platform",
        [
            "Instagram",
            "Twitter",
            "Facebook",
            "LinkedIn",
            "Snapchat",
            "Whatsapp",
            "Telegram",
        ],
    )
    daily_minutes = st.number_input(
        "Daily Usage Time (minutes)",
        min_value=0,
        max_value=1440,
        value=90,
        step=1,
    )
    posts_per_day = st.number_input(
        "Posts Per Day",
        min_value=0,
        max_value=100,
        value=2,
        step=1,
    )
    likes_per_day = st.number_input(
        "Likes Received Per Day",
        min_value=0,
        max_value=10000,
        value=40,
        step=1,
    )
    comments_per_day = st.number_input(
        "Comments Received Per Day",
        min_value=0,
        max_value=1000,
        value=15,
        step=1,
    )
    messages_per_day = st.number_input(
        "Messages Sent Per Day",
        min_value=0,
        max_value=1000,
        value=20,
        step=1,
    )

    submit_single = st.form_submit_button("🔮 Predict")


if submit_single:
    if MODEL is None:
        st.error("Model not available.")
    else:
        df_single = build_input_df(
            age,
            gender,
            platform,
            daily_minutes,
            posts_per_day,
            likes_per_day,
            comments_per_day,
            messages_per_day,
        )

        try:
            pred = MODEL.predict(df_single)
            pred_label = prediction_to_label(pred[0])

            st.success(f"Predicted dominant emotion: **{pred_label}**")

            if hasattr(MODEL, "predict_proba"):
                try:
                    probs = MODEL.predict_proba(df_single)[0]

                    # Use model classes when available; otherwise use the
                    # training notebook's six-class order.
                    classes = getattr(MODEL, "classes_", None)
                    if classes is not None:
                        emotions = [
                            prediction_to_label(cls) for cls in classes
                        ]
                    else:
                        emotions = EMOTION_LABELS[: len(probs)]

                    probs_df = pd.DataFrame(
                        {
                            "emotion": emotions,
                            "probability": probs,
                        }
                    )
                    probs_df["probability"] = probs_df["probability"].round(4)
                    st.dataframe(
                        probs_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.caption(f"Probability display unavailable: {exc}")

        except Exception as exc:
            st.error(
                f"Prediction failed: {type(exc).__name__}: {exc}"
            )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
st.header("Batch Predict from CSV")
st.caption(
    "CSV must contain the eight feature columns used during training. "
    "`Dominant_Emotion` may also be present and will be ignored."
)

uploaded_file = st.file_uploader(
    "Upload CSV (header required)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(
            uploaded_file,
            engine="python",
            on_bad_lines="warn",
        )
    except Exception as exc:
        st.error(f"Failed to read CSV: {type(exc).__name__}: {exc}")
        raw_df = None

    if raw_df is not None:
        st.write("Preview:")
        st.dataframe(raw_df.head(), use_container_width=True)

        df_upload, validation_error = validate_and_prepare_batch(raw_df)

        if validation_error:
            st.error(validation_error)
            st.info(
                "Required columns: "
                + ", ".join(FEATURE_COLUMNS)
            )
        elif df_upload.empty:
            st.warning("The uploaded CSV contains no data rows.")
        elif MODEL is None:
            st.error("Model not available.")
        else:
            try:
                preds = MODEL.predict(df_upload)
                pred_labels = [prediction_to_label(p) for p in preds]

                out = df_upload.copy()
                out["predicted_emotion"] = pred_labels

                st.success(
                    f"Predicted {len(out)} rows successfully."
                )
                st.dataframe(
                    out.head(100),
                    use_container_width=True,
                )

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(
                    f"Batch prediction failed: "
                    f"{type(exc).__name__}: {exc}"
                )


st.markdown("---")
st.caption(
    "Social Media — Dominant Emotion Predictor | "
    "Model inference application"
)
