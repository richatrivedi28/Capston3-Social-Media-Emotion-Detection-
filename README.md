# Social Media — Dominant Emotion Predictor

Small Streamlit app to predict the dominant emotion from social-media usage features.

- Single-row prediction: use the sidebar inputs and click the 🔮 Predict button.
- Batch prediction: upload a CSV and download a CSV with a new `predicted_emotion` column.

## Requirements
- Python 3.10+ (project used Python 3.12)
- See `requirements.txt` and install into your environment:

```bash
pip install -r requirements.txt
```

## Model files
- Place one of the following `.pkl` model files in the project root:
	- `optimized_random_forest_model.pkl`
	- `optimized_xgboost_model.pkl`
	

## Running the app
- From your project folder (use your environment's python executable):

```bash
# example using the workspace conda env
"C:/Users/richa/My Learnings/Practice_work/MLdeployment/Capstone3/myenv/python.exe" -m streamlit run app.py --server.port 8503 --server.headless true
```

## CSV format (required columns)
- Daily_Usage_Time (minutes)
- Posts_Per_Day
- Likes_Received_Per_Day
- Comments_Received_Per_Day
- Messages_Sent_Per_Day
- Age
- Gender
- Platform

## Notes
- `app.py` only attempts to load `.pkl` files via `pickle`.
- The app uses `on_bad_lines='skip'` when reading CSVs; malformed rows will be skipped.
- If you want the app to run on another port, change the `--server.port` argument when launching Streamlit.

If you want, I can add a short example CSV and a requirements installation script.

# Social Media Emotion Predictor (Streamlit)

Simple Streamlit app that loads a trained classifier and predicts a user's dominant emotion based on social-media usage features.

Requirements
- Python 3.9+ (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app

```bash
streamlit run app.py
```

Notes
- The app expects one of the trained model files to exist in the working folder: `optimized_random_forest_model.pkl` or `optimized_xgboost_model.pkl`.
- If you don't have those files, run the notebook `Batch13-Social_Media_Emotion_Detection_Richa_Trivedi.ipynb` to train and save them.
