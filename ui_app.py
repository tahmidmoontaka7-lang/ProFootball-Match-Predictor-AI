import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="ProFootball AI Predictor", page_icon="⚽", layout="centered")

st.title("⚽ ProFootball Match Predictor AI")
st.write("Select two international teams to predict the match outcome probabilities.")
st.markdown("---")


# 2. Optimized On-the-Fly Training Cache Function
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("results.csv")
    df["goal_diff"] = df["home_score"] - df["away_score"]

    conditions = [df["goal_diff"] > 0, df["goal_diff"] < 0, df["goal_diff"] == 0]
    choices = [0, 1, 2]  # 0 = Home Win, 1 = Away Win, 2 = Draw
    df["match_result"] = np.select(conditions, choices, default=2)

    # Encode labels cleanly
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    df["home_team_code"] = df["home_team"].astype("category").cat.codes
    df["away_team_code"] = df["away_team"].astype("category").cat.codes

    # Create mapping dictionaries
    home_mapping = dict(zip(df["home_team"], df["home_team_code"]))
    away_mapping = dict(zip(df["away_team"], df["away_team_code"]))

    # Feature selection
    X = df[["home_team_code", "away_team_code"]]
    y = df["match_result"]

    # Quick train with optimized settings for deployment
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    return clf, home_mapping, away_mapping


# Initialize backend values cleanly
with st.spinner("Initializing AI engine... Please wait a few seconds..."):
    model, home_map, away_map = load_and_train()

# Get unique sorted teams list
teams = sorted(list(home_map.keys()))

# 3. Dropdown UI components
team_a = st.selectbox("Select Home Team (Team A):", teams,
                      index=teams.index("Bangladesh") if "Bangladesh" in teams else 0)
team_b = st.selectbox("Select Away Team (Team B):", teams, index=teams.index("Malaysia") if "Malaysia" in teams else 1)

st.markdown("---")

# 4. Probabilistic Engine Operations
if st.button("Predict Match Outcome"):
    if team_a == team_b:
        st.warning("Please select two different teams!")
    else:
        code_a = home_map[team_a]
        code_b = away_map[team_b]

        input_data = np.array([[code_a, code_b]])
        probabilities = model.predict_proba(input_data)[0]

        st.subheader(f"📊 Prediction Analytics: {team_a} vs {team_b}")

        # Display smooth probability trackers
        st.write(f"🏠 **{team_a} Win Probability:** {probabilities[0] * 100:.1f}%")
        st.progress(float(probabilities[0]))

        st.write(f"🚀 **{team_b} Win Probability:** {probabilities[1] * 100:.1f}%")
        st.progress(float(probabilities[1]))

        st.write(f"🤝 **Draw Probability:** {probabilities[2] * 100:.1f}%")
        st.progress(float(probabilities[2]))

