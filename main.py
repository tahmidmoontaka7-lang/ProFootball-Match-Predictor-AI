import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load dataset and feature engineering
df = pd.read_csv("results.csv")
df["goal_diff"] = df["home_score"] - df["away_score"]

conditions = [df["goal_diff"] > 0, df["goal_diff"] < 0, df["goal_diff"] == 0]
choices = [0, 1, 2]  # 0 = Home Win, 1 = Away Win, 2 = Draw
df["match_result"] = np.select(conditions, choices, default=2)

# 2. Encode categorical team names into numeric codes
df["home_team_code"] = df["home_team"].astype("category").cat.codes
df["away_team_code"] = df["away_team"].astype("category").cat.codes

# 3. Select features (X) and target (y)
X = df[["home_team_code", "away_team_code"]]
y = df["match_result"]

# 4. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest Classifier model
print("Training the machine learning model... Please wait...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model accura
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 7. Serialize and save the trained model and dictionary mappings
joblib.dump(model, "football_model.pkl")

home_mapping = dict(zip(df["home_team"], df["home_team_code"]))
away_mapping = dict(zip(df["away_team"], df["away_team_code"]))
joblib.dump({"home": home_mapping, "away": away_mapping}, "team_mappings.pkl")
print("Model and team mappings saved successfully!")

