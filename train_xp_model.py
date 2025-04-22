import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load cleaned data
df = pd.read_csv("data/clean_xp_data.csv")
X = df[["action_id", "xp_type_id"]]
y = df["xp_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Validation RMSE: {rmse:.2f}")

# Save model
import joblib
joblib.dump(model, "models/xp_predictor.pkl")
print("Saved model to models/xp_predictor.pkl")
