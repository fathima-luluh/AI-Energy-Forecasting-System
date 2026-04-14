import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("data/energy.csv")

print("Columns:", data.columns)

data["Datetime"] = pd.to_datetime(data["Datetime"])
data.set_index("Datetime", inplace=True)

# =========================
# FEATURES
# =========================
data["hour"] = data.index.hour
data["day"] = data.index.dayofweek

X = data[["hour", "day"]]
y = data["Energy"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
predictions = model.predict(X_test)

# =========================
# EVALUATION
# =========================
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "energy_model.pkl")

# =========================
# VISUALIZATION
# =========================
plt.plot(y_test.values[:20], label="Actual")
plt.plot(predictions[:20], label="Predicted")
plt.title("Energy Forecasting")
plt.legend()
plt.show()