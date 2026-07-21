import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create dataset
data = {
    "Hours Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam Score": [35, 45, 48, 55, 68, 75, 82, 88, 95, 98]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Hours Studied']]
y = df['Exam Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))

# Compare actual vs predicted values
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nComparison:")
print(results)

# Predict score for a student who studied 4.5 hours
new_hours = [[4.5]]
predicted_score = model.predict(new_hours)

print("\nPredicted Score for 4.5 Hours:", round(predicted_score[0], 2))