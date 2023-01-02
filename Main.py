import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv("weather.csv")

# Split the data into features (X) and labels (y)
X = data[["humidity", "wind_speed", "pressure"]]
y = data["temperature"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest regressor
model = RandomForestRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)
