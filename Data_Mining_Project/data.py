import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('nike_sales_data.csv')

# Feature engineering
data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])
data['Month'] = data['Invoice Date'].dt.month
data['Year'] = data['Invoice Date'].dt.year

# Prepare the data for the model
X = data[['Month', 'Year', 'Units Sold']]  # Add more features as needed
y = data['Operating Profit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the predictions for Power BI
data['Predicted Profit'] = model.predict(X)
data.to_csv('predicted_sales.csv', index=False)
