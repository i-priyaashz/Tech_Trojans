import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load sales data from a CSV file
# Ensure the CSV has 'Date' and 'Sales' columns
file_path = '/content/products.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)


# Display the first few rows of the dataset
print(data.head())

# Plot historical sales data
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['marked_price'], label='Historical Sales Data')
plt.title('Historical Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['marked_price'], model='additive', period=365)  # Adjust period as needed (365 for yearly)
trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal
residual = decomposition.resid.dropna()

# Plot decomposed components
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(trend, label='Trend Component', color='blue')
plt.title('Trend Component')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.subplot(3, 1, 2)
plt.plot(seasonal, label='Seasonal Component', color='orange')
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.subplot(3, 1, 3)
plt.plot(residual, label='Residual Component', color='green')
plt.title('Residual Component')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# Save the trend component data to a new CSV file
trend_df = pd.DataFrame({'Date': trend.index, 'Trend': trend.values})
trend_df.to_csv('trend_component.csv', index=False)

# Print the trend component data
print(trend_df.head())
