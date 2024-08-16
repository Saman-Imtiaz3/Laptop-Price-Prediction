import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("C:/Users/Mrlaptop/Downloads/codexcue internship projects/Laptop Price Prediction/laptopPrice.csv")

print(df.info())
print(df.describe())

print(df.isnull().sum())

df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)
df['graphic_card_gb'] = df['graphic_card_gb'].str.replace(' GB', '').astype(int)

sns.scatterplot(x='ram_gb', y='Price', data=df)
plt.title('Laptop Price vs RAM Size')
plt.show()

df.fillna(df.mode().iloc[0], inplace=True)

label_encoder = LabelEncoder()

df['brand_encoded'] = label_encoder.fit_transform(df['brand'])
df['processor_brand_encoded'] = label_encoder.fit_transform(df['processor_brand'])
df['processor_name_encoded'] = label_encoder.fit_transform(df['processor_name'])
df['ram_type_encoded'] = label_encoder.fit_transform(df['ram_type'])
df['os_encoded'] = label_encoder.fit_transform(df['os'])

X = df[['brand_encoded', 'processor_brand_encoded', 'processor_name_encoded', 
        'ram_gb', 'ram_type_encoded', 'ssd', 'hdd', 'os_encoded', 'graphic_card_gb']]
y = df['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
random_forest_model = RandomForestRegressor()

linear_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression - MAE: {mae_linear}, RMSE: {rmse_linear}")

y_pred_rf = random_forest_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Laptop Prices (Random Forest)')
plt.show()
