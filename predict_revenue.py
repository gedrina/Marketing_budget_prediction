 # type: ignore

import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.impute import SimpleImputer


# Load the dataset
data = pd.read_csv('datasets/clean_data.csv')

# Preprocess: Drop rows with missing target values and separate features and target
data.dropna(subset=['transaction_revenue'], inplace=True)
features = data.drop(columns=['transaction_revenue', 'total_sessions', 'unique_transactions',
                              'clicks', 'impressions', 'active_users', 'transaction_total'])
target = data['transaction_total']

# create the future data we would like to predict
# max_year = features['year'].max()
# features = features[features['year'] == max_year]
# max_month = features['month'].max()
# features = features[features['month'] == max_month]
# max_day = features['day'].max()
# features = features[features['day'] == max_day]
# last date is 31/12/2021

# Create a range of dates from 01/01/2022 to 31/12/2022
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Create empty lists to store the data
days = []
months = []
years = []
traffic_sources = []
costs = []

# Loop through the dates, traffic_source and cost, create all possible permutations
for date in dates:
    for traffic_source in ['google', 'meta', 'rtbhouse']:
        for cost in [100, 250, 500, 1000]:
            days.append(date.day)
            months.append(date.month)
            years.append(date.year)
            traffic_sources.append(traffic_source)
            costs.append(cost)

# Create the pandas_deploy DataFrame
pandas_deploy = pd.DataFrame({
    'day': days,
    'month': months,
    'year': years,
    'traffic_source': traffic_sources,
    'cost': costs
})

# Encode categorical data and fill missing values with median for simplicity
features = pd.get_dummies(features)
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)

# Encode categorical data for pandas deploy
features_deploy = pd.get_dummies(pandas_deploy)
X_deploy = imputer.fit_transform(features_deploy)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# Dictionary of models - search over models
# models = {
#     "Linear Regression": LinearRegression(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Random Forest": RandomForestRegressor(n_estimators=10),  # Reduced for quicker execution
#     "Gradient Boosting": GradientBoostingRegressor()
# }
# Random forest reaches the best results

# search over number of trees
# models = {
#     "Random Forest 10": RandomForestRegressor(n_estimators=10),
#     "Random Forest 50": RandomForestRegressor(n_estimators=50),
#     "Random Forest 100": RandomForestRegressor(n_estimators=100),
#     "Random Forest 200": RandomForestRegressor(n_estimators=200)
# }
# We get the best results with 100 trees

# search over max_depth
# models = {
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=4),
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8),
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16),
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=32),
#     "Random Forest": RandomForestRegressor(n_estimators=100)
# }
# best results with 16 trees

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16),
}

# Loop through models, train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predict on test set
    r2 = r2_score(y_test, y_pred)  # Calculate R^2 score
    print(f"{name}: R^2 = {r2}")

# let's predict how much revenue we will get in the next year
y_deploy_predict = model.predict(X_deploy)

# now let's check how much come from google, meta and rtbhouse
google_100, meta_100, rtbhouse_100 = 0, 0, 0
google_250, meta_250, rtbhouse_250 = 0, 0, 0
google_500, meta_500, rtbhouse_500 = 0, 0, 0
google_1000, meta_1000, rtbhouse_1000 = 0, 0, 0

for i in range(0, y_deploy_predict.shape[0] - 12, 12):
    # when investing $100 in marketing
    google_100 += y_deploy_predict[i]
    google_250 += y_deploy_predict[i+1]
    google_500 += y_deploy_predict[i+2]
    google_1000 += y_deploy_predict[i+3]

    # when investing $500 in marketing
    meta_100 += y_deploy_predict[i+4]
    meta_250 += y_deploy_predict[i+5]
    meta_500 += y_deploy_predict[i+6]
    meta_1000 += y_deploy_predict[i+7]

    # when investing $1000 in marketing
    rtbhouse_100 += y_deploy_predict[i+8]
    rtbhouse_250 += y_deploy_predict[i+9]
    rtbhouse_500 += y_deploy_predict[i+10]
    rtbhouse_1000 += y_deploy_predict[i+11]

print("Investing $100 per day in Google, brings a revenue of: " + str(google_100))
print("Investing $100 per day in Meta, brings a revenue of: " + str(meta_100))
print("Investing $100 per day in RTBHouse, brings a revenue of: " + str(rtbhouse_100))
print("Investing $250 per day in Google, brings a revenue of: " + str(google_250))
print("Investing $250 per day in Meta, brings a revenue of: " + str(meta_250))
print("Investing $250 per day in RTBHouse, brings a revenue of: " + str(rtbhouse_250))
print("Investing $500 per day in Google, brings a revenue of: " + str(google_500))
print("Investing $500 per day in Meta, brings a revenue of: " + str(meta_500))
print("Investing $500 per day in RTBHouse, brings a revenue of: " + str(rtbhouse_500))
print("Investing $1000 per day in Google, brings a revenue of: " + str(google_1000))
print("Investing $1000 per day in Meta, brings a revenue of: " + str(meta_1000))
print("Investing $1000 per day in RTBHouse, brings a revenue of: " + str(rtbhouse_1000))
print('\n')

# total_revenue / total_invested
print("Investing $100 per day in Google, revenue per dollar spend in marketing: " + str(google_100 / (100 * 365)))
print("Investing $100 per day in Meta, revenue per dollar spend in marketing: " + str(meta_100 / (100 * 365)))
print("Investing $100 per day in RTBHouse, revenue per dollar spend in marketing: " + str(rtbhouse_100 / (100 * 365)))
print("Investing $250 per day in Google, revenue per dollar spend in marketing: " + str(google_250 / (250 * 365)))
print("Investing $250 per day in Meta, revenue per dollar spend in marketing: " + str(meta_250 / (250 * 365)))
print("Investing $250 per day in RTBHouse, revenue per dollar spend in marketing: " + str(rtbhouse_250 / (250 * 365)))
print("Investing $500 per day in Google, revenue per dollar spend in marketing: " + str(google_500 / (500 * 365)))
print("Investing $500 per day in Meta, brevenue per dollar spend in marketing: " + str(meta_500 / (500 * 365)))
print("Investing $500 per day in RTBHouse, revenue per dollar spend in marketing: " + str(rtbhouse_500 / (500 * 365)))
print("Investing $1000 per day in Google, revenue per dollar spend in marketing: " + str(google_1000 / (1000 * 365)))
print("Investing $1000 per day in Meta, revenue per dollar spend in marketing: " + str(meta_1000 / (1000 * 365)))
print("Investing $1000 per day in RTBHouse, revenue per dollar spend in marketing: " + str(rtbhouse_1000 / (1000 * 365)))