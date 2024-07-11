 # type: ignore

import numpy as np
import pandas as pd


# Adplatform data
adplatform = pd.read_csv('datasets/adplatform_data.csv')
adplatform = adplatform.drop(columns=['criteo_cost', 'criteo_clicks', 'criteo_impressions', 
                                      'tiktok_cost', 'tiktok_clicks','tiktok_impressions'],axis=1)
adplatform['date'] = pd.to_datetime(adplatform['date'])
adplatform['day'] = adplatform['date'].dt.day
adplatform['month'] = adplatform['date'].dt.month
adplatform['year'] = adplatform['date'].dt.year
adplatform = adplatform.drop(columns=['date'],axis=1)

# Create a list of sets for each traffic source and their associated columns
traffic_sources = [
    {'prefix': 'google', 'columns': ['google_cost', 'google_clicks', 'google_impressions']},
    {'prefix': 'meta', 'columns': ['meta_cost', 'meta_clicks', 'meta_impressions']},
    {'prefix': 'rtbhouse', 'columns': ['rtbhouse_cost', 'rtbhouse_clicks', 'rtbhouse_impressions']}
]

transformed_dfs = []

# Iterate through each traffic source
for source in traffic_sources:
    # Select columns related to the current source, and the date columns
    cols = source['columns'] + ['day', 'month', 'year']
    temp_df = adplatform[cols].copy()
    
    # Add a 'traffic_source' column filled with the current source prefix
    temp_df['traffic_source'] = source['prefix']
    
    # Rename the source-specific metric columns to generic metric names
    for col in source['columns']:
        new_col_name = col.split('_', 1)[1]  # Remove the prefix
        temp_df.rename(columns={col: new_col_name}, inplace=True)
    
    # Append the transformed DataFrame to the list
    transformed_dfs.append(temp_df)

# Concatenate all the transformed DataFrames
adplatform = pd.concat(transformed_dfs, ignore_index=True)

print(adplatform.head())

# Transactions
group_columns = ['day','month','year', 'traffic_source', 'traffic_medium', 'device_category']

def update_traffic_source(value):
    if 'facebook' in value or 'instagram' in value:
        return 'meta'
    elif 'email' in value or 'youtube' in value or 'dv360' in value:
        return 'google'
    else:
        return value
    
transactions = pd.read_csv('datasets/transactions.csv')   
transactions['traffic_source'] = transactions['traffic_source'].apply(update_traffic_source)

transactions['date'] = pd.to_datetime(transactions['date'])
transactions['day'] = transactions['date'].dt.day
transactions['month'] = transactions['date'].dt.month
transactions['year'] = transactions['date'].dt.year


agg_dict = {
    'unique_transactions': pd.NamedAgg(column='transaction_id', aggfunc='nunique'),  # Count unique transaction IDs
    'transaction_revenue': pd.NamedAgg(column='transaction_revenue', aggfunc='sum'),  # Sum up transaction revenue
    'transaction_total': pd.NamedAgg(column='transaction_total', aggfunc='sum'),  # Sum up transaction totals
    'active_users': pd.NamedAgg(column='user_crm_id', aggfunc='count')  # Count non-null 'user_crm_id'
}

# Group by the specified columns and apply the aggregation
transactions = transactions.groupby(group_columns).agg(**agg_dict).reset_index()

print(transactions.head())

# Sessions

sessions = pd.read_csv('datasets/sessions.csv')
sessions['traffic_source'] = sessions['traffic_source'].apply(update_traffic_source)

sessions = sessions.rename(columns={'day':'date'})
sessions['date'] = pd.to_datetime(sessions['date'])
sessions['day'] = sessions['date'].dt.day
sessions['month'] = sessions['date'].dt.month
sessions['year'] = sessions['date'].dt.year

sessions = sessions.groupby(group_columns).agg({'total_sessions':'sum'}).reset_index()

print(sessions.head())

# Combine sessions with transactions

sessions_transactions = sessions.merge(transactions, on=group_columns, how='left')

sessions_transactions = sessions_transactions.groupby(['day','month','year','traffic_source']).agg(
    {'total_sessions':'sum', 
     'unique_transactions':'sum',
     'transaction_revenue':'sum',
     'transaction_total':'sum',
     'active_users':'sum'}).reset_index()

print(sessions_transactions.head())

# Combine all

df = adplatform.merge(sessions_transactions, on=['day','month','year','traffic_source'], how='left')
# order columns nicely

df = df[['day', 'month', 'year', 'traffic_source', 'total_sessions', 'unique_transactions', 'transaction_revenue', 'transaction_total', 'active_users', 'cost', 'clicks', 'impressions']]

print(df)

df.to_csv('datasets/clean_data.csv', index=False)



