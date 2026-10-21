import pandas as pd

# Read final dataset
df = pd.read_csv('../dataset/insider_activity_with_anomalies.csv')

# Convert to JSON for frontend
df.to_json('../frontend/data.json', orient='records')

print("JSON created successfully")
