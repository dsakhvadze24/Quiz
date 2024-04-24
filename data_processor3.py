import ipaddress
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

df = pd.read_csv("Darknet.csv")
df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)
df = df.dropna()
df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv")

# Replace NaN with 0 or another appropriate value
df[np.isnan(df)] = 0

# Clip the values to a specific range
min_value = df.values.min()  # Minimum finite value in the data
max_value = 1e+6  # Maximum value in the data
df_clipped = np.clip(df.values, min_value, max_value)

# Convert the clipped NumPy array back to a DataFrame
df_clipped = pd.DataFrame(df_clipped, columns=df.columns)

scaler = RobustScaler()
dfs = scaler.fit_transform(df_clipped)

# Convert the scaled NumPy array back to a DataFrame
dfs_df = pd.DataFrame(dfs, columns=df.columns)

# Save the scaled DataFrame to a CSV file
dfs_df.to_csv("scaled3.csv", index=False)