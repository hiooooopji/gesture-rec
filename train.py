import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load recorded gesture data
file_path = "gestures.csv"  # Ensure this file exists with recorded data
df = pd.read_csv(file_path, header=None)

# Check if data is empty
if df.empty:
    print("🚨 Error: No valid gesture data found. Ensure you have recorded gestures.")
    exit()

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply K-Means clustering
num_clusters = 3  # Adjust based on how many gestures you expect
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered data
df.to_csv("gestures_clustered.csv", index=False)
print("✅ Training complete. Gestures grouped into clusters and saved.")
