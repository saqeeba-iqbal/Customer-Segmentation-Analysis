# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore') 

# --- 1. Data Collection ---
# Load the dataset
try:
    df = pd.read_csv('archive/customer_segmentation_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'archive/customer_segmentation_data.csv' not found. Please ensure the file is in the correct path.")
    exit() # Exit if the file is not found

# --- 2. Data Exploration and Cleaning ---
print("\n--- Data Exploration ---")
print("Dataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum()) # Check for missing values

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Basic descriptive statistics for numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

# Check for unique values in categorical columns
print("\nUnique values in 'gender':", df['gender'].unique())
print("Unique values in 'preferred_category':", df['preferred_category'].unique())

# --- 3. Descriptive Statistics (Key Metrics) ---
print("\n--- Descriptive Statistics (Key Metrics) ---")
# Average purchase amount
avg_purchase_amount = df['last_purchase_amount'].mean()
print(f"Average Last Purchase Amount: ${avg_purchase_amount:.2f}")

# Average spending score
avg_spending_score = df['spending_score'].mean()
print(f"Average Spending Score: {avg_spending_score:.2f}")

# Average purchase frequency
avg_purchase_frequency = df['purchase_frequency'].mean()
print(f"Average Purchase Frequency (per month): {avg_purchase_frequency:.2f}")

# --- 4. Customer Segmentation (K-Means Clustering) ---
print("\n--- Customer Segmentation (K-Means Clustering) ---")

# Select features for clustering
# Using numerical features that are most relevant for segmentation
features = ['income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Determine optimal number of clusters using the Elbow Method
# We'll test a range of K values and plot the inertia
inertia = []
silhouette_scores = []
k_range = range(2, 11) # Test K from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    # Calculate silhouette score if k > 1
    # The condition `if k > 1` is always true here because k_range starts from 2.
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the Elbow Method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
# Corrected: Use the full k_range as the x-axis, as silhouette_scores has a value for each k in k_range
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Scores for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on the elbow plot, let's assume an optimal K (e.g., 3 or 4) for demonstration.
# You would typically choose K where the elbow is most pronounced.
# For this example, let's pick K=4 as a reasonable starting point.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nClustering completed with {optimal_k} clusters.")
print("\nDistribution of customers per cluster:")
print(df['cluster'].value_counts().sort_index())

# --- 5. Visualization ---
print("\n--- Visualization of Customer Segments ---")

# Scatter plot of Income vs Spending Score, colored by cluster
plt.figure(figsize=(10, 7))
sns.scatterplot(x='income', y='spending_score', hue='cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments: Income vs. Spending Score')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Pairplot to visualize relationships between features and clusters (can be slow for large datasets)
# sns.pairplot(df, vars=features, hue='cluster', palette='viridis', diag_kind='kde')
# plt.suptitle('Pairplot of Features by Cluster', y=1.02)
# plt.show()

# Bar plots for categorical features distribution within each cluster
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='cluster', hue='gender', data=df, palette='pastel')
plt.title('Gender Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Gender')

plt.subplot(1, 2, 2)
sns.countplot(x='cluster', hue='preferred_category', data=df, palette='tab10')
plt.title('Preferred Category Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Preferred Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Visualize average feature values for each cluster
cluster_means = df.groupby('cluster')[features].mean()
print("\nAverage Feature Values per Cluster:")
print(cluster_means)

cluster_means.plot(kind='bar', figsize=(12, 7), colormap='plasma')
plt.title('Average Feature Values per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
