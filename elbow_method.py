import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Generate Synthetic Data ---
# We create a dataset with a clear underlying structure of 3 main dimensions,
# plus some noise. This will make the "elbow" more visible.
np.random.seed(42)
# Create 3 distinct latent variables
latent_vars = np.random.rand(100, 3)
# Linearly combine them to create 10 features
# The first 3 features will be strongly correlated with the latent variables
data = np.dot(latent_vars, np.random.rand(3, 10))
# Add some random noise to the data
data += np.random.normal(0, 0.1, size=data.shape)

# --- 2. Standardize the Data ---
# It's crucial to standardize the data before performing PCA.
# PCA is sensitive to the scale of the features.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- 3. Perform PCA ---
# We initialize PCA. By not specifying n_components,
# it will compute all possible components (in this case, 10).
pca = PCA()
pca.fit(data_scaled)

# --- 4. Plot the Explained Variance (The Elbow Method) ---
# We will now plot the explained variance ratio for each principal component.
# The explained variance ratio tells us how much of the total variance
# in the dataset is captured by each component.

# Get the explained variance for each component
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2, color='blue')

# --- 5. Annotate the Plot ---
plt.title('Scree Plot for PCA (Elbow Method)', fontsize=16)
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight the "elbow"
# In this example, the drop-off in explained variance becomes less steep after the 3rd component.
# This suggests that 3 components are a good choice.
plt.annotate('The "Elbow"\nOptimal number of PCs is likely 3',
             xy=(3, explained_variance[2]),
             xytext=(4, explained_variance[2] + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))

plt.savefig(r"C:\Users\willy\Uni\seminar grafik\scree_plot.pdf", transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
       )

# --- 6. Cumulative Explained Variance (Optional but useful) ---
# It's also helpful to look at the cumulative variance.
# cumulative_variance = np.cumsum(explained_variance)

# plt.figure(figsize=(10, 7))
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2, color='green')
# plt.title('Cumulative Explained Variance', fontsize=16)
# plt.xlabel('Number of Principal Components', fontsize=12)
# plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
# plt.xticks(range(1, len(cumulative_variance) + 1))
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True, linestyle='--', alpha=0.6)

# # Add a horizontal line at 95% variance, a common threshold
# plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance Threshold')
# plt.legend(loc='lower right')

# plt.show()

# Print the explained variance for each component
print("Explained Variance Ratio per Component:")
for i, var in enumerate(explained_variance):
    print(f"  PC-{i+1}: {var:.4f} ({var*100:.2f}%)")

print("\nCumulative Explained Variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"  Up to PC-{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")