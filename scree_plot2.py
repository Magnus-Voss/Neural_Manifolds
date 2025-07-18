import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data (e.g., 100 samples, 10 features)
np.random.seed(42)
X = np.random.rand(100, 10)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Get the explained variance (eigenvalues)
explained_variance_ratio = pca.explained_variance_ratio_

# --- Create the Scree Plot ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(8, 6))

# Plot the explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-', linewidth=2, color='blue', label='Individual explained variance')

# Plot the cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's--', linewidth=2, color='red', label='Cumulative explained variance')

# Add labels and title
plt.title('Scree Plot', fontsize=16)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Proportion of Variance Explained', fontsize=12)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc='best')
plt.grid(True)

# Add a horizontal line at 95% cumulative variance
plt.axhline(y=0.95, color='green', linestyle=':', linewidth=1.5, label='95% threshold')
plt.legend(loc='best')


plt.show()