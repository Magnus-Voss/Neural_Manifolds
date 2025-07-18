# Erforderliche Bibliotheken importieren
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
# Wichtig für die 3D-Darstellung
from mpl_toolkits.mplot3d import Axes3D

# 1. Erstellen eines künstlichen 3D-Datensatzes mit Clustern
n_samples = 200
n_features = 3
n_clusters = 3

X_3d, y = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    n_features=n_features,
    random_state=42 # Für reproduzierbare Ergebnisse
)

# 2. t-SNE-Modell initialisieren und anpassen
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_3d)

# 3. Ergebnisse visualisieren (3D und 2D nebeneinander)
# Erstelle eine Abbildung mit zwei Subplots (1 Zeile, 2 Spalten)
fig = plt.figure(figsize=(16, 8))

# --- Linker Plot: Originale 3D-Daten ---
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(
    X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', s=50, alpha=0.8
)
ax1.set_title('Originale 3D-Daten mit Clustern')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
ax1.grid(True)

# --- Rechter Plot: t-SNE 2D-Visualisierung ---
ax2 = fig.add_subplot(122)
scatter2 = ax2.scatter(
    X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.9
)
ax2.set_title('t-SNE Visualisierung (auf 2D reduziert)')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.grid(True)

# Eine gemeinsame Legende für beide Plots erstellen
legend_labels = [f'Cluster {i+1}' for i in range(n_clusters)]
fig.legend(handles=scatter2.legend_elements()[0],
           labels=legend_labels,
           title="Datencluster",
           loc='upper right',
           bbox_to_anchor=(0.98, 0.95))

# Sorgt für saubere Abstände zwischen den Plots
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Platz für die Legende lassen

# Zeige die Plots an
plt.show()