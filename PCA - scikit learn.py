import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Wir verwenden denselben Beispieldatensatz wie oben
np.random.seed(42)
X_random = np.random.rand(150, 2)
transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
X = X_random @ transformation_matrix
X = X + np.random.normal(0, 0.1, X.shape)

# 1. Daten standardisieren
# Scikit-learn hat dafür eine eigene Klasse
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA anwenden
# Erstelle eine PCA-Instanz und gib die Anzahl der gewünschten Komponenten an
n_components = 2
pca = PCA(n_components=n_components)

# Führe die PCA durch und transformiere die Daten in einem Schritt
X_pca_sklearn = pca.fit_transform(X_scaled)

# Die Ergebnisse sind im PCA-Objekt gespeichert
print("--- PCA mit Scikit-learn ---")
print("Form der transformierten Daten:", X_pca_sklearn.shape)

# pca.components_ enthält die Eigenvektoren (Hauptkomponenten)
# ACHTUNG: Die Form ist (n_components, n_features), also transponiert im Vergleich zu oben
print("\nHauptkomponenten (Eigenvektoren):\n", pca.components_.T) 

# pca.explained_variance_ratio_ enthält den Anteil der erklärten Varianz
print("\nErklärte Varianz pro Komponente:", pca.explained_variance_ratio_)
print("Kumulative erklärte Varianz:", np.sum(pca.explained_variance_ratio_))


# --- Visualisierung (ähnlich wie oben) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Originaldaten
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.7)
ax1.set_title("Standardisierte Originaldaten")
ax1.set_xlabel("Feature 1 (standardisiert)")
ax1.set_ylabel("Feature 2 (standardisiert)")
ax1.axis('equal')

# Transformierte Daten
ax2.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], alpha=0.7, c='b')
ax2.set_title("Daten nach Scikit-learn PCA")
ax2.set_xlabel("Hauptkomponente 1")
ax2.set_ylabel("Hauptkomponente 2")
ax2.axis('equal')

plt.suptitle("PCA mit Scikit-learn", fontsize=16)
plt.show()