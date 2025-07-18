# Erforderliche Bibliotheken importieren
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# 1. Daten laden
# Der Iris-Datensatz ist ein klassischer Datensatz für Klassifizierungs- und Visualisierungsaufgaben.
# Er enthält 150 Beobachtungen von Iris-Blumen, jede mit 4 Merkmalen (Länge/Breite von Kelch- und Blütenblättern)
# und einer von drei Arten (Klassen).
iris = load_iris()
X = iris.data  # Die Merkmale (Datenpunkte)
y = iris.target  # Die dazugehörigen Labels (Klassen)
target_names = iris.target_names # Namen der Klassen

print(f"Dimension der Originaldaten: {X.shape}")

# 2. t-SNE-Modell initialisieren und anpassen
# n_components=2: Wir wollen die Daten auf 2 Dimensionen für eine 2D-Visualisierung reduzieren.
# perplexity: Steht im Zusammenhang mit der Anzahl der nächsten Nachbarn, die für jeden Punkt berücksichtigt werden.
#             Ein typischer Wert liegt zwischen 5 und 50.
# n_iter: Anzahl der Iterationen für die Optimierung.
# random_state: Sorgt für reproduzierbare Ergebnisse.
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)

# Führe die Dimensionsreduktion durch
X_tsne = tsne.fit_transform(X)

print(f"Dimension nach t-SNE: {X_tsne.shape}")

# 3. Ergebnisse visualisieren
# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)

# Füge eine Legende hinzu, um die Klassen zu identifizieren
legend1 = plt.legend(handles=scatter.legend_elements()[0],
                    labels=list(target_names),
                    title="Klassen")
plt.gca().add_artist(legend1)

# Titel und Achsenbeschriftungen hinzufügen
plt.title('t-SNE Visualisierung des Iris-Datensatzes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True) # Fügt ein Gitter für bessere Lesbarkeit hinzu

# Zeige den Plot an
plt.show()