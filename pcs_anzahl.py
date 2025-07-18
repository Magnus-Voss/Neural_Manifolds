import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Daten erstellen ---
# Erstelle Beispieldaten (z.B. 100 Proben, 10 Features)
# In einer echten Anwendung würdest du hier deine eigenen Daten laden
X = np.random.rand(100, 10)

# --- 2. Daten standardisieren ---
# Es ist entscheidend, die Daten vor der PCA zu skalieren
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# --- 3. PCA durchführen ---
# Initialisiere PCA ohne die Anzahl der Komponenten vorab festzulegen
pca = PCA()
X_pca = pca.fit_transform(X)

# --- 4. Erklärte Varianz extrahieren ---
# Die 'explained_variance_' gibt die Varianz jedes PCs zurück
explained_variance = pca.explained_variance_
print(f"Varianzen der einzelnen PCs: \n{explained_variance}\n")

# --- 5. Auswahlkriterium anwenden ---
# Berechne die durchschnittliche Varianz
average_variance = np.mean(explained_variance)
print(f"Durchschnittliche Varianz aller PCs: {average_variance:.4f}\n")

# Finde heraus, wie viele PCs eine Varianz größer als der Durchschnitt haben
n_components_to_keep = np.sum(explained_variance > average_variance)
print(f"Anzahl der zu behaltenden PCs: {n_components_to_keep}")
print("Diese PCs haben eine überdurchschnittliche Varianz und werden beibehalten.")

# --- 6. Visualisierung ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Balkendiagramm der Varianzen
bar_positions = np.arange(len(explained_variance))
ax.bar(bar_positions, explained_variance, alpha=0.7, color='skyblue', label='Varianz der PCs')

# Horizontale Linie für die durchschnittliche Varianz
ax.axhline(y=average_variance, color='r', linestyle='--', label=f'average variance ({average_variance*100:.2f}%)')

# Punkt, der die "Cutoff"-Stelle markiert
# Platziere ihn an der Position des letzten zu behaltenden PCs
cutoff_point_x = n_components_to_keep - 1
cutoff_point_y = explained_variance[cutoff_point_x]
ax.plot(cutoff_point_x, cutoff_point_y, 'o', markersize=12, color='darkorange', label=f'{n_components_to_keep} PCs retained')


# Beschriftungen und Titel
ax.set_xticks(bar_positions)
ax.set_xticklabels([f'PC{i+1}' for i in range(len(explained_variance))])
ax.set_xlabel('Principal Components (PCs)')
ax.set_ylabel('Explained Variance')
ax.set_title('Variance of the PCs')
ax.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\willy\Uni\seminar grafik\averageVarianceMethod.pdf", transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
       )
plt.show()