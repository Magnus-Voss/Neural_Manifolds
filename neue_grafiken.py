import numpy as np
import matplotlib.pyplot as plt

def pca(X, n_components):
    """
    Führt eine Hauptkomponentenanalyse (PCA) auf einem Datensatz durch.

    Args:
        X (np.ndarray): Der Eingabedatensatz, bei dem Zeilen Beobachtungen
                        und Spalten Features sind.
        n_components (int): Die Anzahl der zu behaltenden Hauptkomponenten.

    Returns:
        X_projected (np.ndarray): Der auf die Hauptkomponenten projizierte Datensatz.
        components (np.ndarray): Die Hauptkomponenten (Eigenvektoren).
        explained_variance (np.ndarray): Der von jeder Komponente erklärte Varianzanteil.
    """
    # 1. Daten standardisieren (Mittelwert 0, Standardabweichung 1)
    #X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_standardized = X - np.mean(X, axis=0)
    
    # 2. Kovarianzmatrix berechnen
    cov_matrix = np.cov(X_standardized.T)
    
    # 3. Eigenwerte und Eigenvektoren der Kovarianzmatrix berechnen
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. Eigenvektoren nach absteigenden Eigenwerten sortieren
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 5. Die ersten 'n_components' Eigenvektoren auswählen
    components = sorted_eigenvectors[:, :n_components]
    
    # 6. Daten auf die neue Basis (Hauptkomponenten) projizieren
    X_projected = np.dot(X_standardized, components)
    
    # 7. Erklärte Varianz berechnen
    total_variance = np.sum(eigenvalues)
    explained_variance = sorted_eigenvalues / total_variance
    
    return X_projected, components, explained_variance, X_standardized

# --- Beispielanwendung ---
def originalDataCentered():
    # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    X_random = np.random.rand(150, 2)
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape)

    # Führe PCA durch
    n_components = 2
    X_pca, pca_components, explained_variance, X_standardized = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten:", X_pca.shape)
    print("\nHauptkomponenten (Eigenvektoren):\n", pca_components)
    print("\nErklärte Varianz pro Komponente:", explained_variance[:n_components])
    print("Kumulative erklärte Varianz:", np.sum(explained_variance[:n_components]))

    # --- Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(6, 6))

    # 1. Originaldaten und Hauptkomponenten
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(X_standardized[:, 0], X_standardized[:, 1], alpha=0.7, label='Original Data')
    
    mean = np.mean(X, axis=0)
    
    # --- ANPASSUNG HIER ---
    # Zeichne die Hauptkomponenten als Vektoren mit Länge 1.
    # Die Eigenvektoren (pca_components) sind bereits Einheitsvektoren.
    # Mit scale=1 wird die Länge der Vektoren 1:1 in den Datenkoordinaten abgebildet.
   
    
    ax1.set_title("Original Data Centered")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.axis('equal')  

#------------------------------

    # 3. Transformierte Daten
#    ax3 = fig.add_subplot(1, 3, 3)
#    ax3.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), alpha=0.7, c='b', label='Transformierte Daten')    
    # --- ANPASSUNG HIER ---
    # Zeichne die neuen Achsen (Einheitsvektoren) vom Ursprung (0,0) aus.
#    ax3.quiver(0, 0, 1, 0, color='r', scale=1, angles='xy', scale_units='xy', label='PC1 Achse')    
#    ax3.set_title("Daten nach PCA-Transformation")
#    ax3.set_xlabel("Hauptkomponente 1")
#    ax3.set_ylabel("Hauptkomponente 2")
#    ax3.legend()
#    ax3.axis('equal')

    #plt.suptitle("PCA von Grund auf implementiert", fontsize=16)
    #plt.show()
    
    plt.savefig(r"C:\Users\willy\Uni\pca_graphic_OriginalDataCentered.pdf", transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
       )
def TransformedDataReduktion():
    # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    X_random = np.random.rand(150, 2)
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape)

    # Führe PCA durch
    n_components = 2
    X_pca, pca_components, explained_variance, X_standardized = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten:", X_pca.shape)
    print("\nHauptkomponenten (Eigenvektoren):\n", pca_components)
    print("\nErklärte Varianz pro Komponente:", explained_variance[:n_components])
    print("Kumulative erklärte Varianz:", np.sum(explained_variance[:n_components]))

    # --- Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(6, 6))
    
    mean = np.mean(X, axis=0)
    
    # --- ANPASSUNG HIER ---
    # Zeichne die Hauptkomponenten als Vektoren mit Länge 1.
    # Die Eigenvektoren (pca_components) sind bereits Einheitsvektoren.
    # Mit scale=1 wird die Länge der Vektoren 1:1 in den Datenkoordinaten abgebildet.
   
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), alpha=0.7, c='b', label='Transformed Data')    
    #--- ANPASSUNG HIER ---
    #Zeichne die neuen Achsen (Einheitsvektoren) vom Ursprung (0,0) aus.
    ax3.quiver(0, 0, 1, 0, color='r', scale=1, angles='xy', scale_units='xy', label=f'PC1 ({explained_variance[0]*100:.1f}%)')    
    ax3.set_title("Data After the PCA-Transformation and Dimensionality Reduction")
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    ax3.legend()
    ax3.axis('equal')

    plt.show()
    # plt.savefig(r"C:\Users\willy\Uni\pca_graphic_TransformedDataReduktion.pdf", transparent=None, dpi='figure', format=None,
    #     metadata=None, bbox_inches=None, pad_inches=0.1,
    #     facecolor='auto', edgecolor='auto', backend=None,
    #    )


def pca_graphic_TransformedDataWithPCs2():
     # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    X_random = np.random.rand(150, 2)
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape)

    # Führe PCA durch
    n_components = 2
    X_pca, pca_components, explained_variance, X_standardized = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten:", X_pca.shape)
    print("\nHauptkomponenten (Eigenvektoren):\n", pca_components)
    print("\nErklärte Varianz pro Komponente:", explained_variance[:n_components])
    print("Kumulative erklärte Varianz:", np.sum(explained_variance[:n_components]))

    # --- Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(6, 6))

    # 2. Transformierte Daten
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='b', label='Transformed Data')
    
    # --- ANPASSUNG HIER ---
    # Zeichne die neuen Achsen (Einheitsvektoren) vom Ursprung (0,0) aus.
    ax2.quiver(0, 0, 1, 0, color='r', scale=1, angles='xy', scale_units='xy', label=f'PC1 ({explained_variance[0]*100:.1f}%)')
    ax2.quiver(0, 0, 0, 1, color='g', scale=1, angles='xy', scale_units='xy', label=f'PC2 ({explained_variance[1]*100:.1f}%)')
    
    ax2.set_title("Data After the PCA-Transformation")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.legend()
    ax2.axis('equal')

    plt.show()
    # plt.savefig(r"C:\Users\willy\Uni\pca_graphic_TransformedDataWithPCs2.pdf", transparent=None, dpi='figure', format=None,
    #     metadata=None, bbox_inches=None, pad_inches=0.1,
    #     facecolor='auto', edgecolor='auto', backend=None,
    #    )

# Führe die Funktion aus
originalDataCentered()





    