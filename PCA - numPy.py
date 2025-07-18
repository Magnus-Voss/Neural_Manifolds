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
    # Dies ist wichtig, damit Features mit großen Wertebereichen nicht dominieren.
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # 2. Kovarianzmatrix berechnen
    # Die Kovarianzmatrix beschreibt die Beziehung zwischen den Features.
    # Wir verwenden .T, da np.cov erwartet, dass die Variablen in den Zeilen stehen.
    cov_matrix = np.cov(X_standardized.T)
    
    # 3. Eigenwerte und Eigenvektoren der Kovarianzmatrix berechnen
    # Eigenvektoren sind die Richtungen der maximalen Varianz (die Hauptkomponenten).
    # Eigenwerte geben die Größe dieser Varianz an.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. Eigenvektoren nach absteigenden Eigenwerten sortieren
    # Wir wollen die Komponenten mit der größten Varianz zuerst.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 5. Die ersten 'n_components' Eigenvektoren auswählen
    # Dies ist der Schritt der Dimensionsreduktion.
    components = sorted_eigenvectors[:, :n_components]
    
    # 6. Daten auf die neue Basis (Hauptkomponenten) projizieren
    X_projected = np.dot(X_standardized, components)
    
    # 7. Erklärte Varianz berechnen
    total_variance = np.sum(eigenvalues)
    explained_variance = sorted_eigenvalues / total_variance
    
    return X_projected, components, explained_variance

# --- Beispielanwendung ---
def daten_in_hauptkompenenten():
    # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    # Erzeuge unkorrelierte 2D-Daten
    X_random = np.random.rand(150, 2)
    # Skaliere und rotiere die Daten, um eine Korrelation zu erzeugen
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape) # Füge etwas Rauschen hinzu


    # Führe PCA durch, um auf 2 Hauptkomponenten zu reduzieren
    # (In diesem 2D-Fall ist es keine Reduktion, sondern eine Transformation)
    n_components = 2
    X_pca, pca_components, explained_variance = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten:", X_pca.shape)
    print("\nHauptkomponenten (Eigenvektoren):\n", pca_components)
    print("\nErklärte Varianz pro Komponente:", explained_variance[:n_components])
    print("Kumulative erklärte Varianz:", np.sum(explained_variance[:n_components]))

    # --- Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 6))

    # 1. Originaldaten und Hauptkomponenten
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Originaldaten')
    
    # Zeichne die Hauptkomponenten als Vektoren vom Mittelwert aus
    mean = np.mean(X, axis=0)
    # Skaliere die Vektoren mit der Wurzel der Eigenwerte für bessere Sichtbarkeit
    scaled_eigenvalues = np.sqrt(np.linalg.eig(np.cov(((X - mean) / np.std(X, axis=0)).T))[0])
    
    ax1.quiver(mean[0], mean[1], pca_components[0, 0], pca_components[1, 0], 
               color='r', scale=scaled_eigenvalues[0]*0.5, angles='xy', scale_units='xy', 
               label=f'PC1 ({explained_variance[0]*100:.1f}%)')
    ax1.quiver(mean[0], mean[1], pca_components[0, 1], pca_components[1, 1],
               color='g', scale=scaled_eigenvalues[1]*0.5, angles='xy', scale_units='xy',
               label=f'PC2 ({explained_variance[1]*100:.1f}%)')
    
    ax1.set_title("Originaldaten und Hauptkomponenten")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.axis('equal')

    # 2. Transformierte Daten
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='b', label='Transformierte Daten')
    ax2.set_title("Daten nach PCA-Transformation")
    ax2.set_xlabel("Hauptkomponente 1")
    ax2.set_ylabel("Hauptkomponente 2")
    ax2.legend()
    ax2.axis('equal')

    plt.suptitle("PCA von Grund auf implementiert", fontsize=16)
    plt.show()

# --- Beispielanwendung ---
def reduktion_auf_1_neu():
    # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    # Erzeuge unkorrelierte 2D-Daten
    X_random = np.random.rand(150, 2)
    # Skaliere und rotiere die Daten, um eine Korrelation zu erzeugen
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape) # Füge etwas Rauschen hinzu


    # Führe PCA durch, um auf 2 Hauptkomponenten zu reduzieren
    # (In diesem 2D-Fall ist es keine Reduktion, sondern eine Transformation)
    n_components = 2
    X_pca, pca_components, explained_variance = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten:", X_pca.shape)
    print("\nHauptkomponenten (Eigenvektoren):\n", pca_components)
    print("\nErklärte Varianz pro Komponente:", explained_variance[:n_components])
    print("Kumulative erklärte Varianz:", np.sum(explained_variance[:n_components]))

    # --- Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 6))

    # 1. Originaldaten und Hauptkomponenten
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Originaldaten')
    
    # Zeichne die Hauptkomponenten als Vektoren vom Mittelwert aus
    mean = np.mean(X, axis=0)
    # Skaliere die Vektoren mit der Wurzel der Eigenwerte für bessere Sichtbarkeit
    scaled_eigenvalues = np.sqrt(np.linalg.eig(np.cov(((X - mean) / np.std(X, axis=0)).T))[0])
    
    ax1.quiver(mean[0], mean[1], pca_components[0, 0], pca_components[1, 0], 
               color='r', scale=scaled_eigenvalues[0]*0.5, angles='xy', scale_units='xy', 
               label=f'PC1 ({explained_variance[0]*100:.1f}%)')
    ax1.quiver(mean[0], mean[1], pca_components[0, 1], pca_components[1, 1],
               color='g', scale=scaled_eigenvalues[1]*0.5, angles='xy', scale_units='xy',
               label=f'PC2 ({explained_variance[1]*100:.1f}%)')
    
    ax1.set_title("Originaldaten und Hauptkomponenten")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.axis('equal')

    # 2. Transformierte Daten
    ax2 = fig.add_subplot(1, 2, 2)
    #ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='b', label='Transformierte Daten')
    ax2.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), alpha=0.7, c='b', label='Transformierte Daten')
    ax2.set_title("Daten nach PCA-Transformation")
    ax2.set_xlabel("Hauptkomponente 1")
    ax2.set_ylabel("Hauptkomponente 2")
    ax2.legend()
    ax2.axis('equal')

    plt.suptitle("PCA von Grund auf implementiert", fontsize=16)
    plt.show()



# --- Beispielanwendung für 1D-Reduktion ---
def reduktion_auf_1_alt(): 
    # Erstelle einen Beispieldatensatz mit einer klaren Korrelation
    np.random.seed(42)
    X_random = np.random.rand(150, 2)
    transformation_matrix = np.array([[1.5, 1.0], [0.5, 1.2]])
    X = X_random @ transformation_matrix
    X = X + np.random.normal(0, 0.1, X.shape)

    # ----------------------------------------------------
    # ÄNDERUNG 1: Reduziere auf nur EINE Dimension
    # ----------------------------------------------------
    n_components = 1
    X_pca, pca_components, explained_variance = pca(X, n_components)

    print("Form der Originaldaten:", X.shape)
    print("Form der transformierten Daten (1D):", X_pca.shape)
    print("\nErste Hauptkomponente (Eigenvektor):\n", pca_components)
    
    # Gib nur die Varianz der ersten Komponente aus
    print(f"\nErklärte Varianz durch die 1. Komponente: {explained_variance[0]*100:.2f}%")

    # --- Angepasste Visualisierung ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 6))

    # 1. Originaldaten und die EINE Hauptkomponente
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7, label='Originaldaten')
    
    mean = np.mean(X, axis=0)
    scaled_eigenvalues = np.sqrt(np.linalg.eig(np.cov(((X - mean) / np.std(X, axis=0)).T))[0])
    
    # Zeichne nur die erste Hauptkomponente
    ax1.quiver(mean[0], mean[1], pca_components[0, 0], pca_components[1, 0], 
               color='r', scale=scaled_eigenvalues[0]*0.5, angles='xy', scale_units='xy', 
               label=f'PC1 ({explained_variance[0]*100:.1f}%)')
    
    ax1.set_title("Originaldaten und 1. Hauptkomponente")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.axis('equal')

    # ----------------------------------------------------
    # ÄNDERUNG 2: Visualisiere die 1D-Daten
    # ----------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    # Da wir nur eine Dimension haben, setzen wir die y-Werte auf 0,
    # um die Punkte entlang einer Linie darzustellen.
    ax2.scatter(X_pca, np.zeros_like(X_pca), alpha=0.7, c='b', label='Transformierte Daten (1D)')
    ax2.set_title("Daten nach Projektion auf 1 Dimension")
    ax2.set_xlabel("Wert auf Hauptkomponente 1")
    ax2.set_ylabel("") # Die y-Achse hat keine Bedeutung mehr
    ax2.tick_params(axis='y', left=False, labelleft=False) # y-Achse ausblenden
    ax2.legend()
    ax2.spines['left'].set_visible(False) # Achsenlinie entfernen
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    plt.suptitle("PCA-Reduktion auf eine Dimension", fontsize=16)
    plt.show()

if __name__ == "__main__": 
    daten_in_hauptkompenenten()
    reduktion_auf_1_neu()