import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from scipy.stats import t

# --- Teil 1: Hochdimensionale Wahrscheinlichkeiten (Gauss-Verteilung) ---

def plot_high_dimensional_prob(sigma=1.0, x_j = np.array([-2.5, -0.5, 1.5, 2.8])):
    """
    Erstellt einen interaktiven Plot der bedingten Wahrscheinlichkeiten p_j|i
    basierend auf einer Gauss-Verteilung im hochdimensionalen Raum.
    Die Varianz (sigma) kann mit einem Schieberegler angepasst werden.
    """
    # Datenpunkte definieren
    x_i = 0.0
    
    # Erstelle die Figur und die Achse
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # x-Werte für die Gauss-Kurve
    x_range = np.linspace(-5, 5, 500)
    
    # Gauss-Verteilung (unnormalisiert) berechnen
    # Dies entspricht dem Zähler in der p_j|i Formel
    gaussian_pdf = np.exp(-(x_range - x_i)**2 / (2 * sigma**2))
    
    # Plot der Gauss-Verteilung
    ax.plot(x_range, gaussian_pdf, color='royalblue', lw=2, label=f'normal distribution (σ = {sigma:.2f})')
    
    # Wahrscheinlichkeiten für die Punkte x_j berechnen
    p_values = np.exp(-(x_j - x_i)**2 / (2 * sigma**2))
    
    # Punkte und Wahrscheinlichkeitslinien plotten
    # Punkt x_i
    ax.scatter(x_i, 0, color='red', s=150, zorder=5, label='$x_i$ (center)')
    ax.text(x_i, -0.15, '$x_i$', fontsize=14, ha='center')

    # Punkte x_j
    ax.scatter(x_j, np.zeros_like(x_j), color='green', s=100, zorder=5, label='$x_j$ (neighbours)')
    for k, point in enumerate(x_j):
        ax.text(point, -0.15, f'$x_{{j_{k+1}}}$', fontsize=14, ha='center')

    # Gestrichelte Linien von x_j zur Kurve
    ax.vlines(x_j, 0, p_values, color='gray', linestyle='--', lw=1.5)
    ax.hlines(p_values, x_j, x_j, color='gray', lw=3) # Markiert den Wert auf der Kurve
    
    # Titel und Formel
    ax.set_title('Unnormalized High-Dimensional Probability $p_{j|i}$', fontsize=16, pad=20)
    #formula = r'$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$'
    #ax.text(0.5, 1.1, formula, fontsize=18, ha='center', va='center', transform=ax.transAxes)
    
    # Layout und Labels
    ax.set_xlabel('Distance to $x_i$', fontsize=12)
    ax.set_ylabel('Similarity', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Achsen-Grenzen
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.25, 1.2)
    
    # y-Achse ausblenden für einen sauberen Look
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().set_ticks([0, 0.5, 1.0])
    
    # plt.show()
    # plt.savefig(r"C:\Users\willy\Uni\seminar grafik\autogeneration\t-sne_gaussian_variance"+str(sigma)+".pdf", transparent=None, dpi='figure', format=None,
    #     metadata=None, bbox_inches=None, pad_inches=0.1,
    #     facecolor='auto', edgecolor='auto', backend=None,
    #    )

    plt.savefig(f"C:\\Users\\willy\\Uni\\seminar grafik\\autogeneration\\t-sne_gaussian_variance{str(sigma)}.pdf", transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
       )


# --- Teil 2: Niedrigdimensionale Wahrscheinlichkeiten (Student-t-Verteilung) ---

def plot_low_dimensional_prob(y_j = np.array([-2.5, -0.8, 1.5, 2.8])):
    """
    Erstellt einen statischen Plot der Ähnlichkeiten q_ij
    basierend auf einer Student-t-Verteilung im niedrigdimensionalen Raum.
    """
    # Datenpunkte definieren (entsprechen den y_i und y_j im niedrigdim. Raum)
    y_i = 0.0
    # y_j = np.array([-2.5, -0.8, 1.5, 2.8])
    
    # Erstelle die Figur und die Achse
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # x-Werte für die t-Verteilung
    x_range = np.linspace(-5, 5, 500)
    
    # Student-t-Verteilung (unnormalisiert, df=1)
    # Dies entspricht dem Zähler in der q_ij Formel
    t_dist = 1 / (1 + (x_range - y_i)**2)
    
    # Plot der t-Verteilung
    ax.plot(x_range, t_dist, color='darkorange', lw=2, label="Student's t-distribution (df=1)")
    
    # Ähnlichkeiten für die Punkte y_j berechnen
    q_values = 1 / (1 + (y_j - y_i)**2)
    
    # Punkte und Wahrscheinlichkeitslinien plotten
    # Punkt y_i
    ax.scatter(y_i, 0, color='red', s=150, zorder=5, label='$y_i$ (center)')
    ax.text(y_i, -0.15, '$y_i$', fontsize=14, ha='center')

    # Punkte y_j
    ax.scatter(y_j, np.zeros_like(y_j), color='purple', s=100, zorder=5, label='$y_j$ (neighbours)')
    for k, point in enumerate(y_j):
        ax.text(point, -0.15, f'$y_{{j_{k+1}}}$', fontsize=14, ha='center')

    # Gestrichelte Linien von y_j zur Kurve
    ax.vlines(y_j, 0, q_values, color='gray', linestyle='--', lw=1.5)
    ax.hlines(q_values, y_j, y_j, color='gray', lw=3)
    
    # Titel und Formel
    ax.set_title('Unnormalized Low-Dimensional Probability $q_{ij}$', fontsize=16, pad=20)
    #formula = r'$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$'
    #ax.text(0.5, 1.1, formula, fontsize=18, ha='center', va='center', transform=ax.transAxes)
    
    # Layout und Labels
    ax.set_xlabel('Distance to $y_i$', fontsize=12)
    ax.set_ylabel('Similarity', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Achsen-Grenzen
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.25, 1.2)
    
    # y-Achse ausblenden
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().set_ticks([0, 0.5, 1.0])
    
    plt.show()
    # plt.savefig(r"C:\Users\willy\Uni\seminar grafik\t-sne_t-distribution_CrowdingProblem.pdf", transparent=None, dpi='figure', format=None,
    #     metadata=None, bbox_inches=None, pad_inches=0.1,
    #     facecolor='auto', edgecolor='auto', backend=None,
    #    )

# --- Interaktiven Plot erstellen und anzeigen ---
print("Visualisierung für hochdimensionale Daten (p_j|i):")
# interactive_plot = interactive(
#     plot_high_dimensional_prob, 
#     sigma=FloatSlider(value=1.0, min=0.1, max=3.0, step=0.05, description='Sigma (σ):')
# )
#display(interactive_plot)


# --- Statischen Plot für niedrigdimensionale Daten erstellen ---
#print("\n\nVisualisierung für niedrigdimensionale Daten (q_ij):")
#plot_high_dimensional_prob(1, np.array([2]))
#plot_low_dimensional_prob(np.array([2.8]))

for i in range(15):
    sigma = (i+5)/10
    plot_high_dimensional_prob(sigma)

