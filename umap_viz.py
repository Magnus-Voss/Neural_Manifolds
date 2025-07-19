import matplotlib.pyplot as plt
import numpy as np


def plot_high_dimensional_prob(sigma=1.0, x_j = np.array([-2.5, -0.5, 1.5, 2.8])):
    x_i = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))

    x_range = np.linspace(-5, 5, 500)

    rho_i = np.min([np.abs(x_i - x) for x in x_j])
    gaussian_pdf = np.exp(np.divide(-(np.abs(x_range - x_i) - rho_i), sigma))

    ax.plot(x_range, gaussian_pdf, color='royalblue', lw=2, label=f'RBF (σ = {sigma:.2f})')

    p_values = np.exp(np.divide(-(np.abs(x_j - x_i) - rho_i), sigma))

    ax.scatter(x_i, 0, color='red', s=150, zorder=5, label='$x_i$ (center)')
    ax.text(x_i, -0.15, '$x_i$', fontsize=14, ha='center')

    ax.scatter(x_j, np.zeros_like(x_j), color='green', s=100, zorder=5, label='$x_j$ (neighbours)')
    for k, point in enumerate(x_j):
        ax.text(point, -0.15, f'$x_{{j_{k+1}}}$', fontsize=14, ha='center')

    ax.vlines(x_j, 0, p_values, color='gray', linestyle='--', lw=1.5)
    ax.hlines(p_values, x_j, x_j, color='gray', lw=3)

    ax.set_title('High-Dimensional Probability $p_{j|i}$', fontsize=16, pad=20)
    ax.set_xlabel('Distance to $x_i$', fontsize=12)
    ax.set_ylabel('Similarity', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.25, 2)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    plt.savefig(f"./umap.pdf")

if __name__ == "__main__":
    plot_high_dimensional_prob()
