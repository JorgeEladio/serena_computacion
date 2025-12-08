import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyección 3D


"""
soliton_ansatz_compacto_esmeralda.py

Visualiza la configuración tipo Hopf (ansatz compacto) para la teoría Serena.
Genera:
  - Nube 3D en torno al toro (densidad de energía + color en nz)
  - Corte 2D en el plano y = 0 con densidad de energía y campo tangencial.
"""


def generar_ansatz_hopf_compacto(X, Y, Z, R_scale=2.5):
    """
    Genera el campo n(x) a partir del ansatz Hopf compacto:
        R^3 -> S^3 -> S^2

    Parámetros
    ----------
    X, Y, Z : ndarray
        Mallas 3D de coordenadas.
    R_scale : float
        Escala del núcleo del solitón.

    Devuelve
    --------
    nx, ny, nz : ndarray
        Componentes del campo director n(x) en S^2.
    """
    R_sq = X**2 + Y**2 + Z**2
    denom = R_sq + R_scale**2

    # Proyección estereográfica inversa R^3 -> S^3
    x1 = 2 * R_scale * X / denom
    x2 = 2 * R_scale * Y / denom
    x3 = 2 * R_scale * Z / denom
    x4 = (R_sq - R_scale**2) / denom

    # Espinores complejos
    Z0 = x1 + 1j * x2
    Z1 = x3 + 1j * x4

    # Mapa de Hopf S^3 -> S^2
    n_complex = 2 * np.conj(Z0) * Z1
    nx = np.real(n_complex)
    ny = np.imag(n_complex)
    nz = np.abs(Z1) ** 2 - np.abs(Z0) ** 2

    return nx, ny, nz


def main():
    print("Visualización de la configuración tipo Hopf (ansatz compacto)...")

    # 1. Dominio y resolución para visualización
    L = 6.0
    N = 60
    x = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # 2. Ansatz Hopf compacto
    nx, ny, nz = generar_ansatz_hopf_compacto(X, Y, Z, R_scale=2.5)

    # 3. Densidad de energía aproximada: E = 1/2 Σ (∂i n)^2
    dnx = np.gradient(nx)
    dny = np.gradient(ny)
    dnz = np.gradient(nz)

    Energy = (dnx[0]**2 + dnx[1]**2 + dnx[2]**2 +
              dny[0]**2 + dny[1]**2 + dny[2]**2 +
              dnz[0]**2 + dnz[1]**2 + dnz[2]**2)

    E_norm = Energy / np.max(Energy)

    # 4. Visualización
    fig = plt.figure(figsize=(14, 6))

    # --- Panel 1: geometría 3D del solitón ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    mask = E_norm > 0.2
    size = E_norm[mask] * 30.0

    ax1.scatter(
        X[mask], Y[mask], Z[mask],
        c=nz[mask],
        cmap='twilight',
        s=size,
        alpha=0.6,
        edgecolors='none'
    )

    ax1.set_title("Geometría 3D del solitón tipo Hopf (ansatz compacto)", fontsize=12)
    ax1.set_xlim(-L, L)
    ax1.set_ylim(-L, L)
    ax1.set_zlim(-L, L)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # --- Panel 2: Corte en el plano y = 0 ---
    ax2 = fig.add_subplot(1, 2, 2)

    mid = N // 2
    X_slice = X[:, mid, :]
    Z_slice = Z[:, mid, :]
    E_slice = E_norm[:, mid, :]
    nx_slice = nx[:, mid, :]
    nz_slice = nz[:, mid, :]

    contour = ax2.contourf(X_slice, Z_slice, E_slice, levels=40, cmap='inferno')
    plt.colorbar(contour, ax=ax2, label='Densidad de energía')

    step = 3
    ax2.quiver(
        X_slice[::step, ::step],
        Z_slice[::step, ::step],
        nx_slice[::step, ::step],
        nz_slice[::step, ::step],
        color='cyan',
        scale=20,
        width=0.004,
        alpha=0.8
    )

    ax2.set_title("Corte en el plano y = 0: energía y campo tangencial", fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
