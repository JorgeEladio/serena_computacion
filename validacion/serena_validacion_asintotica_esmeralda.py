import numpy as np
import matplotlib.pyplot as plt


"""
serena_validacion_asintotica_esmeralda.py

Apéndice A: Validación asintótica de la configuración Esmeralda.
Incluye:
  - Ansatz compacto estereográfico (R^3 -> S^3 -> S^2)
  - Relajación difusiva corta (suavizado geométrico)
  - Derivadas de 4º orden para energía y campo efectivo
  - Ajuste log-log de la cola (exponentes ~4 y ~2)
"""


def laplacian_fast(f, dx):
    """Laplaciano rápido (stencil de 7 puntos)."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) +
            np.roll(f, 1, 2) + np.roll(f, -1, 2) - 6.0 * f) / dx**2


def d4(f, axis, dx):
    """Derivada centrada de 4º orden."""
    return (-np.roll(f, -2, axis) + 8*np.roll(f, -1, axis)
            - 8*np.roll(f, 1, axis) + np.roll(f, 2, axis)) / (12 * dx)


def main():
    print("Iniciando simulación Serena (Configuración Esmeralda)...")

    # 1. Dominio y discretización
    L = 40.0
    N = 140
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # 2. Ansatz compacto estereográfico
    R_scale = 2.5
    denom = R**2 + R_scale**2

    x1 = 2 * R_scale * X / denom
    x2 = 2 * R_scale * Y / denom
    x3 = 2 * R_scale * Z / denom
    x4 = (R**2 - R_scale**2) / denom

    Z0 = x1 + 1j * x2
    Z1 = x3 + 1j * x4

    n_complex = 2 * np.conj(Z0) * Z1
    nx = np.real(n_complex)
    ny = np.imag(n_complex)
    nz = np.abs(Z1)**2 - np.abs(Z0)**2

    # 3. Relajación difusiva breve
    print("Aplicando relajación difusiva para eliminar ruido...")

    mask = np.ones_like(nx)
    m = 2
    mask[:m, :, :] = 0
    mask[-m:, :, :] = 0
    mask[:, :m, :] = 0
    mask[:, -m:, :] = 0
    mask[:, :, :m] = 0
    mask[:, :, -m:] = 0

    dt = 0.005
    steps = 50

    for t in range(steps):
        nx += dt * laplacian_fast(nx, dx)
        ny += dt * laplacian_fast(ny, dx)
        nz += dt * laplacian_fast(nz, dx)

        norm = np.sqrt(nx**2 + ny**2 + nz**2 + 1e-12)
        nx /= norm
        ny /= norm
        nz /= norm

        nx *= mask
        ny *= mask
        nz = nz * mask + (1.0 - mask)

    # 4. Cálculo de energía y campo efectivo (derivadas de 4º orden)
    print("Calculando derivadas de alta precisión...")

    E = 0.0
    for comp in (nx, ny, nz):
        for ax in (0, 1, 2):
            E += 0.5 * d4(comp, ax, dx)**2

    F = np.sqrt(2.0 * E)

    # 5. Ajuste asintótico en ventana radial
    mask_fit = (R > 12.0) & (R < 32.0)

    R_fit = R[mask_fit]
    E_fit = E[mask_fit]
    F_fit = F[mask_fit]

    coeffs_E = np.polyfit(np.log(R_fit), np.log(E_fit), 1)
    n_E = -coeffs_E[0]
    A_E = np.exp(coeffs_E[1])

    coeffs_F = np.polyfit(np.log(R_fit), np.log(F_fit), 1)
    n_F = -coeffs_F[0]
    A_F = np.exp(coeffs_F[1])

    print("-" * 40)
    print("RESULTADOS DEL ANÁLISIS ASINTÓTICO:")
    print(f"Exponente Energía (Masa): {n_E:.4f}  (Teórico: 4.00)")
    print(f"Exponente Campo  (Fuerza): {n_F:.4f}  (Teórico: 2.00)")
    print("-" * 40)

    # 6. Gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    step_plot = 30

    # Energía
    ax1.loglog(R_fit[::step_plot], E_fit[::step_plot], 'k.', alpha=0.3)
    ax1.loglog(R_fit, A_E * R_fit**(-n_E), 'b-', lw=2,
               label=f'Sim: {n_E:.2f}')
    ax1.loglog(
        R_fit,
        E_fit[0] * (R_fit / R_fit[0])**(-4.0),
        'r--', label='Ref: 4.00'
    )
    ax1.set_title("Cola de Energía (Masa finita)")
    ax1.set_xlabel("r")
    ax1.set_ylabel("Densidad E")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Campo
    ax2.loglog(R_fit[::step_plot], F_fit[::step_plot], 'k.', alpha=0.3)
    ax2.loglog(R_fit, A_F * R_fit**(-n_F), 'g-', lw=2,
               label=f'Sim: {n_F:.2f}')
    ax2.loglog(
        R_fit,
        F_fit[0] * (R_fit / R_fit[0])**(-2.0),
        'r--', label='Ref: 2.00'
    )
    ax2.set_title("Cola de Campo (Newton/Coulomb)")
    ax2.set_xlabel("r")
    ax2.set_ylabel("|F|")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
