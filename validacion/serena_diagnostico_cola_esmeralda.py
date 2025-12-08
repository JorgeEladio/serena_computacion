import numpy as np
import matplotlib.pyplot as plt


"""
serena_diagnostico_cola_esmeralda.py

Diagnóstico de la cola asintótica de la configuración Esmeralda.
Incluye:
  - Ansatz Hopf compacto
  - Relajación difusiva suave
  - Cálculo de energía y campo efectivo
  - Ajuste log-log de la cola (n_E, n_F)
"""


def generate_hopf_compact(X, Y, Z, R_scale=2.5):
    """
    Genera el campo tipo Hopf mediante compactificación estereográfica
    y mapa S³ -> S².

    Devuelve nx, ny, nz.
    """
    r2 = X**2 + Y**2 + Z**2
    denom = r2 + R_scale**2

    x1 = 2 * R_scale * X / denom
    x2 = 2 * R_scale * Y / denom
    x3 = 2 * R_scale * Z / denom
    x4 = (r2 - R_scale**2) / denom

    Z0 = x1 + 1j * x2
    Z1 = x3 + 1j * x4
    n_complex = 2 * np.conj(Z0) * Z1

    nx = np.real(n_complex)
    ny = np.imag(n_complex)
    nz = np.abs(Z1)**2 - np.abs(Z0)**2
    return nx, ny, nz


def laplacian_fast(f, dx):
    """Laplaciano rápido (stencil de 7 puntos)."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) +
            np.roll(f, 1, 2) + np.roll(f, -1, 2) - 6.0 * f) / dx**2


def gradient_4th_order(f, dx, axis):
    """
    Derivada central de 4º orden:
    (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
    """
    return (-np.roll(f, -2, axis) + 8*np.roll(f, -1, axis)
            - 8*np.roll(f, 1, axis) + np.roll(f, 2, axis)) / (12 * dx)


def main():
    print("Iniciando simulación Serena (configuración Esmeralda)...")

    # 1. Dominio numérico
    L = 40.0
    N = 160
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R_dist = np.sqrt(X**2 + Y**2 + Z**2)

    print(f"Dominio: {2*L}^3")
    print(f"Resolución de malla: {N}^3")
    print(f"Paso espacial dx: {dx:.4f}")

    # 2. Ansatz Hopf compacto
    nx, ny, nz = generate_hopf_compact(X, Y, Z, R_scale=2.5)

    # 3. Relajación difusiva suave
    print("Aplicando relajación difusiva suave...")
    dt = 0.01
    steps = 100

    mask = np.ones_like(nx)
    border = 3
    mask[:border, :, :] = 0
    mask[-border:, :, :] = 0
    mask[:, :border, :] = 0
    mask[:, -border:, :] = 0
    mask[:, :, :border] = 0
    mask[:, :, -border:] = 0

    for t in range(steps):
        if t % 25 == 0:
            print(f"  Paso de relajación: {t}/{steps}...")

        nx += dt * laplacian_fast(nx, dx)
        ny += dt * laplacian_fast(ny, dx)
        nz += dt * laplacian_fast(nz, dx)

        # Renormalización |n| = 1
        norm = np.sqrt(nx**2 + ny**2 + nz**2 + 1e-12)
        nx /= norm
        ny /= norm
        nz /= norm

        # Imponer vacío en bordes
        nx *= mask
        ny *= mask
        nz = nz * mask + (1.0 - mask)

    # 4. Derivadas de 4º orden y densidad de energía
    print("Calculando derivadas de cuarto orden y densidad de energía...")

    dnx_x = gradient_4th_order(nx, dx, 0)
    dnx_y = gradient_4th_order(nx, dx, 1)
    dnx_z = gradient_4th_order(nx, dx, 2)
    dny_x = gradient_4th_order(ny, dx, 0)
    dny_y = gradient_4th_order(ny, dx, 1)
    dny_z = gradient_4th_order(ny, dx, 2)
    dnz_x = gradient_4th_order(nz, dx, 0)
    dnz_y = gradient_4th_order(nz, dx, 1)
    dnz_z = gradient_4th_order(nz, dx, 2)

    E_dens = 0.5 * (
        dnx_x**2 + dnx_y**2 + dnx_z**2 +
        dny_x**2 + dny_y**2 + dny_z**2 +
        dnz_x**2 + dnz_y**2 + dnz_z**2
    )

    Field = np.sqrt(2.0 * E_dens)

    # 5. Ajuste asintótico en región libre
    print("Realizando ajuste asintótico en la región 10 < r < 32...")
    mask_asympt = (R_dist > 10.0) & (R_dist < 32.0)

    r_clean = R_dist[mask_asympt]
    e_clean = E_dens[mask_asympt]
    f_clean = Field[mask_asympt]

    coeffs_E = np.polyfit(np.log(r_clean), np.log(e_clean), 1)
    n_E = -coeffs_E[0]
    A_E = np.exp(coeffs_E[1])

    coeffs_F = np.polyfit(np.log(r_clean), np.log(f_clean), 1)
    n_F = -coeffs_F[0]
    A_F = np.exp(coeffs_F[1])

    rel_err_E = abs(4.0 - n_E) / 4.0 * 100.0
    rel_err_F = abs(2.0 - n_F) / 2.0 * 100.0

    print("\n==============================================")
    print(" RESULTADOS ASINTÓTICOS (CONFIGURACIÓN ESMERALDA)")
    print("==============================================")
    print(f"Exponente de energía (n_E): {n_E:.5f}  [Referencia: 4.00000]")
    print(f"Exponente de campo   (n_F): {n_F:.5f}  [Referencia: 2.00000]")
    print(f"Error relativo energía: {rel_err_E:.2f}%")
    print(f"Error relativo campo  : {rel_err_F:.2f}%")
    print("==============================================\n")

    # 6. Gráficas
    plt.figure(figsize=(14, 6))

    # Energía
    plt.subplot(1, 2, 1)
    plt.loglog(r_clean[::60], e_clean[::60], 'ko', markersize=2, alpha=0.3,
               label='Datos numéricos')
    plt.loglog(r_clean, A_E * r_clean**(-n_E), 'b-', lw=2,
               label=f'Ajuste numérico: -{n_E:.4f}')
    plt.loglog(
        r_clean,
        e_clean[0] * (r_clean / r_clean[0])**(-4.0),
        'r--', lw=1.5, alpha=0.8,
        label='Referencia teórica: -4.0'
    )
    plt.title("Densidad de energía radial")
    plt.xlabel("r")
    plt.ylabel("E(r)")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", alpha=0.3)

    # Campo
    plt.subplot(1, 2, 2)
    plt.loglog(r_clean[::60], f_clean[::60], 'ko', markersize=2, alpha=0.3,
               label='Datos numéricos')
    plt.loglog(r_clean, A_F * r_clean**(-n_F), 'g-', lw=2,
               label=f'Ajuste numérico: -{n_F:.4f}')
    plt.loglog(
        r_clean,
        f_clean[0] * (r_clean / r_clean[0])**(-2.0),
        'r--', lw=1.5, alpha=0.8,
        label='Referencia teórica: -2.0'
    )
    plt.title("Campo efectivo a gran distancia")
    plt.xlabel("r")
    plt.ylabel("|F(r)|")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
