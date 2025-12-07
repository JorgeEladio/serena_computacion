import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# APENDICE A: VALIDACION ASINTOTICA (CONFIGURACION ESMERALDA)
# Incluye: Ansatz compacto, Relajacion difusiva y Derivadas de 4to orden
# =============================================================================

print("Iniciando simulacion Serena (Configuracion Esmeralda)...")

# -----------------------------------------------------------------------------
# 1. DOMINIO Y DISCRETIZACION (Ref: Sec 3.1)
# -----------------------------------------------------------------------------
L = 40.0           # Semilado del universo (Evita efectos de borde)
N = 140            # Resolucion alta (140^3 puntos)
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Malla 3D (indexing='ij' para correspondencia matricial correcta)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)

# -----------------------------------------------------------------------------
# 2. ANSATZ COMPACTO ESTEREOGRAFICO (R3 -> S3 -> S2) (Ref: Sec 2)
# -----------------------------------------------------------------------------
# Garantiza n -> (0,0,1) en el infinito
R_scale = 2.5      # Escala del nucleo
denom = R**2 + R_scale**2

# Proyeccion inversa estereografica
x1 = 2 * R_scale * X / denom
x2 = 2 * R_scale * Y / denom
x3 = 2 * R_scale * Z / denom
x4 = (R**2 - R_scale**2) / denom

# Campos complejos auxiliares
Z0 = x1 + 1j * x2
Z1 = x3 + 1j * x4

# Mapa de Hopf: n = Z_dag * sigma * Z
n_complex = 2 * np.conj(Z0) * Z1
nx = np.real(n_complex)
ny = np.imag(n_complex)
nz = np.abs(Z1)**2 - np.abs(Z0)**2

# -----------------------------------------------------------------------------
# 3. RELAJACION DIFUSIVA (Suavizado geometrico - Ref: Sec 3.2)
# -----------------------------------------------------------------------------
# Laplaciano rapido (stencil de 7 puntos)
def laplacian_fast(f, dx):
    return (np.roll(f,1,0) + np.roll(f,-1,0) +
            np.roll(f,1,1) + np.roll(f,-1,1) +
            np.roll(f,1,2) + np.roll(f,-1,2) - 6*f) / dx**2

# Mascara de frontera (Fijar vacio en los bordes)
mask = np.ones_like(nx); m = 2
mask[:m,:,:]=0; mask[-m:,:,:]=0
mask[:,:m,:]=0; mask[:,-m:,:]=0
mask[:,:,:m]=0; mask[:,:,-m:]=0

# Bucle de relajacion breve
print("Aplicando relajacion difusiva para eliminar ruido...")
dt = 0.005
steps = 50

for t in range(steps):
    # Ecuacion de calor: dn/dt = Laplaciano(n)
    nx += dt * laplacian_fast(nx, dx)
    ny += dt * laplacian_fast(ny, dx)
    nz += dt * laplacian_fast(nz, dx)
    
    # Renormalizacion OBLIGATORIA (mantenerse en S2)
    # Esto elimina errores de redondeo acumulados
    norm = np.sqrt(nx**2 + ny**2 + nz**2 + 1e-12)
    nx/=norm; ny/=norm; nz/=norm
    
    # Aplicar condiciones de frontera (Clamp al Vacio)
    nx *= mask
    ny *= mask
    nz = nz * mask + (1-mask)

# -----------------------------------------------------------------------------
# 4. CALCULO DE ENERGIA (Derivadas de 4to Orden - Ref: Sec 3.3)
# -----------------------------------------------------------------------------
# Derivada centrada de 4to orden (Precision O(h^4))
def d4(f, axis):
    return (-np.roll(f, -2, axis) + 8*np.roll(f, -1, axis) 
            - 8*np.roll(f, 1, axis) + np.roll(f, 2, axis)) / (12 * dx)

print("Calculando derivadas de alta precision...")
# Densidad de Energia: E = 1/2 * sum((dn_i)^2)
E = 0.0
for comp in (nx, ny, nz):
    for ax in (0, 1, 2):
        E += 0.5 * d4(comp, ax)**2

# Campo Efectivo (Proxy de curvatura)
F = np.sqrt(2 * E)

# -----------------------------------------------------------------------------
# 5. AJUSTE ASINTOTICO (Ref: Sec 3.4)
# -----------------------------------------------------------------------------
# Ventana radial libre de nucleo y bordes
mask_fit = (R > 12) & (R < 32)

# Regresion lineal Log-Log para Energia
coeffs_E = np.polyfit(np.log(R[mask_fit]), np.log(E[mask_fit]), 1)
n_E = -coeffs_E[0]
A_E = np.exp(coeffs_E[1])

# Regresion lineal Log-Log para Campo
coeffs_F = np.polyfit(np.log(R[mask_fit]), np.log(F[mask_fit]), 1)
n_F = -coeffs_F[0]
A_F = np.exp(coeffs_F[1])

# Salida de resultados
print("-" * 40)
print("RESULTADOS DEL ANALISIS:")
print(f"Exponente Energia (Masa): {n_E:.4f} (Teorico: 4.00)")
print(f"Exponente Campo (Fuerza): {n_F:.4f} (Teorico: 2.00)")
print("-" * 40)

# -----------------------------------------------------------------------------
# 6. GRAFICAS
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Submuestreo para graficar puntos ligeros
step_plot = 30

# Grafica Energia
ax1.loglog(R[mask_fit][::step_plot], E[mask_fit][::step_plot], 'k.', alpha=0.3)
ax1.loglog(R[mask_fit], A_E * R[mask_fit]**-n_E, 'b-', lw=2, label=f'Sim: {n_E:.2f}')
ax1.loglog(R[mask_fit], E[mask_fit][0]*(R[mask_fit]/R[mask_fit][0])**-4, 'r--', label='Ref: 4.00')
ax1.set_title("Cola de Energia (Masa Finita)")
ax1.set_xlabel("r"); ax1.set_ylabel("Densidad E")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

# Grafica Campo
ax2.loglog(R[mask_fit][::step_plot], F[mask_fit][::step_plot], 'k.', alpha=0.3)
ax2.loglog(R[mask_fit], A_F * R[mask_fit]**-n_F, 'g-', lw=2, label=f'Sim: {n_F:.2f}')
ax2.loglog(R[mask_fit], F[mask_fit][0]*(R[mask_fit]/R[mask_fit][0])**-2, 'r--', label='Ref: 2.00')
ax2.set_title("Cola de Campo (Newton/Coulomb)")
ax2.set_xlabel("r"); ax2.set_ylabel("|F|")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
