import numpy as np

print("--- PRUEBA DE ADN: CERTIFICACIÓN TOPOLÓGICA (WHITEHEAD) ---")
print("Objetivo: Confirmar Carga de Hopf Q = 1.000 con máscara de seguridad.")

# 1. CONFIGURACIÓN (Coincidente con el Paper)
L = 40.0; N = 140
x = np.linspace(-L, L, N); dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
R_sq = X**2 + Y**2 + Z**2

# 2. GENERACIÓN DEL ANSATZ ANALÍTICO
# Validamos el objeto matemático puro antes de la discretización
R_scale = 2.5; denom = R_sq + R_scale**2
x1 = 2 * R_scale * X / denom; x2 = 2 * R_scale * Y / denom
x3 = 2 * R_scale * Z / denom; x4 = (R_sq - R_scale**2) / denom
z0 = x1 + 1j * x2; z1 = x3 + 1j * x4

# 3. DERIVADAS DE 4TO ORDEN (Alta Precisión)
def d4(f, axis):
    return (-np.roll(f, -2, axis) + 8*np.roll(f, -1, axis) 
            - 8*np.roll(f, 1, axis) + np.roll(f, 2, axis)) / (12 * dx)

# 4. CÁLCULO DE GAUGE (A) Y CURVATURA (B)
d0x=d4(z0,0); d1x=d4(z1,0); Ax=np.imag(np.conj(z0)*d0x+np.conj(z1)*d1x)
d0y=d4(z0,1); d1y=d4(z1,1); Ay=np.imag(np.conj(z0)*d0y+np.conj(z1)*d1y)
d0z=d4(z0,2); d1z=d4(z1,2); Az=np.imag(np.conj(z0)*d0z+np.conj(z1)*d1z)

Bx = d4(Az,1)-d4(Ay,2); By = d4(Ax,2)-d4(Az,0); Bz = d4(Ay,0)-d4(Ax,1)

# 5. INTEGRAL CON MÁSCARA DE SEGURIDAD (El Ajuste del Revisor)
# Apagamos la integral en los bordes para eliminar el error de np.roll
mask = np.ones_like(Ax)
b = 4  # Margen de seguridad de 4 píxeles
mask[:b,:,:]=0; mask[-b:,:,:]=0
mask[:,:b,:]=0; mask[:,-b:,:]=0
mask[:,:,:b]=0; mask[:,:,-b:]=0

# Densidad de Helicidad h = A . B
h_dens = Ax*Bx + Ay*By + Az*Bz
Integral_H = np.sum(h_dens * mask) * dx**3

# Carga Normalizada (Factor teórico exacto: 1 / 4pi^2)
Q_calc = Integral_H / (4 * np.pi**2)

print("-" * 40)
print(f"CARGA DE HOPF (Q):  {Q_calc:.6f}")
print(f"Valor Teórico:      1.000000")
print(f"Error Numérico:     {abs(1-Q_calc)*100:.4f}%")
print("-" * 40)
