import importlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ============================================================
# FIGURE SCRIPT (POST-PROCESS) - PUBLICATION READY
# Requisitos:
#   - serena_certification_v6_gold.py en la MISMA carpeta.
#   - Python 3 + numpy + matplotlib + numba (opcional)
# ============================================================

MODULE_NAME = "serena_certification_v6_gold"

# Rutas absolutas para reproducibilidad
HERE = os.path.dirname(os.path.abspath(__file__))

# Parámetros físicos
N = 160
L = 40.0
R_KNOT = 2.5

# Vectores de dirección
U1 = np.array([1.0, 0.0, 0.0])  # +x
U2 = np.array([0.0, 1.0, 0.0])  # +y

# Parámetros de trazado
DS_PHYS = 0.10
MAX_STEPS = 25000
NEWTON_TOL = 1e-5
NEWTON_MAXIT = 12
EPS_FD = 0.01
CLOSURE_TOL_PHYS = 0.6
MIN_LEN_PHYS = 10.0

# Parámetros seed
SEED_STRIDE = 5
SEED_REFINE_STEPS = 15
SEED_REFINE_EPS = 0.5
SEED_LAMBDA = 0.8
SEED_MIN_DOT = 0.70
SEED_MAX_ERR = 0.35

# Borde duro (argumento requerido por el Gold file)
GUARD_IDX = 6.0  

# Nombres de salida
OUT_PNG = "fibers_linked.png"
OUT_PDF = "fibers_linked.pdf"


# ============================================================
# Helpers
# ============================================================

def _require(mod, names):
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(
            f"El módulo {MODULE_NAME} no tiene símbolos necesarios:\n"
            f"  {missing}\n"
        )

def _get_find_seed(mod):
    # Detectar función de seed disponible
    candidates = ["_find_seed_hard_border", "_find_seed", "_find_seed_hardborder"]
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name), name
    raise RuntimeError("No se encontró ninguna función de seed en el módulo.")

def _call_find_seed(find_seed_func, seed_name,
                    u, v1, v2,
                    z0_r, z0_i, z1_r, z1_i,
                    N):
    """
    Intenta llamar a la función de seed adaptándose a la firma del Gold file.
    """
    # 1. Firma Gold V6 (13 args, guard_idx al final)
    try:
        return find_seed_func(
            u, v1, v2,
            SEED_LAMBDA, SEED_STRIDE, SEED_REFINE_STEPS, SEED_REFINE_EPS,
            z0_r, z0_i, z1_r, z1_i, N, GUARD_IDX
        )
    except TypeError:
        pass

    # 2. Firma Standard (12 args, sin guard_idx explícito)
    try:
        return find_seed_func(
            u, v1, v2,
            SEED_LAMBDA, SEED_STRIDE, SEED_REFINE_STEPS, SEED_REFINE_EPS,
            z0_r, z0_i, z1_r, z1_i, N
        )
    except TypeError as e:
        raise RuntimeError(
            f"Error crítico llamando a {seed_name}. Revise argumentos.\nError: {e}"
        )

def _status_to_text(status):
    return {
        1: "OK", 0: "Open", -1: "NewtonFail", -2: "Stuck", 
        -3: "SeedNewtonFail", -4: "Out"
    }.get(status, f"Code{status}")

def _trace_one(mod, gen, u_vec, find_seed_func, seed_name):
    z0, z1 = gen.get_field(mode="HOPF")
    
    # Contiguos para Numba
    z0_r = np.ascontiguousarray(z0.real)
    z0_i = np.ascontiguousarray(z0.imag)
    z1_r = np.ascontiguousarray(z1.real)
    z1_i = np.ascontiguousarray(z1.imag)

    dx = gen.dx
    ds_idx_base = DS_PHYS / dx
    close_tol_idx = CLOSURE_TOL_PHYS / dx
    min_len_idx = MIN_LEN_PHYS / dx
    guard_val = float(GUARD_IDX)

    u = u_vec.astype(np.float64)
    v1, v2 = mod._basis_for_u(u)

    print(f"   -> Buscando seed para u={u_vec}...")
    sx, sy, sz = _call_find_seed(
        find_seed_func, seed_name,
        u, v1, v2,
        z0_r, z0_i, z1_r, z1_i,
        N
    )

    seed_dot, seed_err = mod._seed_metrics(sx, sy, sz, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
    if seed_dot < SEED_MIN_DOT:
        print(f"      [WARN] Seed calidad baja: Dot={seed_dot:.3f}")

    # Trazar (probando firma con y sin guard_idx)
    try:
        c_idx, status, avg_err = mod._trace_fiber(
            sx, sy, sz, u, v1, v2,
            z0_r, z0_i, z1_r, z1_i, N,
            ds_idx_base, MAX_STEPS, NEWTON_TOL, NEWTON_MAXIT, EPS_FD,
            close_tol_idx, min_len_idx, guard_val
        )
    except TypeError:
        c_idx, status, avg_err = mod._trace_fiber(
            sx, sy, sz, u, v1, v2,
            z0_r, z0_i, z1_r, z1_i, N,
            ds_idx_base, MAX_STEPS, NEWTON_TOL, NEWTON_MAXIT, EPS_FD,
            close_tol_idx, min_len_idx
        )

    if status != 1:
        raise RuntimeError(f"Fallo trazo u={u_vec}: {_status_to_text(status)}")

    return c_idx * dx - L

def _set_equal_axes(ax, curves):
    all_pts = np.vstack(curves)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def main():
    print(f"[INIT] Iniciando script en: {HERE}")
    
    try:
        mod = importlib.import_module(MODULE_NAME)
    except ImportError:
        print(f"[ERROR] No se encuentra '{MODULE_NAME}.py' en {HERE}")
        return

    _require(mod, ["AnsatzGenerator", "_basis_for_u", "_seed_metrics", "_trace_fiber"])
    
    find_seed_func, seed_name = _get_find_seed(mod)
    print(f"[INFO] Backend cargado. Seed function: {seed_name}")

    gen = mod.AnsatzGenerator(N, L, R_KNOT)

    try:
        c1 = _trace_one(mod, gen, U1, find_seed_func, seed_name)
        c2 = _trace_one(mod, gen, U2, find_seed_func, seed_name)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    lk = None
    if hasattr(mod, "_gauss_integral_raw"):
        lk = mod._gauss_integral_raw(c1, c2)
        print(f"[RESULTADO] Linking Number = {lk:.6f}")

    # --- PLOT SETUP (Publication Style) ---
    fig = plt.figure(figsize=(6.5, 6.5)) # Tamaño columna paper
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(c1[:, 0], c1[:, 1], c1[:, 2], lw=2.0, c='blue', label=r"Fiber $u=+\hat{x}$")
    ax.plot(c2[:, 0], c2[:, 1], c2[:, 2], lw=2.0, c='red', label=r"Fiber $u=+\hat{y}$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    title_str = f"Linked Hopf Fibers (N={N})"
    if lk is not None:
        title_str += f"\nLk = {lk:.5f}"
    ax.set_title(title_str)
    ax.legend()
    
    _set_equal_axes(ax, [c1, c2])

    # --- SAVING ---
    out_png_path = os.path.join(HERE, OUT_PNG)
    out_pdf_path = os.path.join(HERE, OUT_PDF)

    plt.tight_layout()
    
    # 1. Guardar PNG
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Imagen guardada en:\n  {out_png_path}")

    # 2. Guardar PDF (Vectorial)
    plt.savefig(out_pdf_path, bbox_inches="tight")
    if os.path.exists(out_pdf_path):
        print(f"[OK] PDF guardado en:\n  {out_pdf_path}")

    # 3. Mostrar interactivo
    print("[INFO] Mostrando figura...")
    plt.show()

if __name__ == "__main__":
    main()
