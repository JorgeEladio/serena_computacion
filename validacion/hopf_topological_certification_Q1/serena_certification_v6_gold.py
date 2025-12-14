import numpy as np
import time

# Intentar importar Numba
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("ADVERTENCIA: Numba no detectado. Será lento.")
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

print("=== SERENA CERTIFICATION V31.2 (HARD BORDER SEED + RETRY OUT + NEWTON DAMP ADAPT) ===")
print(f"Aceleracion: {'[NUMBA ACTIVO]' if HAS_NUMBA else '[SOLO PYTHON]'}")

CONFIG = {
    "N_list": [120, 160, 200],
    "L": 40.0,
    "R_knot": 2.5,

    "link_pairs": [
        (np.array([0., 0.,  1.]), np.array([0., 0., -1.])),
        (np.array([1., 0.,  0.]), np.array([-1., 0., 0.])),
        (np.array([0., 1.,  0.]), np.array([0., -1., 0.])),
    ],

    "ds_phys": 0.10,
    "max_steps": 25000,
    "newton_tol": 1e-5,
    "newton_maxit": 12,
    "closure_tol_phys": 0.6,
    "min_len_phys": 10.0,

    "seed_stride": 5,
    "seed_refine_steps": 15,
    "seed_refine_eps": 0.5,
    "seed_lambda": 0.8,
    "seed_min_dot": 0.70,
    "seed_max_err": 0.35,

    "eps_fd": 0.01,

    # --- Cambios clave ---
    "border_guard_phys": 3.0,          # antes 1.0 (esto es lo importante)
    "retry_scales": [1.0, 0.7, 0.5, 0.35],
    "min_ds_scale": 0.25,
    "max_retry": 4,
}

@jit(nopython=True)
def _clamp(val, a, b):
    if val < a:
        return a
    if val > b:
        return b
    return val

@jit(nopython=True)
def _fast_interp(x, y, z, data_r, data_i):
    max_idx = data_r.shape[0] - 1
    x = _clamp(x, 0.0, float(max_idx))
    y = _clamp(y, 0.0, float(max_idx))
    z = _clamp(z, 0.0, float(max_idx))

    x0 = int(x); x1 = min(x0 + 1, max_idx)
    y0 = int(y); y1 = min(y0 + 1, max_idx)
    z0 = int(z); z1 = min(z0 + 1, max_idx)

    dx = x - x0; dy = y - y0; dz = z - z0

    # Real
    c00 = data_r[x0, y0, z0]*(1-dx) + data_r[x1, y0, z0]*dx
    c01 = data_r[x0, y0, z1]*(1-dx) + data_r[x1, y0, z1]*dx
    c10 = data_r[x0, y1, z0]*(1-dx) + data_r[x1, y1, z0]*dx
    c11 = data_r[x0, y1, z1]*(1-dx) + data_r[x1, y1, z1]*dx
    c0 = c00*(1-dy) + c10*dy
    c1 = c01*(1-dy) + c11*dy
    vr = c0*(1-dz) + c1*dz

    # Imag
    c00 = data_i[x0, y0, z0]*(1-dx) + data_i[x1, y0, z0]*dx
    c01 = data_i[x0, y0, z1]*(1-dx) + data_i[x1, y0, z1]*dx
    c10 = data_i[x0, y1, z0]*(1-dx) + data_i[x1, y1, z0]*dx
    c11 = data_i[x0, y1, z1]*(1-dx) + data_i[x1, y1, z1]*dx
    c0 = c00*(1-dy) + c10*dy
    c1 = c01*(1-dy) + c11*dy
    vi = c0*(1-dz) + c1*dz

    return vr, vi

@jit(nopython=True)
def _n_components(ix, iy, iz, z0_r, z0_i, z1_r, z1_i):
    zr0, zi0 = _fast_interp(ix, iy, iz, z0_r, z0_i)
    zr1, zi1 = _fast_interp(ix, iy, iz, z1_r, z1_i)

    nx = 2.0 * (zr0*zr1 + zi0*zi1)
    ny = 2.0 * (zi0*zr1 - zr0*zi1)
    nz = (zr0*zr0 + zi0*zi0) - (zr1*zr1 + zi1*zi1)

    nsq = nx*nx + ny*ny + nz*nz
    if nsq < 1e-24:
        return 0.0, 0.0, 1.0
    inv = 1.0 / np.sqrt(nsq)
    return nx*inv, ny*inv, nz*inv

@jit(nopython=True)
def _F_values(ix, iy, iz, v1, v2, z0_r, z0_i, z1_r, z1_i):
    nx, ny, nz = _n_components(ix, iy, iz, z0_r, z0_i, z1_r, z1_i)
    f1 = nx*v1[0] + ny*v1[1] + nz*v1[2]
    f2 = nx*v2[0] + ny*v2[1] + nz*v2[2]
    return f1, f2

@jit(nopython=True)
def _seed_metrics(ix, iy, iz, u, v1, v2, z0_r, z0_i, z1_r, z1_i):
    nx, ny, nz = _n_components(ix, iy, iz, z0_r, z0_i, z1_r, z1_i)
    dotu = nx*u[0] + ny*u[1] + nz*u[2]
    f1 = nx*v1[0] + ny*v1[1] + nz*v1[2]
    f2 = nx*v2[0] + ny*v2[1] + nz*v2[2]
    err = np.sqrt(f1*f1 + f2*f2)
    return dotu, err

@jit(nopython=True)
def _find_seed_hard_border(u, v1, v2, lam, stride, refine_steps, refine_eps,
                           z0_r, z0_i, z1_r, z1_i, N, guard_idx):
    # IMPORTANTE: no exploramos cerca del borde
    g = int(guard_idx)
    if g < 2:
        g = 2
    lo = g
    hi = (N - 1) - g
    if hi <= lo + 2:
        # caja demasiado pequeña
        return N/2.0, N/2.0, N/2.0

    best_s = -1e9
    bx = (lo + hi) * 0.5
    by = bx
    bz = bx

    for i in range(lo, hi, stride):
        fi = float(i)
        for j in range(lo, hi, stride):
            fj = float(j)
            for k in range(lo, hi, stride):
                fk = float(k)
                dotu, err = _seed_metrics(fi, fj, fk, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
                s = dotu - lam * err
                if s > best_s:
                    best_s = s
                    bx, by, bz = fi, fj, fk

    # hill climb local (respetando borde duro)
    cx, cy, cz = bx, by, bz
    for _ in range(refine_steps):
        dotu0, err0 = _seed_metrics(cx, cy, cz, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
        s0 = dotu0 - lam * err0
        best = s0
        nx, ny, nz = 0.0, 0.0, 0.0

        for dx in (refine_eps, -refine_eps):
            x = cx + dx
            if x <= lo or x >= hi:
                continue
            dotu, err = _seed_metrics(x, cy, cz, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
            s = dotu - lam * err
            if s > best:
                best = s; nx = dx; ny = 0.0; nz = 0.0

        for dy in (refine_eps, -refine_eps):
            y = cy + dy
            if y <= lo or y >= hi:
                continue
            dotu, err = _seed_metrics(cx, y, cz, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
            s = dotu - lam * err
            if s > best:
                best = s; nx = 0.0; ny = dy; nz = 0.0

        for dz in (refine_eps, -refine_eps):
            z = cz + dz
            if z <= lo or z >= hi:
                continue
            dotu, err = _seed_metrics(cx, cy, z, u, v1, v2, z0_r, z0_i, z1_r, z1_i)
            s = dotu - lam * err
            if s > best:
                best = s; nx = 0.0; ny = 0.0; nz = dz

        if best > s0 + 1e-6:
            cx += nx; cy += ny; cz += nz
        else:
            break

    return cx, cy, cz

@jit(nopython=True)
def _newton_correct(ix, iy, iz, v1, v2, tol, maxit, eps_fd, z0_r, z0_i, z1_r, z1_i):
    x = ix; y = iy; z = iz
    damp = 0.7

    for _ in range(maxit):
        f1, f2 = _F_values(x, y, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
        err2 = f1*f1 + f2*f2
        if err2 < tol*tol:
            return x, y, z, True, np.sqrt(err2)

        f1p, f2p = _F_values(x+eps_fd, y, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(x-eps_fd, y, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dx = (f1p - f1m) / (2*eps_fd)
        df2dx = (f2p - f2m) / (2*eps_fd)

        f1p, f2p = _F_values(x, y+eps_fd, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(x, y-eps_fd, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dy = (f1p - f1m) / (2*eps_fd)
        df2dy = (f2p - f2m) / (2*eps_fd)

        f1p, f2p = _F_values(x, y, z+eps_fd, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(x, y, z-eps_fd, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dz = (f1p - f1m) / (2*eps_fd)
        df2dz = (f2p - f2m) / (2*eps_fd)

        a11 = df1dx*df1dx + df1dy*df1dy + df1dz*df1dz
        a12 = df1dx*df2dx + df1dy*df2dy + df1dz*df2dz
        a22 = df2dx*df2dx + df2dy*df2dy + df2dz*df2dz
        det = a11*a22 - a12*a12
        if det < 1e-18:
            return x, y, z, False, np.sqrt(err2)

        inv11 =  a22 / det
        inv12 = -a12 / det
        inv22 =  a11 / det

        w1 = inv11*f1 + inv12*f2
        w2 = inv12*f1 + inv22*f2

        dx = -(df1dx*w1 + df2dx*w2)
        dy = -(df1dy*w1 + df2dy*w2)
        dz = -(df1dz*w1 + df2dz*w2)

        xn = x + damp*dx
        yn = y + damp*dy
        zn = z + damp*dz
        f1n, f2n = _F_values(xn, yn, zn, v1, v2, z0_r, z0_i, z1_r, z1_i)
        err2n = f1n*f1n + f2n*f2n

        if err2n < err2:
            x, y, z = xn, yn, zn
            damp = min(0.9, damp*1.1)
        else:
            damp *= 0.5
            if damp < 0.05:
                return x, y, z, False, np.sqrt(err2)

    f1, f2 = _F_values(x, y, z, v1, v2, z0_r, z0_i, z1_r, z1_i)
    err2 = f1*f1 + f2*f2
    return x, y, z, (err2 < tol*tol), np.sqrt(err2)

@jit(nopython=True)
def _trace_fiber(start_ix, start_iy, start_iz, u, v1, v2,
                 z0_r, z0_i, z1_r, z1_i, N,
                 ds_idx_base, max_steps,
                 tol, newton_maxit, eps_fd,
                 close_tol_idx, min_len_idx,
                 guard_idx):
    path = np.zeros((max_steps, 3), dtype=np.float64)

    sx, sy, sz, ok, e0 = _newton_correct(start_ix, start_iy, start_iz, v1, v2, tol, newton_maxit, eps_fd,
                                         z0_r, z0_i, z1_r, z1_i)
    if not ok:
        path[0] = np.array([start_ix, start_iy, start_iz])
        return path[:1], -3, e0

    cx, cy, cz = sx, sy, sz
    path[0] = np.array([cx, cy, cz])
    count = 1

    ptx, pty, ptz = 0.0, 0.0, 0.0
    total_err = 0.0
    cur_ds = ds_idx_base

    ux, uy, uz = u[0], u[1], u[2]

    for step in range(1, max_steps):
        nx, ny, nz = _n_components(cx, cy, cz, z0_r, z0_i, z1_r, z1_i)
        if nx*ux + ny*uy + nz*uz < 0.0:
            return path[:count], 0, (total_err / max(1, count))

        f1p, f2p = _F_values(cx+eps_fd, cy, cz, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(cx-eps_fd, cy, cz, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dx = (f1p - f1m)/(2*eps_fd); df2dx = (f2p - f2m)/(2*eps_fd)

        f1p, f2p = _F_values(cx, cy+eps_fd, cz, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(cx, cy-eps_fd, cz, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dy = (f1p - f1m)/(2*eps_fd); df2dy = (f2p - f2m)/(2*eps_fd)

        f1p, f2p = _F_values(cx, cy, cz+eps_fd, v1, v2, z0_r, z0_i, z1_r, z1_i)
        f1m, f2m = _F_values(cx, cy, cz-eps_fd, v1, v2, z0_r, z0_i, z1_r, z1_i)
        df1dz = (f1p - f1m)/(2*eps_fd); df2dz = (f2p - f2m)/(2*eps_fd)

        tx = df1dy*df2dz - df1dz*df2dy
        ty = df1dz*df2dx - df1dx*df2dz
        tz = df1dx*df2dy - df1dy*df2dx

        tnorm = np.sqrt(tx*tx + ty*ty + tz*tz)
        if tnorm < 1e-14:
            return path[:count], -2, (total_err / max(1, count))

        inv = 1.0/tnorm
        tx *= inv; ty *= inv; tz *= inv

        if step > 1 and (tx*ptx + ty*pty + tz*ptz) < 0.0:
            tx = -tx; ty = -ty; tz = -tz
        ptx, pty, ptz = tx, ty, tz

        converged = False
        attempt = cur_ds
        px = cx; py = cy; pz = cz
        last_err = 0.0

        for _ in range(6):
            px = cx + tx*attempt
            py = cy + ty*attempt
            pz = cz + tz*attempt

            nx2, ny2, nz2, ok2, e2 = _newton_correct(px, py, pz, v1, v2, tol, newton_maxit, eps_fd,
                                                     z0_r, z0_i, z1_r, z1_i)
            last_err = e2
            if ok2:
                px, py, pz = nx2, ny2, nz2
                converged = True
                if attempt < ds_idx_base:
                    cur_ds = min(ds_idx_base, attempt*1.5)
                break
            attempt *= 0.5
            cur_ds = attempt

        if not converged:
            return path[:count], -1, (total_err / max(1, count))

        cx, cy, cz = px, py, pz
        total_err += last_err

        if (cx < guard_idx or cx > (N-1)-guard_idx or
            cy < guard_idx or cy > (N-1)-guard_idx or
            cz < guard_idx or cz > (N-1)-guard_idx):
            return path[:count], -4, (total_err / max(1, count))

        if step > 60:
            dx0 = cx - path[0, 0]
            dy0 = cy - path[0, 1]
            dz0 = cz - path[0, 2]
            d2 = dx0*dx0 + dy0*dy0 + dz0*dz0
            if d2 < close_tol_idx*close_tol_idx:
                if (count * ds_idx_base) > min_len_idx:
                    path[count] = path[0]
                    return path[:count+1], 1, (total_err / max(1, count))

        path[count] = np.array([cx, cy, cz])
        count += 1

    return path[:count], 0, (total_err / max(1, count))

@jit(nopython=True)
def _gauss_integral_raw(c1, c2):
    L = 0.0
    n1 = len(c1); n2 = len(c2)
    for i in range(n1-1):
        r1x = c1[i,0]; r1y = c1[i,1]; r1z = c1[i,2]
        dr1x = c1[i+1,0]-r1x; dr1y = c1[i+1,1]-r1y; dr1z = c1[i+1,2]-r1z
        m1x = r1x + 0.5*dr1x; m1y = r1y + 0.5*dr1y; m1z = r1z + 0.5*dr1z

        for j in range(n2-1):
            r2x = c2[j,0]; r2y = c2[j,1]; r2z = c2[j,2]
            dr2x = c2[j+1,0]-r2x; dr2y = c2[j+1,1]-r2y; dr2z = c2[j+1,2]-r2z
            m2x = r2x + 0.5*dr2x; m2y = r2y + 0.5*dr2y; m2z = r2z + 0.5*dr2z

            dx = m1x - m2x; dy = m1y - m2y; dz = m1z - m2z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-15

            cx = dr1y*dr2z - dr1z*dr2y
            cy = dr1z*dr2x - dr1x*dr2z
            cz = dr1x*dr2y - dr1y*dr2x

            num = dx*cx + dy*cy + dz*cz
            L += num / (dist*dist*dist)

    return L / (4.0*np.pi)

class AnsatzGenerator:
    def __init__(self, N, L, R):
        self.N = N
        self.L = L
        self.R = R
        self.r = np.linspace(-L, L, N, dtype=np.float64)
        self.dx = self.r[1] - self.r[0]
        self.X, self.Y, self.Z = np.meshgrid(self.r, self.r, self.r, indexing="ij")

    def get_field(self, mode="HOPF"):
        if mode == "VACUUM":
            z0 = np.ones_like(self.X, dtype=np.complex128)
            z1 = np.zeros_like(self.X, dtype=np.complex128)
        else:
            r2 = self.X*self.X + self.Y*self.Y + self.Z*self.Z
            den = r2 + self.R*self.R
            x1 = 2*self.R*self.X/den
            x2 = 2*self.R*self.Y/den
            x3 = 2*self.R*self.Z/den
            x4 = (r2 - self.R*self.R)/den
            z0 = x1 + 1j*x2
            z1 = x3 + 1j*x4
        return z0, z1

def _basis_for_u(u):
    u = u.astype(np.float64)
    if abs(u[0]) > 0.9:
        tmp = np.array([0., 1., 0.])
    else:
        tmp = np.array([1., 0., 0.])
    v1 = np.cross(u, tmp)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    return v1, v2

def _trace_with_retry(sx, sy, sz, u, v1, v2,
                      z0_r, z0_i, z1_r, z1_i, N,
                      ds_idx_base, max_steps, tol, newton_maxit, eps_fd,
                      close_tol_idx, min_len_idx,
                      guard_idx):
    best_curve = None
    best_status = -999
    best_err = 1e30
    best_tries = 0

    for tries, scale in enumerate(CONFIG["retry_scales"][:CONFIG["max_retry"]]):
        ds_try = ds_idx_base * scale
        if ds_try < ds_idx_base * CONFIG["min_ds_scale"]:
            ds_try = ds_idx_base * CONFIG["min_ds_scale"]

        c_idx, status, avg_err = _trace_fiber(
            sx, sy, sz, u, v1, v2,
            z0_r, z0_i, z1_r, z1_i,
            N,
            ds_try,
            max_steps,
            tol,
            newton_maxit,
            eps_fd,
            close_tol_idx,
            min_len_idx,
            guard_idx
        )

        if status == 1:
            return c_idx, status, avg_err, tries

        # mejor “no OK”: más puntos y menor error
        score = (len(c_idx) / max(1e-12, avg_err))
        best_score = (len(best_curve) / max(1e-12, best_err)) if best_curve is not None else -1e9
        if best_curve is None or score > best_score:
            best_curve = c_idx
            best_status = status
            best_err = avg_err
            best_tries = tries

    return best_curve, best_status, best_err, best_tries

def run_scientific_check():
    print("[Numba] Compilando kernels...", flush=True)
    d = np.zeros((5,5,5), dtype=np.float64)
    u0 = np.array([0.,0.,1.])
    v10, v20 = _basis_for_u(u0)
    _find_seed_hard_border(u0, v10, v20, 0.8, 2, 2, 0.5, d,d,d,d, 5, 2.0)
    _newton_correct(2.0,2.0,2.0, v10, v20, 1e-3, 2, 0.05, d,d,d,d)
    _trace_fiber(2.0,2.0,2.0, u0, v10, v20, d,d,d,d, 5, 0.1, 10, 1e-3, 2, 0.05, 1.0, 1.0, 2.0)

    for N in CONFIG["N_list"]:
        print(f"\n################ RESOLUCION N={N} ################")
        gen = AnsatzGenerator(N, CONFIG["L"], CONFIG["R_knot"])
        dx = gen.dx

        ds_idx_base = CONFIG["ds_phys"] / dx
        close_tol_idx = CONFIG["closure_tol_phys"] / dx
        min_len_idx = CONFIG["min_len_phys"] / dx
        guard_idx = max(2.0, CONFIG["border_guard_phys"] / dx)

        for mode in ["HOPF", "VACUUM"]:
            print(f"\n>>> ESCENARIO: {mode}")
            t0 = time.time()

            z0, z1 = gen.get_field(mode=mode)
            z0_r = np.ascontiguousarray(z0.real)
            z0_i = np.ascontiguousarray(z0.imag)
            z1_r = np.ascontiguousarray(z1.real)
            z1_i = np.ascontiguousarray(z1.imag)

            valid_links = []

            for ip, (uA, uB) in enumerate(CONFIG["link_pairs"]):
                pair_curves = []
                for target in (uA, uB):
                    u = target.astype(np.float64)
                    v1, v2 = _basis_for_u(u)

                    sx, sy, sz = _find_seed_hard_border(
                        u, v1, v2,
                        CONFIG["seed_lambda"],
                        CONFIG["seed_stride"],
                        CONFIG["seed_refine_steps"],
                        CONFIG["seed_refine_eps"],
                        z0_r, z0_i, z1_r, z1_i,
                        N,
                        guard_idx
                    )

                    seed_dot, seed_err = _seed_metrics(sx, sy, sz, u, v1, v2, z0_r, z0_i, z1_r, z1_i)

                    if seed_dot < CONFIG["seed_min_dot"] or seed_err > CONFIG["seed_max_err"]:
                        print(f"  Target {target}: BadSeed (SeedDot={seed_dot:.2f}, SeedErr={seed_err:.2e})")
                        continue

                    c_idx, status, avg_err, tries = _trace_with_retry(
                        sx, sy, sz,
                        u, v1, v2,
                        z0_r, z0_i, z1_r, z1_i,
                        N,
                        ds_idx_base,
                        CONFIG["max_steps"],
                        CONFIG["newton_tol"],
                        CONFIG["newton_maxit"],
                        CONFIG["eps_fd"],
                        close_tol_idx,
                        min_len_idx,
                        guard_idx
                    )

                    st_map = {1:"OK", 0:"Open", -1:"NewtonFail", -2:"Stuck", -3:"SeedNewtonFail", -4:"Out"}
                    st = st_map.get(status, f"Code{status}")

                    if st == "OK":
                        c_phys = c_idx * dx - CONFIG["L"]
                        pair_curves.append(c_phys)
                        if tries > 0:
                            print(f"  Target {target}: OK* (retry={tries}, Pts={len(c_phys)}, SeedDot={seed_dot:.2f}, SeedErr={seed_err:.2e}, AvgErr={avg_err:.2e})")
                        else:
                            print(f"  Target {target}: OK (Pts={len(c_phys)}, SeedDot={seed_dot:.2f}, SeedErr={seed_err:.2e}, AvgErr={avg_err:.2e})")
                    else:
                        print(f"  Target {target}: {st} (retry={tries}, Pts={len(c_idx)}, SeedDot={seed_dot:.2f}, SeedErr={seed_err:.2e}, AvgErr={avg_err:.2e})")

                if len(pair_curves) == 2:
                    Lk = _gauss_integral_raw(pair_curves[0], pair_curves[1])
                    valid_links.append(Lk)
                    print(f"    -> Linking Par {ip}: {Lk:.6f}")
                else:
                    if mode == "HOPF":
                        print(f"    -> Linking Par {ip}: N/A")

            if mode == "HOPF":
                if valid_links:
                    ok_int = all(abs(l - round(l)) < 0.05 for l in valid_links)
                    print(f"  RESULTADO FINAL: {'CERTIFICADO' if ok_int else 'DRIFT'} | Links={['%.6f'%x for x in valid_links]}")
                else:
                    print("  RESULTADO FINAL: FAIL (Sin pares completos)")

            print(f"  Tiempo: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_scientific_check()
