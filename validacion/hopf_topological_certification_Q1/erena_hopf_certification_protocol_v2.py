import numpy as np
import pandas as pd

# =============================================================================
# SERENA PROTOCOL V2 (CLEAN)
# - dx exacto (linspace con N puntos)
# - Ansatz físico: espinor (z0,z1) -> potencial A
# - Derivadas centradas 4º orden SIN periodicidad (sin np.roll)
# - Integración con margen mínimo correcto para derivadas encadenadas
#
# Convención de ejes (meshgrid indexing='ij'):
#   axis=0 -> X, axis=1 -> Y, axis=2 -> Z
# =============================================================================

CONFIG = {
    "L": 40.0,
    "N_list": [100, 120, 140, 160, 180, 200],
    "margins": [4, 6, 8, 10, 12],
    "R_scales": [2.0, 2.5, 3.0, 3.5],
    "controls": ["HOPF_POSITIVO", "VACIO_NEGATIVO"],
    # Margen mínimo recomendado:
    # A = d(z) invalida 2 capas; curl(A)=d(A) invalida 2 capas adicionales -> total ~4
    "min_margin": 4,
    # Tolerancia absoluta para el vacío
    "vacuum_abs_tol": 1e-4,
    # Tolerancia relativa (%) para robustez en Hopf (dispersión entre márgenes)
    "hopf_rel_tol_pct": 1.0,
}


class MathCore:
    """Derivadas no periódicas y álgebra (convención axis: 0->X, 1->Y, 2->Z)."""

    @staticmethod
    def d4_no_roll(f: np.ndarray, h: float, axis: int) -> np.ndarray:
        """
        Derivada centrada de 4º orden SIN condiciones periódicas.
        Los bordes (2 capas) quedan en 0 para evitar contaminación.

        Stencil: (-f(x+2) + 8f(x+1) - 8f(x-1) + f(x-2)) / (12 h)
        """
        res = np.zeros_like(f)
        inv_12h = 1.0 / (12.0 * h)

        if axis == 0:
            # válido en [2:-2, :, :]
            term_p2 = f[4:, :, :]
            term_p1 = f[3:-1, :, :]
            term_m1 = f[1:-3, :, :]
            term_m2 = f[:-4, :, :]
            res[2:-2, :, :] = (-term_p2 + 8 * term_p1 - 8 * term_m1 + term_m2) * inv_12h

        elif axis == 1:
            # válido en [:, 2:-2, :]
            term_p2 = f[:, 4:, :]
            term_p1 = f[:, 3:-1, :]
            term_m1 = f[:, 1:-3, :]
            term_m2 = f[:, :-4, :]
            res[:, 2:-2, :] = (-term_p2 + 8 * term_p1 - 8 * term_m1 + term_m2) * inv_12h

        elif axis == 2:
            # válido en [:, :, 2:-2]
            term_p2 = f[:, :, 4:]
            term_p1 = f[:, :, 3:-1]
            term_m1 = f[:, :, 1:-3]
            term_m2 = f[:, :, :-4]
            res[:, :, 2:-2] = (-term_p2 + 8 * term_p1 - 8 * term_m1 + term_m2) * inv_12h

        else:
            raise ValueError("axis debe ser 0, 1 o 2")

        return res

    @staticmethod
    def compute_A_from_Psi(z0: np.ndarray, z1: np.ndarray, dx: float):
        r"""
        Calcula A = Im(z^\dagger dz) numéricamente:
            A_mu = Im(conj(z0)*d_mu(z0) + conj(z1)*d_mu(z1))

        """
        # d/dX (axis 0)
        dz0 = MathCore.d4_no_roll(z0, dx, 0)
        dz1 = MathCore.d4_no_roll(z1, dx, 0)
        Ax = np.imag(np.conj(z0) * dz0 + np.conj(z1) * dz1)

        # d/dY (axis 1)
        dz0 = MathCore.d4_no_roll(z0, dx, 1)
        dz1 = MathCore.d4_no_roll(z1, dx, 1)
        Ay = np.imag(np.conj(z0) * dz0 + np.conj(z1) * dz1)

        # d/dZ (axis 2)
        dz0 = MathCore.d4_no_roll(z0, dx, 2)
        dz1 = MathCore.d4_no_roll(z1, dx, 2)
        Az = np.imag(np.conj(z0) * dz0 + np.conj(z1) * dz1)

        return Ax, Ay, Az

    @staticmethod
    def helicity_density(Ax: np.ndarray, Ay: np.ndarray, Az: np.ndarray, dx: float) -> np.ndarray:
        """
        Calcula densidad de helicidad h = A · (curl A), SIN periodicidad.
        """
        # curl A:
        dAz_dy = MathCore.d4_no_roll(Az, dx, 1)
        dAy_dz = MathCore.d4_no_roll(Ay, dx, 2)

        dAx_dz = MathCore.d4_no_roll(Ax, dx, 2)
        dAz_dx = MathCore.d4_no_roll(Az, dx, 0)

        dAy_dx = MathCore.d4_no_roll(Ay, dx, 0)
        dAx_dy = MathCore.d4_no_roll(Ax, dx, 1)

        Bx = dAz_dy - dAy_dz
        By = dAx_dz - dAz_dx
        Bz = dAy_dx - dAx_dy

        return Ax * Bx + Ay * By + Az * Bz


class AnsatzGenerator:
    """Coordenadas -> espinor (z0,z1) en S^3."""

    def __init__(self, N: int, L: float, R_scale: float):
        self.N = N
        self.L = L
        self.R = R_scale

        # dx exacto: linspace con N puntos => paso = (2L)/(N-1)
        self.r = np.linspace(-L, L, N, dtype=np.float64)
        self.dx = float(self.r[1] - self.r[0])

        # Convención 'ij': X -> axis 0, Y -> axis 1, Z -> axis 2
        self.X, self.Y, self.Z = np.meshgrid(self.r, self.r, self.r, indexing="ij")

    def generate_psi(self, mode: str):
        if mode == "VACIO_NEGATIVO":
            z0 = np.ones_like(self.X, dtype=np.complex128)
            z1 = np.zeros_like(self.X, dtype=np.complex128)
            return z0, z1

        if mode == "HOPF_POSITIVO":
            # Proyección estereográfica inversa R^3 -> S^3 (radio/escala R)
            r2 = self.X * self.X + self.Y * self.Y + self.Z * self.Z
            denom = r2 + self.R * self.R

            x1 = 2.0 * self.R * self.X / denom
            x2 = 2.0 * self.R * self.Y / denom
            x3 = 2.0 * self.R * self.Z / denom
            x4 = (r2 - self.R * self.R) / denom

            # Identificación S^3 ⊂ R^4 con C^2: z0 = x1 + i x2, z1 = x3 + i x4
            z0 = x1 + 1j * x2
            z1 = x3 + 1j * x4

            # |z0|^2 + |z1|^2 = 1 analíticamente; no renormalizamos.
            return z0, z1

        raise ValueError(f"Modo desconocido: {mode}")


def _robust_metrics(values: np.ndarray, control_type: str):
    """
    Métricas de robustez:
      - Hopf: dispersión relativa (max-min)/|mean| en %
      - Vacío: error absoluto máximo
    También devuelve std (diagnóstico).
    """
    q_mean = float(np.mean(values))
    q_max = float(np.max(values))
    q_min = float(np.min(values))
    q_std = float(np.std(values))

    if control_type == "VACIO_NEGATIVO":
        max_abs = float(np.max(np.abs(values)))
        pass_flag = max_abs < CONFIG["vacuum_abs_tol"]
        robust_str = "N/A (Abs)"
        robust_metric = max_abs
        return q_mean, q_std, pass_flag, robust_metric, robust_str

    # HOPF_POSITIVO
    if abs(q_mean) < 1e-12:
        disp_pct = float("inf")
    else:
        disp_pct = float((q_max - q_min) / abs(q_mean) * 100.0)

    pass_flag = disp_pct < CONFIG["hopf_rel_tol_pct"]
    robust_str = f"{disp_pct:.3f}%"
    robust_metric = disp_pct
    return q_mean, q_std, pass_flag, robust_metric, robust_str


def run_protocol():
    print("--- SERENA PROTOCOL V2 (CLEAN) ---")
    print("Validando: dx correcto, A desde Psi, Derivadas no-periódicas.")
    print("Convención ejes: axis 0->X, 1->Y, 2->Z.\n")

    records = []

    for control_type in CONFIG["controls"]:
        print(f">>> EJECUTANDO CONTROL: {control_type}")

        for N in CONFIG["N_list"]:
            for R in CONFIG["R_scales"]:
                # 1) Espinor analítico
                gen = AnsatzGenerator(N, CONFIG["L"], R)
                z0, z1 = gen.generate_psi(control_type)
                dx = gen.dx

                # 2) Potencial A
                Ax, Ay, Az = MathCore.compute_A_from_Psi(z0, z1, dx)

                # 3) Densidad de helicidad h = A·curl(A)
                h = MathCore.helicity_density(Ax, Ay, Az, dx)

                # 4) Integración con recorte por márgenes
                vol_element = dx ** 3
                prefactor = 1.0 / (4.0 * np.pi ** 2)

                q_by_margin = {}
                for m in CONFIG["margins"]:
                    if m < CONFIG["min_margin"]:
                        continue
                    h_cut = h[m:-m, m:-m, m:-m]
                    q_by_margin[m] = float(np.sum(h_cut) * vol_element * prefactor)

                if not q_by_margin:
                    raise RuntimeError("No hay márgenes válidos para integrar (revisa CONFIG).")

                vals = np.array(list(q_by_margin.values()), dtype=np.float64)

                q_mean, q_std, robust_pass, robust_metric, robust_str = _robust_metrics(vals, control_type)

                # Lambda (diagnóstico): 1/Q_mean solo para Hopf
                lambda_cal = 0.0
                if control_type == "HOPF_POSITIVO" and abs(q_mean) > 1e-12:
                    lambda_cal = 1.0 / q_mean

                # Guardar registros
                for m, q_raw_m in q_by_margin.items():
                    records.append(
                        {
                            "Type": control_type,
                            "N": N,
                            "dx": dx,
                            "R_scale": R,
                            "margin": m,
                            "Q_raw": q_raw_m,
                            "Q_mean_over_margins": q_mean,
                            "Q_std_over_margins": q_std,
                            "Lambda_cal": lambda_cal,
                            "Robust_Metric": robust_metric,
                            "Pass": robust_pass,
                        }
                    )

                status = "PASS" if robust_pass else "FAIL"
                print(
                    f"N={N:<3} R={R:<3} | Q_raw={q_mean:.5f} | "
                    f"Lambda={lambda_cal:.4f} | Robust={robust_str} | {status}"
                )

    # Exportar
    df = pd.DataFrame(records)
    out_csv = "serena_protocol_v2_results.csv"
    df.to_csv(out_csv, index=False)

    # Resumen final: Lambda promedio por (N,R) para Hopf
    print("\n=== DIAGNÓSTICO DE CALIBRACIÓN (LAMBDA) ===")
    print("Verificar si Lambda depende de R (Efecto de caja) o N (Discretización)")

    hopf = df[df["Type"] == "HOPF_POSITIVO"]
    if not hopf.empty:
        summary = hopf.groupby(["N", "R_scale"])["Lambda_cal"].mean().unstack()
        print(summary)
        print("\nCriterio de éxito: estable horizontal (indep. de R) y convergente vertical (con N).")

    print(f"\nCSV generado: {out_csv}")


if __name__ == "__main__":
    run_protocol()
