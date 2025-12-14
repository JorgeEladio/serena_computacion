# Hopf Topological Certification (Q = 1)

This folder contains the numerical codes used to certify the topological
linking of Hopf fibers for the Q = 1 configuration in the Serena model.

Associated paper:
"Certificación Topológica Numérica de una Configuración de Hopf con Carga Unitaria"
(Zenodo, 2025)

## Contents

- serena_certification_v6_gold.py  
  Core certification backend: fiber tracing, Newton correction, Gauss linking integral.

- make_fiber_figure.py  
  Post-processing script to generate the publication-ready linked fibers figure.

- serena_hopf_certification_protocol_v2.py  
  Independent validation protocol based on helicity integration and robustness checks.

## Reproducibility

All scripts are self-contained. Python 3 with numpy and matplotlib is required.
Numba is optional but strongly recommended for performance.
