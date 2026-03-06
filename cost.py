"""
cost.py — Quadratic cost functions and analytic derivatives for iLQR FFEC.

Stage cost:
    l_k = G * (q_out_k - q_ref_k)^2 + u_k @ R @ u_k

Terminal cost:
    l_N = G_f * (q_out_N - q_ref_N)^2

The cost Hessians use the Gauss-Newton approximation (drop second-order output
term), which is standard for iLQR and ensures positive semi-definiteness:
    l_xx ≈ 2*G * dq_dx @ dq_dx.T
    l_uu ≈ 2*G * dq_du @ dq_du.T + 2*R

Inputs:
    q_out_k  [m³/s]  scalar
    q_ref_k  [m³/s]  scalar
    u_k      [2,]    [Q_cmd (m³/s), w_cmd (m)]
    G        scalar  tracking gain (unitless numerically — choose to balance tracking vs control)
    R        [2,2]   control cost matrix
    G_f      scalar  terminal tracking gain
    dq_dx    [n,]    ∂q_out/∂x  (output Jacobian w.r.t. state)
    dq_du    [2,]    ∂q_out/∂u  (output Jacobian w.r.t. control)
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Stage cost
# ------------------------------------------------------------------

def stage_cost(
    q_out_k: float,
    q_ref_k: float,
    u_k: np.ndarray,
    G: float,
    R: np.ndarray,
) -> float:
    """
    Quadratic stage cost.

    Args:
        q_out_k: predicted output flowrate [m³/s]
        q_ref_k: reference flowrate [m³/s]
        u_k:     control [Q_cmd, w_cmd], shape (2,)
        G:       tracking weight (scalar)
        R:       control weight matrix [2,2]

    Returns:
        l_k = G*(q_out_k - q_ref_k)^2 + u_k @ R @ u_k
    """
    e = q_out_k - q_ref_k
    return float(G * e * e + u_k @ R @ u_k)


# ------------------------------------------------------------------
# Terminal cost
# ------------------------------------------------------------------

def terminal_cost(
    q_out_N: float,
    q_ref_N: float,
    G_f: float,
) -> float:
    """
    Quadratic terminal cost.

    Args:
        q_out_N: output flowrate at final step [m³/s]
        q_ref_N: reference flowrate at final step [m³/s]
        G_f:     terminal tracking weight (scalar)

    Returns:
        l_N = G_f*(q_out_N - q_ref_N)^2
    """
    e = q_out_N - q_ref_N
    return float(G_f * e * e)


# ------------------------------------------------------------------
# Derivatives for iLQR backward pass
# ------------------------------------------------------------------

def stage_derivatives(
    q_out_k: float,
    q_ref_k: float,
    u_k: np.ndarray,
    dq_dx: np.ndarray,
    dq_du: np.ndarray,
    G: float,
    R: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Analytic first- and second-order stage cost derivatives.

    Args:
        q_out_k: output flowrate at step k [m³/s]
        q_ref_k: reference flowrate at step k [m³/s]
        u_k:     control vector [2,]
        dq_dx:   ∂q_out/∂x [n,]  (state output Jacobian)
        dq_du:   ∂q_out/∂u [2,]  (control output Jacobian)
        G:       tracking weight
        R:       control weight [2,2]

    Returns:
        dict with:
            l_x:  [n,]    ∂l/∂x
            l_u:  [2,]    ∂l/∂u
            l_xx: [n,n]   ∂²l/∂x²  (Gauss-Newton approximation)
            l_uu: [2,2]   ∂²l/∂u²  (Gauss-Newton approximation)
            l_xu: [n,2]   ∂²l/∂x∂u (Gauss-Newton approximation)
    """
    e = q_out_k - q_ref_k
    two_G_e = 2.0 * G * e

    l_x = two_G_e * dq_dx                          # [n]
    l_u = two_G_e * dq_du + 2.0 * (R @ u_k)        # [2]

    # Gauss-Newton: drop second-order term (∂²q_out/∂x² term)
    l_xx = (2.0 * G) * np.outer(dq_dx, dq_dx)              # [n, n]
    l_uu = (2.0 * G) * np.outer(dq_du, dq_du) + 2.0 * R    # [2, 2]
    l_xu = (2.0 * G) * np.outer(dq_dx, dq_du)              # [n, 2]

    return {"l_x": l_x, "l_u": l_u, "l_xx": l_xx, "l_uu": l_uu, "l_xu": l_xu}


def terminal_derivatives(
    q_out_N: float,
    q_ref_N: float,
    dq_dx: np.ndarray,
    G_f: float,
) -> dict[str, np.ndarray]:
    """
    Terminal cost derivatives for the Riccati boundary condition.

    Args:
        q_out_N: output flowrate at terminal step [m³/s]
        q_ref_N: reference flowrate at terminal step [m³/s]
        dq_dx:   ∂q_out/∂x [n,]
        G_f:     terminal tracking weight

    Returns:
        dict with:
            l_x:  [n,]   ∂l_N/∂x  (terminal gradient)
            l_xx: [n,n]  ∂²l_N/∂x² (terminal Hessian, Gauss-Newton)
    """
    e = q_out_N - q_ref_N
    l_x = (2.0 * G_f * e) * dq_dx
    l_xx = (2.0 * G_f) * np.outer(dq_dx, dq_dx)
    return {"l_x": l_x, "l_xx": l_xx}
