import numpy as np
import plotly.graph_objects as go

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def compute_rect_bead_height(x, y, z, w, q, dt, *,
                             ds_eps=1e-9,
                             w_eps=1e-9,
                             h_clip=(0.0, np.inf)):
    """
    Rectangular cross-section model:
      V_step = q * dt
      A = V_step / ds
      A = w * h  ->  h = A / w

    Returns h, ds, A.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    w = np.asarray(w); q = np.asarray(q)

    P = np.stack([x, y, z], axis=1)
    dP = np.diff(P, axis=0, prepend=P[:1])
    ds = np.linalg.norm(dP, axis=1)

    V_step = q * dt
    A = np.zeros_like(V_step, dtype=float)

    good = ds > ds_eps
    A[good] = V_step[good] / ds[good]

    w_safe = np.maximum(w, w_eps)
    h = np.zeros_like(w_safe, dtype=float)
    h[good] = A[good] / w_safe[good]

    hmin, hmax = h_clip
    h = np.clip(h, hmin, hmax)
    return h, ds, A

def make_swept_rectangle_mesh(
    x, y, z,
    w, h,
    *,
    stride=2,
    mask=None,
):
    """
    Sweeps a rectangle along the polyline. For each centerline point:
      - width axis aligned with a lateral vector B (perp to tangent)
      - height axis aligned with N (as close to global +Z as possible, but perp to tangent)
    Rectangle corners at (±w/2, ±h/2) in (B,N) axes.

    Returns: Vx,Vy,Vz and triangle indices i,j,k for Plotly Mesh3d.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    w = np.asarray(w); h = np.asarray(h)

    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Not enough points selected by mask to build bead mesh.")
    idx = idx[::stride]
    if idx.size < 3:
        raise ValueError("Downsampling too aggressive; reduce `stride`.")

    P = np.stack([x[idx], y[idx], z[idx]], axis=1)
    ww = w[idx]
    hh = h[idx]

    # Tangent along the path
    dP = np.gradient(P, axis=0)
    T = _normalize(dP)

    # Build frame: try to keep N close to global Z, but perpendicular to T
    Zup = np.array([0.0, 0.0, 1.0])
    proj = (T @ Zup)[:, None] * T
    N0 = Zup - proj
    N = _normalize(N0)

    # If tangent is near vertical => N0 ~ 0, pick alternate axis
    bad = np.linalg.norm(N0, axis=1) < 1e-6
    if np.any(bad):
        Xref = np.array([1.0, 0.0, 0.0])
        proj2 = (T[bad] @ Xref)[:, None] * T[bad]
        N_alt = _normalize(Xref - proj2)
        N[bad] = N_alt

    # Lateral axis for width
    B = _normalize(np.cross(T, N))

    # Rectangle corners (4 per ring): (±w/2)*B + (±h/2)*N
    # Order corners consistently around perimeter:
    # 0: -B,-N  1: +B,-N  2: +B,+N  3: -B,+N
    half_w = 0.5 * ww
    half_h = 0.5 * hh

    ring = np.zeros((P.shape[0], 4, 3), dtype=float)
    ring[:, 0, :] = P - half_w[:, None] * B - half_h[:, None] * N
    ring[:, 1, :] = P + half_w[:, None] * B - half_h[:, None] * N
    ring[:, 2, :] = P + half_w[:, None] * B + half_h[:, None] * N
    ring[:, 3, :] = P - half_w[:, None] * B + half_h[:, None] * N

    Np = ring.shape[0]
    M = 4
    V = ring.reshape(Np * M, 3)

    # Triangulate between consecutive rings
    faces_i, faces_j, faces_k = [], [], []
    for r in range(Np - 1):
        base0 = r * M
        base1 = (r + 1) * M
        for m in range(M):
            m2 = (m + 1) % M
            v00 = base0 + m
            v01 = base0 + m2
            v11 = base1 + m2
            v10 = base1 + m
            # two triangles per quad
            faces_i += [v00, v00]
            faces_j += [v01, v11]
            faces_k += [v11, v10]

    return V[:, 0], V[:, 1], V[:, 2], np.array(faces_i), np.array(faces_j), np.array(faces_k)

def plot_bead_rect_plotly(
    df,
    *,
    width_col="w_pred",         # or "w"
    dt=None,                    # pathgen.dt_real recommended
    mask_mode="q>0",            # "q>0" or "r==True" or None
    ds_min=1e-6,                # ignore near-zero motion for height computation
    h_clip=(0.0, np.inf),       # clamp bead height (helps with dwells/purges)
    stride=2,
    show_centerline=True,
    show_print_centerline=True,
):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    w = df[width_col].to_numpy()
    q = df["q"].to_numpy()

    if dt is None:
        t = df["t"].to_numpy()
        dt = float(np.median(np.diff(t)))

    # Compute height from volume/length using rectangle model
    h, ds, A = compute_rect_bead_height(
        x, y, z, w, q, dt,
        ds_eps=ds_min,
        h_clip=h_clip
    )

    # Build deposition mask
    if mask_mode is None:
        mask = np.ones_like(q, dtype=bool)
    elif mask_mode == "q>0":
        mask = q > 0
    elif mask_mode == "r==True":
        mask = df["r"].to_numpy().astype(bool)
    else:
        raise ValueError("mask_mode must be 'q>0', 'r==True', or None")

    # OPTIONAL: also require meaningful motion to avoid huge heights when ds~0
    mask = mask & (ds > ds_min)

    Vx, Vy, Vz, i, j, k = make_swept_rectangle_mesh(
        x, y, z, w, h,
        stride=stride,
        mask=mask
    )

    fig = go.Figure()

    # Bead mesh
    fig.add_trace(go.Mesh3d(
        x=Vx, y=Vy, z=Vz,
        i=i, j=j, k=k,
        opacity=0.9,
        name="Bead (rect)",
        flatshading=True,
        showscale=False,
    ))

    # Centerline context
    if show_centerline:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=3),
            name="Centerline (all moves)",
            opacity=0.25,
        ))

    if show_print_centerline:
        fig.add_trace(go.Scatter3d(
            x=x[mask], y=y[mask], z=z[mask],
            mode="lines",
            line=dict(width=5),
            name="Centerline (printing)",
            opacity=0.5,
        ))

    fig.update_layout(
        title="3D Bead Sweep (Rectangle cross-section, volume-consistent height)",
        scene=dict(
            aspectmode="data",
            xaxis_title="X [mm]",
            yaxis_title="Y [mm]",
            zaxis_title="Z [mm]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )

    fig.show()

    return fig, h, A, ds