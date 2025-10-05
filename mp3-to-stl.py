import numpy as np, librosa, trimesh
from trimesh import repair, smoothing

def ultra_mvp_vase_resampled(
    audio="audio.mp3",
    output="output.stl",
    H=150.0,            # height (mm)
    R=38.0,             # baseline radius (mm)
    VAR=9.0,            # radius wobble (±mm)
    slices=300,         # fixed vertical slices -> small STL
    smooth_secs=0.5,    # rolling window (seconds)
    wall=0,             # wall thickness (mm); set 0 for single-shell
    add_top=True,       # add top padding & closure
    add_bottom=True,    # add bottom padding & closure
    pad_mm=2.0          # base pad height in mm (applies to both top and bottom)
):
    # Hard-set internals most users won't change
    n_theta = 96               # points around circumference
    rms_frame = 2048
    hop = 512
    smooth_mesh_iters = 5     # geometry smoothing iterations
    taubin_lambda = 0.5
    taubin_nu = -0.5
    min_radius = 2.0           # avoid collapsed rings
    overhang_deg = 45          # max overhang angle (degrees); None = no limit
    
    # 1) Load audio (mono)
    y, sr = librosa.load(audio, mono=True)

    # 2) RMS envelope
    rms = librosa.feature.rms(y=y, frame_length=rms_frame, hop_length=hop).flatten()

    # 3) Rolling average in seconds
    win_frames = max(3, int(smooth_secs * sr / hop) | 1)  # odd, >=3
    kernel = np.ones(win_frames, dtype=float) / win_frames
    env = np.convolve(rms, kernel, mode="same")

    # 4) Normalise 0–1 robustly
    e_min, e_max = env.min(), env.max()
    env = np.full_like(env, 0.5) if np.isclose(e_min, e_max) else (env - e_min) / (e_max - e_min)

    # 5) Resample to fixed number of slices (no dropping frames)
    # Reserve slices for both bottom and top pads if requested
    bottom_pad_slices = 0
    top_pad_slices = 0
    
    if add_bottom and pad_mm > 0:
        bottom_pad_slices = int(round((pad_mm / max(H, 1e-6)) * slices))
        bottom_pad_slices = max(1, min(bottom_pad_slices, max(1, slices // 3)))
    
    if add_top and pad_mm > 0:
        top_pad_slices = int(round((pad_mm / max(H, 1e-6)) * slices))
        top_pad_slices = max(1, min(top_pad_slices, max(1, slices // 3)))

    core_slices = max(2, slices - bottom_pad_slices - top_pad_slices)
    t_src = np.linspace(0.0, 1.0, len(env))
    t_dst = np.linspace(0.0, 1.0, core_slices)
    env_rs = np.interp(t_dst, t_src, env)

    # 6) Map to radius and build z
    r_core = R + VAR * (2*env_rs - 1.0)     # ≈ R±VAR for core
    
    # Build radius array with pads
    r_parts = []
    z_parts = []
    z_current = 0.0
    
    if bottom_pad_slices > 0:
        r_parts.append(np.full(bottom_pad_slices, R, dtype=float))
        z_parts.append(np.linspace(z_current, z_current + float(pad_mm), bottom_pad_slices, endpoint=False))
        z_current += float(pad_mm)
    
    r_parts.append(r_core)
    core_height = float(H) - z_current - (float(pad_mm) if top_pad_slices > 0 else 0.0)
    z_parts.append(np.linspace(z_current, z_current + core_height, core_slices, endpoint=(top_pad_slices == 0)))
    z_current += core_height
    
    if top_pad_slices > 0:
        # Use the last radius from core for top padding to prevent overhangs
        last_core_radius = r_core[-1]
        r_parts.append(np.full(top_pad_slices, last_core_radius, dtype=float))
        z_parts.append(np.linspace(z_current, float(H), top_pad_slices))
    
    r = np.concatenate(r_parts)
    z = np.concatenate(z_parts)
    
    # Prevent collapsed rings that can trigger repairs removing faces
    if min_radius is not None and min_radius > 0:
        r = np.maximum(r, float(min_radius))
    
    # Optional: clamp overhang slope (degrees from vertical). Lower = safer.
    # 45° is a conservative default for most materials without supports.
    overhang_deg = 45.0
    
    # Limit overhangs by clamping per-step slope |dr/dz| using actual dz per step
    if overhang_deg is not None:
        max_slope = float(np.tan(np.radians(float(overhang_deg))))  # max |dr/dz|
        r_clamped = r.copy()
        n = len(r_clamped)
        # Protect flat base and top pads from drifting
        lock_bottom = bottom_pad_slices
        lock_top_start = len(r_clamped) - top_pad_slices if top_pad_slices > 0 else len(r_clamped)
        passes = 4
        for _ in range(passes):
            # forward pass
            for i in range(1, n):
                if i < lock_bottom or i >= lock_top_start:
                    continue
                dz_i = float(z[i] - z[i-1])
                if dz_i <= 0:
                    continue
                dr_i = r_clamped[i] - r_clamped[i - 1]
                limit = max_slope * dz_i
                if dr_i > limit:
                    r_clamped[i] = r_clamped[i - 1] + limit
                elif dr_i < -limit:
                    r_clamped[i] = r_clamped[i - 1] - limit
            # backward pass
            for i in range(n - 2, -1, -1):
                if i < lock_bottom or i >= lock_top_start:
                    continue
                dz_i = float(z[i+1] - z[i])
                if dz_i <= 0:
                    continue
                dr_i = r_clamped[i] - r_clamped[i + 1]
                limit = max_slope * dz_i
                if dr_i > limit:
                    r_clamped[i] = r_clamped[i + 1] + limit
                elif dr_i < -limit:
                    r_clamped[i] = r_clamped[i + 1] - limit
            # re-lock pad segments to exact radius (baseline for bottom, last core for top)
            if lock_bottom > 0:
                r_clamped[:lock_bottom] = R
            if top_pad_slices > 0:
                r_clamped[lock_top_start:] = last_core_radius
        r = r_clamped

    make_hollow = (wall is not None and float(wall) > 0)
    if make_hollow:
        r_in = np.maximum(r - float(wall), float(min_radius) * 0.5)

    # 7) Surface of revolution (outer shell [+ inner if hollow])
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    X = np.outer(r, np.cos(theta))
    Y = np.outer(r, np.sin(theta))
    Z = np.outer(z, np.ones_like(theta))
    verts_outer = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    verts = verts_outer.copy()
    if make_hollow:
        Xi = np.outer(r_in, np.cos(theta))
        Yi = np.outer(r_in, np.sin(theta))
        Zi = Z
        verts_inner = np.c_[Xi.ravel(), Yi.ravel(), Zi.ravel()]
        verts = np.vstack([verts, verts_inner])

    faces = []
    nz = len(z)
    for i in range(nz - 1):
        a, b = i * n_theta, (i + 1) * n_theta
        for j in range(n_theta):
            k = (j + 1) % n_theta
            faces += [[a + j, a + k, b + k], [a + j, b + k, b + j]]
    # Inner surface (reversed) if hollow
    if make_hollow:
        offset = nz * n_theta
        for i in range(nz - 1):
            a, b = offset + i * n_theta, offset + (i + 1) * n_theta
            for j in range(n_theta):
                k = (j + 1) % n_theta
                faces += [[a + j, b + k, a + k], [a + j, b + j, b + k]]

    # 7b) Bottom closure - ALWAYS SOLID (never hollow)
    if add_bottom:
        center_idx = len(verts)
        verts = np.vstack([verts, np.array([[0.0, 0.0, 0.0]])])
        
        # Always fill bottom with solid disk from center to outer edge
        outer_start = 0
        for j in range(n_theta):
            k = (j + 1) % n_theta
            vj = outer_start + j
            vk = outer_start + k
            faces.append([center_idx, vk, vj])

    # 7c) Top closure - ALWAYS SOLID (never hollow) 
    if add_top:
        # Add center vertex at top
        top_center_idx = len(verts)
        top_center_vert = np.array([[0.0, 0.0, z[-1]]])
        verts = np.vstack([verts, top_center_vert])
        
        # Always fill top with solid disk from center to outer edge
        top_layer_start = (nz - 1) * n_theta
        for j in range(n_theta):
            k = (j + 1) % n_theta
            vj = top_layer_start + j
            vk = top_layer_start + k
            faces.append([top_center_idx, vj, vk])  # Reversed winding for top

    # Build mesh with process=False to avoid auto-simplification that can drop thin rings
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)

    # 8) Optional smoothing to soften sharp changes (surface fairing)
    # Protect the base pad/bottom ring (and top cap ring) from smoothing drift
    verts_before_smooth = mesh.vertices.copy()
    if smooth_mesh_iters and smooth_mesh_iters > 0:
        try:
            smoothing.filter_taubin(mesh, lamb=taubin_lambda, nu=taubin_nu, iterations=int(smooth_mesh_iters))
        except Exception:
            # Fallback to simple Laplacian if Taubin unavailable
            smoothing.filter_laplacian(mesh, lamb=0.5, iterations=int(smooth_mesh_iters))
        # Re-snap protected rings to their original planar positions
        tol = 1e-6
        if add_bottom and bottom_pad_slices > 0:
            lock_z_bottom = float(pad_mm)
            lock_mask_bottom = verts_before_smooth[:, 2] <= (lock_z_bottom + tol)
            mesh.vertices[lock_mask_bottom] = verts_before_smooth[lock_mask_bottom]
        if add_top and top_pad_slices > 0:
            lock_z_top = float(H) - float(pad_mm)
            lock_mask_top = verts_before_smooth[:, 2] >= (lock_z_top - tol)
            mesh.vertices[lock_mask_top] = verts_before_smooth[lock_mask_top]

    # 9) Repair for printability: merge verts, fix normals, fill holes if requested
    mesh.merge_vertices()
    # Fix normals; fall back if networkx (optional) is unavailable
    try:
        repair.fix_normals(mesh, multibody=True)
    except ModuleNotFoundError:
        # Fallback to per-face/vertex normal recompute without graph ops
        try:
            mesh.fix_normals()
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    # Keep a light touch to avoid removing geometry bands; only drop true degenerates
    # Keep only non-degenerate faces using recommended API
    mesh.update_faces(mesh.nondegenerate_faces())
    if add_top and not make_hollow:
        repair.fill_holes(mesh)
    # Drop duplicate faces using recommended API
    mesh.update_faces(mesh.unique_faces())
    # Validate without triggering networkx-dependent normal fixes
    try:
        mesh.process(validate=False)
    except ModuleNotFoundError:
        pass

    # Sanity: warn if ring count changed unexpectedly
    rings = 2 if make_hollow else 1
    expected_vertices_min = nz * n_theta * rings
    if add_bottom:
        expected_vertices_min += 1  # bottom center vertex
    if add_top:
        expected_vertices_min += 1  # top center vertex
    if len(mesh.vertices) < expected_vertices_min:
        print(f"Warning: vertex count lower than expected (got {len(mesh.vertices)}, expected ≥{expected_vertices_min}). Consider increasing min_radius.")
    mesh.export(output)  # binary STL by default
    
    # Build info string about pads
    pad_info = ""
    if bottom_pad_slices > 0 or top_pad_slices > 0:
        pad_parts = []
        if bottom_pad_slices > 0:
            pad_parts.append(f"bottom={pad_mm:.1f}mm")
        if top_pad_slices > 0:
            pad_parts.append(f"top={pad_mm:.1f}mm")
        pad_info = f", pads={'+'.join(pad_parts)}"
    
    print(f"Exported: {output}\n"
          f"Details: H≈{H:.0f} mm, Ø≈{2*np.max(r):.0f} mm, "
          f"slices={len(r)}, around={n_theta}, wall={'{:.1f}mm'.format(float(wall)) if make_hollow else 'single-shell'}{pad_info}")
    
    closure_info = []
    if add_bottom:
        closure_info.append("bottom closed")
    if add_top:
        closure_info.append("top closed")
    if not closure_info:
        closure_info.append("open design")
    
    print(f"Mode: {', '.join(closure_info)}")
    return output

if __name__ == "__main__":
    ultra_mvp_vase_resampled()
