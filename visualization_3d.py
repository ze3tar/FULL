"""High-quality 3D visualisation helpers for APF-RRT plots."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Needed for projection registration
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _as_array(nodes: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(nodes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Nodes must be a 2D array with at least 3 columns")
    return arr[:, :3]


def _clean_axes(ax: Axes3D) -> None:
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_box_aspect([1.0, 1.0, 1.0])


def plot_rrt_tree_3d(
    tree_nodes: Sequence[np.ndarray],
    parents: Mapping[int, Optional[int]],
    ax: Axes3D,
    *,
    scatter: bool = True,
) -> None:
    coords = _as_array(tree_nodes)
    segments = []
    for child_idx, parent_idx in parents.items():
        if parent_idx is None:
            continue
        if child_idx >= len(coords) or parent_idx >= len(coords):
            continue
        segments.append([coords[parent_idx], coords[child_idx]])
    if segments:
        collection = Line3DCollection(
            segments,
            colors="0.5",
            linewidths=1.0,
            alpha=0.25,
        )
        ax.add_collection3d(collection)
    if scatter:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=6, color="0.5", alpha=0.3)


def _cylinder_mesh(center: np.ndarray, radius: float, height: float, resolution: int = 24):
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(center[2] - height / 2.0, center[2] + height / 2.0, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    return x, y, z_grid


def _box_polys(center: np.ndarray, dims: Tuple[float, float, float]) -> Iterable[np.ndarray]:
    dx, dy, dz = [dim / 2.0 for dim in dims]
    corners = np.array(
        [
            [center[0] - dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] + dy, center[2] + dz],
            [center[0] - dx, center[1] + dy, center[2] + dz],
        ]
    )
    faces = [
        [corners[i] for i in [0, 1, 2, 3]],
        [corners[i] for i in [4, 5, 6, 7]],
        [corners[i] for i in [0, 1, 5, 4]],
        [corners[i] for i in [2, 3, 7, 6]],
        [corners[i] for i in [1, 2, 6, 5]],
        [corners[i] for i in [4, 7, 3, 0]],
    ]
    return faces


def plot_obstacles_3d(obstacles: Sequence[object], ax: Axes3D) -> None:
    for obstacle in obstacles:
        centre = None
        radius: Optional[float] = None
        dims: Optional[Tuple[float, float, float]] = None
        if hasattr(obstacle, "centre") and hasattr(obstacle, "radius"):
            centre = np.asarray(obstacle.centre, dtype=np.float64)[:3]
            radius = float(getattr(obstacle, "radius"))
        elif isinstance(obstacle, (tuple, list)) and len(obstacle) == 2:
            centre = np.asarray(obstacle[0], dtype=np.float64)[:3]
            radius = float(obstacle[1])
        elif isinstance(obstacle, Mapping):
            centre = np.asarray(obstacle.get("centre", obstacle.get("center", (0, 0, 0))), dtype=np.float64)[:3]
            if obstacle.get("type") == "rectangle":
                dims = tuple(obstacle.get("dims", obstacle.get("size", (1.0, 1.0, 1.0))))  # type: ignore[arg-type]
            else:
                radius = float(obstacle.get("radius", 0.5))
        if centre is None:
            continue

        face_color = obstacle.get("color", "red") if isinstance(obstacle, Mapping) else "red"
        alpha = 0.4
        if dims is not None:
            polys = _box_polys(centre, dims)
            box = Poly3DCollection(
                polys,
                facecolors=face_color,
                linewidths=0.8,
                edgecolors="0.4",
                alpha=alpha,
            )
            box.set_facecolor(face_color)
            box.set_alpha(alpha)
            ax.add_collection3d(box)
            continue

        height = radius * 2.0 if radius is not None else 1.0
        x, y, z = _cylinder_mesh(centre, radius or 0.5, height)
        ax.plot_surface(x, y, z, shade=True, color=face_color, alpha=alpha, linewidth=0)
        ax.plot_surface(x, y, np.full_like(x, centre[2] + height / 2.0), color=face_color, alpha=alpha)
        ax.plot_surface(x, y, np.full_like(x, centre[2] - height / 2.0), color=face_color, alpha=alpha)


def plot_path_3d(path: Sequence[np.ndarray], ax: Axes3D) -> None:
    coords = _as_array(path)
    segments = [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]
    if not segments:
        return

    diffs = np.diff(coords, axis=0)
    # Curvature approximation using discrete second derivative
    curvatures = np.zeros(len(coords))
    if len(coords) > 2:
        for i in range(1, len(coords) - 1):
            v1 = diffs[i - 1]
            v2 = diffs[i]
            denom = (np.linalg.norm(v1) ** 3) + 1e-9
            curvatures[i] = np.linalg.norm(np.cross(v1, v2)) / denom
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
    norm_curvature = (curvatures - curvatures.min()) / (curvatures.ptp() + 1e-9)

    lc = Line3DCollection(
        segments,
        cmap=cm.get_cmap("plasma"),
        linewidths=3.5,
    )
    lc.set_array(norm_curvature[:-1])
    lc.set_alpha(0.95)
    ax.add_collection3d(lc)

    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color="black",
        linewidth=1.0,
        alpha=0.6,
    )


def plot_apf_field_3d(
    field_fn,
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ax: Axes3D,
    *,
    density: int = 6,
    use_contours: bool = False,
) -> None:
    x_range, y_range, z_range = domain
    xs = np.linspace(*x_range, density)
    ys = np.linspace(*y_range, density)
    zs = np.linspace(*z_range, density)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    pos = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    vectors = np.apply_along_axis(field_fn, 1, pos)
    if vectors.shape[1] > 3:
        vectors = vectors[:, :3]
    U, V, W = vectors.T
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], U, V, W, length=0.2, normalize=True, color="tab:blue", alpha=0.45)

    if use_contours:
        midpoint = 0.5 * (z_range[0] + z_range[1])
        Xp, Yp = np.meshgrid(xs, ys)
        sample_plane = np.stack([Xp, Yp, np.full_like(Xp, midpoint)], axis=-1).reshape(-1, 3)
        plane_vectors = np.apply_along_axis(field_fn, 1, sample_plane)
        potential = np.linalg.norm(plane_vectors[:, :3], axis=1).reshape(Xp.shape)
        ax.contourf(
            Xp,
            Yp,
            np.full_like(Xp, midpoint),
            potential,
            zdir="z",
            offset=midpoint,
            cmap="Blues",
            alpha=0.35,
        )

    _clean_axes(ax)

