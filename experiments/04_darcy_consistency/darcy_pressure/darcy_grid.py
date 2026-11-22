"""Darcy on the released grid, with pressure reconstruction and flux diagnostics."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

SIZE = 128
INTERIOR = SIZE - 2
H = 1.0 / (SIZE - 1)


def interpolation_matrix(source, target):
    matrix = np.zeros((SIZE, SIZE))
    bounds = np.array([0.0, 1.0])
    basis = np.eye(SIZE)
    for index in range(SIZE):
        spline = RectBivariateSpline(
            source, bounds, np.repeat(basis[:, index, None], 2, axis=1), kx=3, ky=1
        )
        matrix[:, index] = spline(target, np.array([0.5])).ravel()
    return matrix


CENTERS = (np.arange(SIZE) + 0.5) / SIZE
NODES = np.linspace(0.0, 1.0, SIZE)
TO_NODES = interpolation_matrix(CENTERS, NODES)
TO_CENTERS = interpolation_matrix(NODES, CENTERS)[:, 1:-1]
RECOVER_INTERIOR = np.linalg.pinv(TO_CENTERS)
FORCING = (TO_NODES @ np.ones((SIZE, SIZE)) @ TO_NODES.T)[1:-1, 1:-1]


def face_coefficients(permeability):
    """Interpolate permeability to nodes and average it on their connecting faces."""
    nodes = np.clip(TO_NODES @ permeability @ TO_NODES.T, 3.0, 12.0)
    along_rows = (nodes[1:, 1:-1] + nodes[:-1, 1:-1]) * 0.5
    along_columns = (nodes[1:-1, 1:] + nodes[1:-1, :-1]) * 0.5
    return along_rows, along_columns


def divergence(pressure, faces):
    """Discrete net outward flux per unit area on each interior control volume."""
    along_rows, along_columns = faces
    padded = np.zeros((*pressure.shape[:-2], SIZE, SIZE))
    padded[..., 1:-1, 1:-1] = pressure
    row_flux = -along_rows * np.diff(padded[..., :, 1:-1], axis=-2) / H
    column_flux = -along_columns * np.diff(padded[..., 1:-1, :], axis=-1) / H
    return (np.diff(row_flux, axis=-2) + np.diff(column_flux, axis=-1)) / H


def solve(faces):
    """Solve the same SPD matrix as the released CG solver by sparse direct solve."""
    along_rows, along_columns = faces
    diag = (along_rows[:-1] + along_rows[1:] + along_columns[:, :-1] + along_columns[:, 1:])
    row_edges = -along_rows[1:-1].ravel()
    # The zero at each row end prevents wraparound to the next row.
    column_edges = np.pad(-along_columns[:, 1:-1], ((0, 0), (0, 1))).ravel()[:-1]
    matrix = diags(
        [row_edges, column_edges, diag.ravel(), column_edges, row_edges],
        [-INTERIOR, -1, 0, 1, INTERIOR],
        format="csc",
    )
    pressure = spsolve(matrix, (H**2 * FORCING).ravel(), permc_spec="MMD_AT_PLUS_A")
    pressure = pressure.reshape(INTERIOR, INTERIOR)
    residual = float(np.max(np.abs(divergence(pressure, faces) - FORCING)))
    if not np.isfinite(residual) or residual > 1e-7:
        raise ArithmeticError(f"Darcy sparse solve did not satisfy the stencil: {residual:g}")
    return TO_CENTERS @ pressure @ TO_CENTERS.T


def recover_pressure(centers):
    return RECOVER_INTERIOR @ centers @ RECOVER_INTERIOR.T


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def patch_rms(residual, sizes):
    """RMS fractional flux imbalance over every overlapping interior square."""
    cumulative = np.pad(residual.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    result = {}
    for side in sizes:
        patch_sum = (
            cumulative[side:, side:] - cumulative[:-side, side:]
            - cumulative[side:, :-side] + cumulative[:-side, :-side]
        )
        result[side] = rms(patch_sum / side**2)
    return result
