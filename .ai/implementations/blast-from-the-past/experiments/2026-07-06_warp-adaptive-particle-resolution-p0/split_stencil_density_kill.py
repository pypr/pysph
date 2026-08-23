#!/usr/bin/env python3
"""Reproduce the Vacondio 3D split-stencil density minimization.

This is deliberately NumPy-only. It isolates the scientific stencil and mass
weights before any GPU allocation or Warp kernel design is allowed to depend on
them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wendland_c2_3d(r, h):
    """PySPH/Warp WendlandQuintic: support ``r/h < 2``."""
    q = r / h
    out = np.zeros_like(q)
    mask = q < 2.0
    tmp = 1.0 - 0.5 * q[mask]
    out[mask] = (
        21.0 / (16.0 * np.pi * h**3) * tmp**4 * (2.0 * q[mask] + 1.0)
    )
    return out


def cubic_vertices():
    vertices = np.asarray([
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ])
    return vertices / np.linalg.norm(vertices, axis=1)[:, None]


def icosahedron_vertices():
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            vertices.extend(((0.0, a, b), (a, b, 0.0), (b, 0.0, a)))
    vertices = np.unique(np.asarray(vertices), axis=0)
    return vertices / np.linalg.norm(vertices, axis=1)[:, None]


def integrate_stencil(vertices, epsilon, alpha, ngrid):
    """Solve the symmetry-reduced, mass-constrained least-squares problem."""
    limit = 2.0 + epsilon + 0.05
    axis = np.linspace(-limit, limit, ngrid)
    spacing = axis[1] - axis[0]
    nvertices = len(vertices)

    numerator = 0.0
    denominator = 0.0
    cached = []
    for z in axis:
        x, y = np.meshgrid(axis, axis, indexing='ij')
        points = np.stack((x, y, np.full_like(x, z)), axis=-1)
        parent = wendland_c2_3d(np.linalg.norm(points, axis=-1), 1.0)
        center = wendland_c2_3d(np.linalg.norm(points, axis=-1), alpha)
        shell = np.zeros_like(parent)
        for vertex in vertices:
            shell += wendland_c2_3d(
                np.linalg.norm(points - epsilon * vertex, axis=-1), alpha
            )
        direction = shell - nvertices * center
        numerator += np.sum((parent - center) * direction)
        denominator += np.sum(direction * direction)
        cached.append((parent, center, shell))

    vertex_mass = numerator / denominator
    center_mass = 1.0 - nvertices * vertex_mass
    error = 0.0
    equal_error = 0.0
    equal_mass = 1.0 / (nvertices + 1)
    for parent, center, shell in cached:
        reconstruction = vertex_mass * shell + center_mass * center
        equal_reconstruction = equal_mass * (shell + center)
        error += np.sum((parent - reconstruction)**2)
        equal_error += np.sum((parent - equal_reconstruction)**2)
    error *= spacing**3
    equal_error *= spacing**3

    return {
        'vertices': nvertices,
        'daughters_with_center': nvertices + 1,
        'epsilon': epsilon,
        'alpha': alpha,
        'vertex_mass_fraction': float(vertex_mass),
        'center_mass_fraction': float(center_mass),
        'mass_sum': float(nvertices * vertex_mass + center_mass),
        'min_max_mass_ratio': float(
            min(vertex_mass, center_mass) / max(vertex_mass, center_mass)
        ),
        'integrated_density_error': float(error),
        'equal_mass_integrated_density_error': float(equal_error),
        'grid_points_per_axis': ngrid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ngrid', type=int, default=101)
    parser.add_argument('--epsilon', type=float, default=0.65)
    parser.add_argument('--alpha', type=float, default=0.70)
    parser.add_argument('--output')
    args = parser.parse_args()

    result = {
        'kernel': 'PySPH WendlandQuintic C2 3D',
        'paper_icosahedron_reference': {
            'integrated_density_error': 8.326e-5,
            'min_max_mass_ratio': 0.33,
        },
        'cubic_plus_center': integrate_stencil(
            cubic_vertices(), args.epsilon, args.alpha, args.ngrid
        ),
        'icosahedron_plus_center': integrate_stencil(
            icosahedron_vertices(), args.epsilon, args.alpha, args.ngrid
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
