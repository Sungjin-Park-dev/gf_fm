# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sphere obstacle world model for GF guidance.

Creates Warp meshes from sphere parameters for FABRICS collision detection.
No URDF files required - generates icosphere geometry directly.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

try:
    import warp as wp
except ImportError:
    wp = None


def generate_icosphere(subdivisions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate unit icosphere vertices and faces.

    Creates a sphere approximation using recursive subdivision of an icosahedron.

    Args:
        subdivisions: Number of subdivision iterations.
            0 = icosahedron (12 vertices, 20 faces)
            1 = 42 vertices, 80 faces
            2 = 162 vertices, 320 faces (recommended)

    Returns:
        vertices: (V, 3) numpy array of vertex positions (unit sphere)
        faces: (F, 3) numpy array of face indices (triangles)
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Initial icosahedron vertices (12 vertices)
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)

    # Normalize to unit sphere
    vertices /= np.linalg.norm(vertices[0])

    # Initial faces (20 triangles)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)

    return vertices, faces


def _subdivide_icosphere(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide icosphere by splitting each triangle into 4.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices

    Returns:
        new_vertices: (V', 3) subdivided vertices
        new_faces: (F*4, 3) subdivided faces
    """
    # Edge midpoint cache to avoid duplicate vertices
    edge_midpoints = {}
    new_vertices = list(vertices)

    def get_midpoint(i1: int, i2: int) -> int:
        """Get or create midpoint vertex between two vertices."""
        edge = (min(i1, i2), max(i1, i2))
        if edge in edge_midpoints:
            return edge_midpoints[edge]

        # Create new vertex at midpoint
        v1, v2 = vertices[i1], vertices[i2]
        mid = (v1 + v2) / 2
        mid = mid / np.linalg.norm(mid)  # Project to unit sphere

        idx = len(new_vertices)
        new_vertices.append(mid)
        edge_midpoints[edge] = idx
        return idx

    new_faces = []
    for v0, v1, v2 in faces:
        # Get midpoints
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Create 4 new triangles
        new_faces.append([v0, m01, m20])
        new_faces.append([v1, m12, m01])
        new_faces.append([v2, m20, m12])
        new_faces.append([m01, m12, m20])

    return np.array(new_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)


class SphereWorldModel:
    """Lightweight world model for sphere obstacles.

    Creates Warp meshes from sphere parameters (position, radius) and
    provides mesh IDs for FABRICS collision detection.

    Args:
        batch_size: Number of parallel environments
        max_spheres: Maximum number of sphere obstacles
        device: Torch/Warp device string
    """

    def __init__(
        self,
        batch_size: int,
        max_spheres: int = 20,
        device: str = 'cuda:0',
    ):
        if wp is None:
            raise ImportError("Warp is required for SphereWorldModel")

        self.batch_size = batch_size
        self.max_spheres = max_spheres
        self.device = device

        # Parse device for warp
        self._wp_device = device if 'cuda' in device else 'cpu'

        # Pre-generate unit sphere mesh
        self._unit_vertices, self._unit_faces = generate_icosphere(subdivisions=2)

        # Storage for sphere meshes
        self._meshes: List[wp.Mesh] = []
        self._sphere_count = 0

        # Warp arrays for mesh IDs and indicators
        # Shape: (batch_size, max_spheres)
        self._mesh_ids = wp.zeros(
            (batch_size, max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )
        self._mesh_indicator = wp.zeros(
            (batch_size, max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )

    @property
    def sphere_count(self) -> int:
        """Number of active sphere obstacles."""
        return self._sphere_count

    def add_sphere(
        self,
        position: Tuple[float, float, float],
        radius: float,
        env_index: str = 'all'
    ) -> int:
        """Add a sphere obstacle.

        Args:
            position: (x, y, z) center of sphere in world coordinates
            radius: sphere radius in meters
            env_index: 'all' to apply to all environments, or int index

        Returns:
            Index of added sphere

        Raises:
            RuntimeError: If max_spheres limit reached
        """
        if self._sphere_count >= self.max_spheres:
            raise RuntimeError(f"Maximum sphere limit ({self.max_spheres}) reached")

        # Scale and translate unit sphere
        vertices = self._unit_vertices.copy() * radius + np.array(position, dtype=np.float32)

        # Flatten faces for warp
        faces_flat = self._unit_faces.flatten().astype(np.int32)

        # Create Warp mesh
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self._wp_device),
            indices=wp.array(faces_flat, dtype=wp.int32, device=self._wp_device)
        )
        mesh.refit()

        self._meshes.append(mesh)
        idx = self._sphere_count

        # Update mesh IDs and indicators
        # Convert to numpy for modification, then back to warp
        mesh_ids_np = self._mesh_ids.numpy()
        indicator_np = self._mesh_indicator.numpy()

        if env_index == 'all':
            for b in range(self.batch_size):
                mesh_ids_np[b, idx] = mesh.id
                indicator_np[b, idx] = 1
        else:
            b = int(env_index)
            mesh_ids_np[b, idx] = mesh.id
            indicator_np[b, idx] = 1

        # Copy back to warp
        self._mesh_ids = wp.array(mesh_ids_np, dtype=wp.uint64, device=self._wp_device)
        self._mesh_indicator = wp.array(indicator_np, dtype=wp.uint64, device=self._wp_device)

        self._sphere_count += 1
        return idx

    def add_spheres(
        self,
        spheres: List[Tuple[Tuple[float, float, float], float]]
    ) -> List[int]:
        """Add multiple sphere obstacles.

        Args:
            spheres: List of ((x, y, z), radius) tuples

        Returns:
            List of sphere indices
        """
        indices = []
        for pos, radius in spheres:
            idx = self.add_sphere(pos, radius)
            indices.append(idx)
        return indices

    def clear(self):
        """Remove all sphere obstacles."""
        self._meshes.clear()
        self._sphere_count = 0

        # Reset arrays
        self._mesh_ids = wp.zeros(
            (self.batch_size, self.max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )
        self._mesh_indicator = wp.zeros(
            (self.batch_size, self.max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )

    def update_batch_size(self, new_batch_size: int):
        """Update batch size and resize Warp arrays.

        Args:
            new_batch_size: New batch size
        """
        if new_batch_size == self.batch_size:
            return

        self.batch_size = new_batch_size

        # Re-allocate arrays
        self._mesh_ids = wp.zeros(
            (self.batch_size, self.max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )
        self._mesh_indicator = wp.zeros(
            (self.batch_size, self.max_spheres),
            dtype=wp.uint64,
            device=self._wp_device
        )

        # Re-populate from existing meshes
        if self._sphere_count > 0:
            mesh_ids_np = self._mesh_ids.numpy()
            indicator_np = self._mesh_indicator.numpy()

            for idx, mesh in enumerate(self._meshes):
                # Apply to all batch indices
                mesh_ids_np[:, idx] = mesh.id
                indicator_np[:, idx] = 1

            # Copy back to warp
            self._mesh_ids = wp.array(mesh_ids_np, dtype=wp.uint64, device=self._wp_device)
            self._mesh_indicator = wp.array(indicator_np, dtype=wp.uint64, device=self._wp_device)

    def get_object_ids(self) -> Tuple:
        """Get mesh IDs and indicators for FABRICS collision detection.

        Returns:
            (object_ids, object_indicator): Warp arrays of shape (batch_size, max_spheres)
        """
        return self._mesh_ids, self._mesh_indicator

    def get_sphere_info(self, index: int) -> Optional[Tuple[np.ndarray, float]]:
        """Get position and radius of a sphere.

        Args:
            index: Sphere index

        Returns:
            (center, radius) or None if index invalid
        """
        if index < 0 or index >= self._sphere_count:
            return None

        mesh = self._meshes[index]
        points = mesh.points.numpy()

        # Estimate center and radius from mesh points
        center = points.mean(axis=0)
        radius = np.linalg.norm(points - center, axis=1).mean()

        return center, radius

    def __len__(self) -> int:
        """Number of active spheres."""
        return self._sphere_count

    def __repr__(self) -> str:
        return f"SphereWorldModel(batch_size={self.batch_size}, spheres={self._sphere_count}/{self.max_spheres})"
