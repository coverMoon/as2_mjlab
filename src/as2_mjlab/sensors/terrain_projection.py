"""Terrain surface queries with foot-referenced clearance visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.sensor import TerrainHeightSensor, TerrainHeightSensorCfg

if TYPE_CHECKING:
    import mujoco_warp as mjwarp
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def terrain_height_bounds(
    model: mujoco.MjModel, groups: tuple[int, ...] | None
) -> tuple[float, float]:
    """World Z bounds of static terrain, excluding dynamic robot geometry.

    Finite geometry uses MuJoCo's local AABB transformed to world coordinates.
    Horizontal planes contribute their surface height, not their infinite AABB.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        if model.body_weldid[body_id] != 0 or model.body_mocapid[body_id] >= 0:
            continue
        if groups is not None and model.geom_group[geom_id] not in groups:
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        position = data.geom_xpos[geom_id]
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            if not np.allclose(rotation[:2, 2], 0.0, atol=1e-7):
                raise ValueError("Terrain height projection requires horizontal infinite planes")
            bounds.append((position[2], position[2]))
        else:
            center, extent = np.split(model.geom_aabb[geom_id], 2)
            world_center = position + rotation @ center
            half_height = np.abs(rotation[2]) @ extent
            bounds.append((world_center[2] - half_height, world_center[2] + half_height))
    if not bounds:
        raise ValueError("No static terrain geometry matches include_geom_groups")
    return float(min(lo for lo, _ in bounds)), float(max(hi for _, hi in bounds))


@dataclass
class TerrainProjectionSensorCfg(TerrainHeightSensorCfg):
    """Query ground Z from a scene-derived plane, while measuring from the foot.

    max_distance is calculated at initialization from the terrain's Z bounds.
    viz.show_rays draws signed foot-to-ground clearance, not calculation rays.
    """

    query_margin: float = 0.1
    """Clearance above/below terrain bounds for the internal vertical query."""

    def build(self) -> TerrainProjectionSensor:
        return TerrainProjectionSensor(self)


class TerrainProjectionSensor(TerrainHeightSensor):
    cfg: TerrainProjectionSensorCfg

    def initialize(
        self, mj_model: mujoco.MjModel, model: mjwarp.Model,
        data: mjwarp.Data, device: str,
    ) -> None:
        if self.cfg.query_margin <= 0.0:
            raise ValueError("query_margin must be positive")
        if self.cfg.ray_alignment not in ("yaw", "world"):
            raise ValueError("Terrain projection requires yaw/world alignment")
        low, high = terrain_height_bounds(mj_model, self.cfg.include_geom_groups)
        self.query_plane_z = high + self.cfg.query_margin
        self.cfg.max_distance = high - low + 2.0 * self.cfg.query_margin
        super().initialize(mj_model, model, data, device)
        assert self._local_offsets is not None and self._local_directions is not None
        expected = torch.tensor([0., 0., -1.], device=device).expand_as(self._local_directions)
        if not torch.allclose(self._local_directions, expected):
            raise ValueError("Terrain projection requires downward vertical rays")
        if not torch.allclose(self._local_offsets[:, 2], torch.zeros_like(self._local_offsets[:, 2])):
            raise ValueError("Terrain projection sample offsets must have zero Z")

    def prepare_rays(self) -> None:
        # Reuse upstream frame transforms, retaining physical site poses for
        # heights = site_z - hit_z. Only the internal query origins use the plane.
        super().prepare_rays()
        assert self._cached_world_origins is not None and self._ray_pnt is not None
        self._cached_world_origins[..., 2] = self.query_plane_z
        wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self.cfg.debug_vis or not self._debug_vis_enabled:
            return
        if self._cached_world_origins is None or self._cached_frame_pos is None:
            return
        data = self.data
        indices = list(visualizer.get_env_indices(data.distances.shape[0]))
        if not indices:
            return
        starts = self._cached_world_origins[indices].view(
            len(indices), self.num_frames, self.num_rays_per_frame, 3
        ).clone()
        starts[..., 2] = self._cached_frame_pos[indices, :, 2].unsqueeze(-1)
        starts = starts.flatten(1, 2).cpu().numpy()
        hits = data.hit_pos_w[indices].cpu().numpy()
        valid = (data.distances[indices] >= 0).cpu().numpy()
        normals = data.normals_w[indices].cpu().numpy()
        radius = self.cfg.viz.hit_sphere_radius * visualizer.meansize
        width = 0.03 * visualizer.meansize
        for row, env_id in enumerate(indices):
            for ray_id in range(self.num_frames * self.num_rays_per_frame):
                label = f"{self.cfg.name}_{env_id}_{ray_id}"
                start, end = starts[row, ray_id], hits[row, ray_id]
                if not valid[row, ray_id]:
                    # No fabricated ground marker when the query misses terrain.
                    visualizer.add_sphere(center=start, radius=radius,
                                          color=self.cfg.viz.miss_color, label=label + "_miss")
                    continue
                negative = end[2] > start[2]
                color = (1.0, 0.3, 0.0, 0.9) if negative else self.cfg.viz.hit_color
                if self.cfg.viz.show_rays and np.linalg.norm(end - start) > 1e-6:
                    visualizer.add_arrow(start=start, end=end, color=color,
                                         width=width, label=label + "_clearance")
                visualizer.add_sphere(center=end, radius=radius,
                                      color=color if negative else self.cfg.viz.hit_sphere_color,
                                      label=label + "_hit")
                if self.cfg.viz.show_normals:
                    visualizer.add_arrow(
                        start=end, end=end + normals[row, ray_id] * self.cfg.viz.normal_length * visualizer.meansize,
                        color=self.cfg.viz.normal_color, width=width, label=label + "_normal",
                    )
