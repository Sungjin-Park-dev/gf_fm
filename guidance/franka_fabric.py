# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from guidance.body_sphere_repulsion import BodySphereRepulsion
from guidance.body_sphere_repulsion import BaseFabricRepulsion
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path


class FrankaPandaRepulsionFabric(BaseFabric):
    """
    Repulsion-only fabric for Franka Panda.
    Includes joint-limit repulsion and body-sphere repulsion against world meshes.
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True, use_cspace_metric=True):
        fabric_params_filename = "franka_panda_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        robot_dir_name = "franka_panda"
        robot_name = "franka_panda_sim"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.load_robot(robot_dir_name, robot_name, batch_size)

        self.use_cspace_metric = use_cspace_metric
        self.construct_fabric()

    def add_joint_limit_repulsion(self):
        joints = self.urdfpy_robot.joints
        upper_joint_limits = []
        for joint in joints:
            if joint.joint_type == "revolute":
                upper_joint_limits.append(joint.limit.upper)
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params["joint_limit_repulsion"],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

        lower_joint_limits = []
        for joint in joints:
            if joint.joint_type == "revolute":
                lower_joint_limits.append(joint.limit.lower)
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params["joint_limit_repulsion"],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

    def add_body_repulsion(self):
        collision_sphere_frames = self.fabric_params["body_repulsion"]["collision_sphere_frames"]
        self.collision_sphere_radii = self.fabric_params["body_repulsion"]["collision_sphere_radii"]

        collision_sphere_pairs = self.fabric_params["body_repulsion"]["collision_sphere_pairs"]
        collision_matrix = torch.zeros(len(collision_sphere_frames), len(collision_sphere_frames),
                                       dtype=int, device=self.device)
        if len(collision_sphere_pairs) > 0:
            collision_link_prefix_pairs = self.fabric_params["body_repulsion"]["collision_link_prefix_pairs"]
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]
                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])
            for sphere1, sphere2 in collision_sphere_pairs:
                collision_matrix[collision_sphere_frames.index(sphere1),
                                 collision_sphere_frames.index(sphere2)] = 1

        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)
        fabric = BodySphereRepulsion(True, self.fabric_params["body_repulsion"],
                                     self.batch_size, sphere_radius, collision_matrix,
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "repulsion", fabric)

        fabric_geom = BodySphereRepulsion(False, self.fabric_params["body_repulsion"],
                                          self.batch_size, sphere_radius, collision_matrix,
                                          self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        self.base_fabric_repulsion = BaseFabricRepulsion(self.fabric_params["body_repulsion"],
                                                         self.batch_size,
                                                         sphere_radius,
                                                         collision_matrix,
                                                         self.device)

    def add_cspace_metric(self):
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = Attractor(False, self.fabric_params["cspace_attractor"],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "cspace_metric", fabric)

    def construct_fabric(self):
        self.add_joint_limit_repulsion()
        self.add_body_repulsion()
        if self.use_cspace_metric:
            self.add_cspace_metric()

    def set_features(self, batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator, cspace_damping_gain=None,
                     cspace_target=None):
        if self.use_cspace_metric:
            if cspace_target is None:
                self.fabrics_features["identity"]["cspace_metric"] = batched_cspace_position
            else:
                self.fabrics_features["identity"]["cspace_metric"] = cspace_target

        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)
        self.base_fabric_repulsion.calculate_response(body_point_pos,
                                                      body_point_vel,
                                                      object_ids,
                                                      object_indicator)

        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params["cspace_damping"]["gain"] = cspace_damping_gain

    def get_sphere_radii(self):
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        return self.base_fabric_repulsion.collision_status
