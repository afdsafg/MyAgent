import os.path
import logging
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats
import torch
import habitat_sim
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
import supervision as sv
from matplotlib.patches import Wedge
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from src.geom import *
from src.habitat import pos_normal_to_habitat, pos_habitat_to_normal
from src.tsdf_base import TSDFPlannerBase
from src.conceptgraph.slam.slam_classes import MapObjectDict
from src.utils import resize_image


@dataclass
class RoomRegion:
    """Geometric room region segmented from the current TSDF occupancy map."""
    room_id: int
    center: np.ndarray
    region: np.ndarray
    area: int
    room_state: str = "unknown"
    observed_ratio: float = 0.0
    hypothesis_node_id: Optional[str] = None
    frontier_ids: List[int] = field(default_factory=list)


@dataclass
class Frontier:
    """Frontier class for frontier-based exploration."""

    position: np.ndarray  # integer position in voxel grid
    orientation: np.ndarray  # directional vector of the frontier in float
    region: (
        np.ndarray
    )  # boolean array of the same shape as the voxel grid, indicating the region of the frontier
    frontier_id: (
        int  # unique id for the frontier to identify its region on the frontier map
    )
    image: str = None
    target_detected: bool = (
        False  # whether the target object is detected in the snapshot, only used when generating data
    )
    feature: torch.Tensor = (
        None  # the image feature of the snapshot, not used when generating data
    )
    room_id: int = -1
    room_state: str = ""

    def __eq__(self, other):
        if not isinstance(other, Frontier):
            raise TypeError("Cannot compare Frontier with non-Frontier object.")
        return np.array_equal(self.region, other.region)


@dataclass
class SceneGraphItem:
    object_id: int
    bbox_center: np.ndarray
    confidence: float
    image: str


@dataclass
class SnapShot:
    image: str
    color: Tuple[float, float, float]
    obs_point: np.ndarray  # integer position in voxel grid
    full_obj_list: Dict[int, float] = field(
        default_factory=dict
    )  # object id to confidence
    cluster: List[int] = field(default_factory=list)
    position: np.ndarray = None
    room_id: int = -1

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare SnapShot objects.")


class TSDFPlanner(TSDFPlannerBase):
    """Volumetric TSDF Fusion of RGB-D Images. No GPU mode.

    Add frontier-based exploration and semantic map.
    """

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        floor_height,
        floor_height_offset=0,
        pts_init=None,
        init_clearance=0,
        occupancy_height=0.4,
        vision_height=1.2,
        save_visualization=False,
    ):
        super().__init__(
            vol_bnds,
            voxel_size,
            floor_height,
            floor_height_offset,
            pts_init,
            init_clearance,
            save_visualization,
        )

        self.occupancy_height = (
            occupancy_height  # the occupied/navigable map is acquired at this height
        )
        self.vision_height = (
            vision_height  # the visibility map is acquired at this height
        )

        # about navigation
        self.max_point: [Frontier, SnapShot] = (
            None  # the frontier/snapshot the agent chooses
        )
        self.target_point: [np.ndarray] = (
            None  # the corresponding navigable location of max_point. The agent goes to this point to observe the max_point
        )

        self.frontiers: List[Frontier] = []

        # about frontier allocation
        self.frontier_map = np.zeros(self._vol_dim[:2], dtype=int)
        self.frontier_counter = 1

        # Room segmentation state
        self.room_map = np.zeros(self._vol_dim[:2], dtype=int)
        self.room_regions: List[RoomRegion] = []
        self.room_counter = 1

        # about storing occupancy information on each step
        self.unexplored = None
        self.unoccupied = None
        self.occupied = None
        self.island = None
        self.unexplored_neighbors = None
        self.occupied_map_camera = None

    def update_frontier_map(
        self,
        pts,
        cfg,
        scene,
        cnt_step: int,
        save_frontier_image: bool = False,
        eps_frontier_dir=None,
        prompt_img_size: Tuple[int, int] = (320, 320),
    ) -> bool:
        pts_habitat = pts.copy()
        pts = pos_habitat_to_normal(pts)
        cur_point = self.normal2voxel(pts)

        island, unoccupied = self.get_island_around_pts(
            pts, height=self.occupancy_height
        )
        occupied = np.logical_not(unoccupied).astype(int)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        occupied_map_camera = np.logical_not(
            self.get_island_around_pts(pts, height=self.vision_height)[0]
        )
        self.unexplored = unexplored
        self.unoccupied = unoccupied
        self.occupied = occupied
        self.island = island
        self.unexplored_neighbors = unexplored_neighbors
        self.occupied_map_camera = occupied_map_camera

        # detect and update frontiers
        frontier_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_area_min)
            & (unexplored_neighbors <= cfg.frontier_area_max)
        )
        frontier_edge_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_edge_area_min)
            & (unexplored_neighbors <= cfg.frontier_edge_area_max)
        )

        if len(frontier_areas) == 0:
            # this happens when there are stairs on the floor, and the planner cannot handle this situation
            # just skip this question
            logging.error(f"Error in update_frontier_map: frontier area size is 0")
            self.frontiers = []
            return False
        if len(frontier_edge_areas) == 0:
            # this happens rather rarely
            logging.error(f"Error in update_frontier_map: frontier edge area size is 0")
            self.frontiers = []
            return False

        # cluster frontier regions
        db = DBSCAN(eps=cfg.eps, min_samples=2).fit(frontier_areas)
        labels = db.labels_
        # get one point from each cluster
        valid_ft_angles = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = frontier_areas[labels == label]

            # filter out small frontiers
            area = len(cluster)
            if area < cfg.min_frontier_area:
                continue

            # convert the cluster from voxel coordinates to polar angle coordinates
            angle_cluster = np.asarray(
                [
                    np.arctan2(
                        cluster[i, 1] - cur_point[1], cluster[i, 0] - cur_point[0]
                    )
                    for i in range(len(cluster))
                ]
            )  # range from -pi to pi

            # get the range of the angles
            angle_range = get_angle_span(angle_cluster)
            warping_gap = get_warping_gap(
                angle_cluster
            )  # add 2pi to angles that smaller than this to avoid angles crossing -pi/pi line
            if warping_gap is not None:
                angle_cluster[angle_cluster < warping_gap] += 2 * np.pi

            if angle_range > cfg.max_frontier_angle_range_deg * np.pi / 180:
                # cluster again on the angle, ie, split the frontier
                num_clusters = (
                    int(angle_range / (cfg.max_frontier_angle_range_deg * np.pi / 180))
                    + 1
                )
                db_angle = KMeans(n_clusters=num_clusters).fit(angle_cluster[..., None])
                labels_angle = db_angle.labels_
                for label_angle in np.unique(labels_angle):
                    if label_angle == -1:
                        continue
                    ft_angle = np.mean(angle_cluster[labels_angle == label_angle])
                    valid_ft_angles.append(
                        {
                            "angle": (
                                ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle
                            ),
                            "region": self.get_frontier_region_map(
                                cluster[labels_angle == label_angle]
                            ),
                        }
                    )
            else:
                ft_angle = np.mean(angle_cluster)
                valid_ft_angles.append(
                    {
                        "angle": ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle,
                        "region": self.get_frontier_region_map(cluster),
                    }
                )

        # remove frontiers that have been changed
        filtered_frontiers = []
        kept_frontier_area = np.zeros_like(self.frontier_map, dtype=bool)
        scale_factor = int(
            (0.1 / self._voxel_size) ** 2
        )  # when counting the number of pixels in the frontier region, we use a default voxel length of 0.1m. Then other voxel lengths should be scaled by this factor
        for frontier in self.frontiers:
            if frontier in filtered_frontiers:
                continue
            IoU_values = np.asarray(
                [IoU(frontier.region, new_ft["region"]) for new_ft in valid_ft_angles]
            )
            pix_diff_values = np.asarray(
                [
                    pix_diff(frontier.region, new_ft["region"])
                    for new_ft in valid_ft_angles
                ]
            )
            frontier_appended = False
            if np.any(
                (
                    (IoU_values > cfg.region_equal_threshold)
                    & (pix_diff_values < 75 * scale_factor)
                )  # ensure that a normal step on a very large region can cause the large region to be considered as changed
                | (
                    pix_diff_values <= 3 * scale_factor
                )  # ensure that a very small region can be considered as unchanged
            ):  # do not update frontier that is too far from the agent:
                # the frontier is not changed (almost)
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove that new frontier
                ft_idx = np.argmax(IoU_values)
                valid_ft_angles.pop(ft_idx)
            elif (
                np.sum(IoU_values > 0.02) >= 2
                and cfg.region_equal_threshold
                < np.sum(IoU_values[IoU_values > 0.02])
                <= 1
            ):
                # if one old frontier is split into two new frontiers, and their sizes are equal
                # then keep the old frontier
                logging.debug(
                    f"Frontier one split to many: {IoU_values[IoU_values > 0.02]}"
                )
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove those new frontiers
                ft_ids = list(np.argwhere(IoU_values > 0.02).squeeze())
                ft_ids.sort(reverse=True)
                for ft_idx in ft_ids:
                    valid_ft_angles.pop(ft_idx)
            elif np.sum(IoU_values > 0.02) == 1:
                # if some old frontiers are merged into one new frontier
                ft_idx = np.argmax(IoU_values)
                IoU_with_old_ft = np.asarray(
                    [
                        IoU(valid_ft_angles[ft_idx]["region"], ft.region)
                        for ft in self.frontiers
                    ]
                )
                if (
                    np.sum(IoU_with_old_ft > 0.02) >= 2
                    and cfg.region_equal_threshold
                    < np.sum(IoU_with_old_ft[IoU_with_old_ft > 0.02])
                    <= 1
                ):
                    if (
                        np.sum(valid_ft_angles[ft_idx]["region"])
                        / np.sum(
                            [
                                np.sum(self.frontiers[old_ft_id].region)
                                for old_ft_id in np.argwhere(
                                    IoU_with_old_ft > 0.02
                                ).squeeze()
                            ]
                        )
                        > cfg.region_equal_threshold
                    ):
                        # if the new frontier is merged from two or more old frontiers, and their sizes are equal
                        # then add all the old frontiers
                        logging.debug(
                            f"Frontier many merged to one: {IoU_with_old_ft[IoU_with_old_ft > 0.02]}"
                        )
                        for i in list(np.argwhere(IoU_with_old_ft > 0.02).squeeze()):
                            if self.frontiers[i] not in filtered_frontiers:
                                filtered_frontiers.append(self.frontiers[i])
                                kept_frontier_area = (
                                    kept_frontier_area | self.frontiers[i].region
                                )
                        valid_ft_angles.pop(ft_idx)
                        frontier_appended = True

            if not frontier_appended:
                self.free_frontier(frontier)
                if np.any(IoU_values > 0.8):
                    # the frontier is slightly updated
                    # choose the new frontier that updates the current frontier
                    update_ft_idx = np.argmax(IoU_values)
                    ang = valid_ft_angles[update_ft_idx]["angle"]
                    # if the new frontier has no valid observations
                    if 1 > self._voxel_size * get_collision_distance(
                        occupied_map=occupied_map_camera,
                        pos=cur_point,
                        direction=np.array([np.cos(ang), np.sin(ang)]),
                    ):
                        # create a new frontier with the old image
                        old_img_path = frontier.image
                        old_img_feature = frontier.feature
                        filtered_frontiers.append(
                            self.create_frontier(
                                valid_ft_angles[update_ft_idx],
                                frontier_edge_areas=frontier_edge_areas,
                                cur_point=cur_point,
                            )
                        )
                        filtered_frontiers[-1].image = old_img_path
                        filtered_frontiers[-1].target_detected = (
                            frontier.target_detected
                        )
                        filtered_frontiers[-1].feature = old_img_feature
                        valid_ft_angles.pop(update_ft_idx)
                        kept_frontier_area = (
                            kept_frontier_area | filtered_frontiers[-1].region
                        )
        self.frontiers = filtered_frontiers

        # create new frontiers and add to frontier list
        for ft_data in valid_ft_angles:
            # exclude the new frontier's region that is already covered by the existing frontiers
            ft_data["region"] = ft_data["region"] & np.logical_not(kept_frontier_area)
            if np.sum(ft_data["region"]) > 0:
                self.frontiers.append(
                    self.create_frontier(
                        ft_data,
                        frontier_edge_areas=frontier_edge_areas,
                        cur_point=cur_point,
                    )
                )

        # Turn to face each frontier point and get rgb image
        for i, frontier in enumerate(self.frontiers):
            pos_habitat = self.voxel2habitat(frontier.position)
            assert (frontier.image is None and frontier.feature is None) or (
                frontier.image is not None and frontier.feature is not None
            ), f"{frontier.image}, {frontier.feature is None}"
            # Turn to face the frontier point
            if frontier.image is None:
                view_frontier_direction = np.array(
                    [
                        pos_habitat[0] - pts_habitat[0],
                        0.0,
                        pos_habitat[2] - pts_habitat[2],
                    ]
                )
                obs = scene.get_frontier_observation(
                    pts_habitat, view_frontier_direction
                )
                frontier_obs = obs["color_sensor"]

                if save_frontier_image:
                    assert os.path.exists(
                        eps_frontier_dir
                    ), f"Error in update_frontier_map: {eps_frontier_dir} does not exist"
                    plt.imsave(
                        os.path.join(eps_frontier_dir, f"{cnt_step}_{i}.png"),
                        frontier_obs,
                    )
                processed_rgb = resize_image(
                    frontier_obs, prompt_img_size[0], prompt_img_size[1]
                )
                frontier.image = f"{cnt_step}_{i}.png"
                frontier.feature = processed_rgb

        # Update room segmentation
        if pts is not None:
            room_success = self.update_room_map(cfg=cfg, pts=pts)

        return True

    def set_next_navigation_point(
        self,
        choice: Union[SnapShot, Frontier],
        pts,
        objects: MapObjectDict[int, Dict],
        cfg,
        pathfinder,
        random_position=False,
        observe_snapshot=True,
    ) -> bool:
        if self.max_point is not None or self.target_point is not None:
            # if the next point is already set
            logging.error(
                f"Error in set_next_navigation_point: the next point is already set: {self.max_point}, {self.target_point}"
            )
            return False
        pts = pos_habitat_to_normal(pts)
        cur_point = self.normal2voxel(pts)
        self.max_point = choice

        if type(choice) == SnapShot:
            obj_centers = [objects[obj_id]["bbox"].center for obj_id in choice.cluster]
            obj_centers = [self.habitat2voxel(center)[:2] for center in obj_centers]
            obj_centers = list(
                set([tuple(center) for center in obj_centers])
            )  # remove duplicates
            obj_centers = np.asarray(obj_centers)
            snapshot_center = np.mean(obj_centers, axis=0)
            choice.position = snapshot_center

            if not observe_snapshot:
                # if the agent does not need to observe the snapshot, then the target point is the snapshot center
                target_point = snapshot_center
                self.target_point = get_nearest_true_point(
                    target_point, self.unoccupied
                )  # get the nearest unoccupied point for the nav target
                return True

            if len(obj_centers) == 1:
                # if there is only one object in the snapshot, then the target point is the object center
                target_point = snapshot_center
                # # set the object center as the navigation target
                # target_navigable_point = get_nearest_true_point(target_point, unoccupied)  # get the nearest unoccupied point for the nav target
                # since it's not proper to directly go to the target point,
                # we'd better find a navigable point that is certain distance from it to better observe the target
                if not random_position:
                    # the target navigation point is deterministic
                    target_navigable_point = get_proper_observe_point(
                        target_point,
                        self.unoccupied,
                        cur_point=cur_point,
                        dist=cfg.final_observe_distance / self._voxel_size,
                    )
                else:
                    target_navigable_point = get_random_observe_point(
                        target_point,
                        self.unoccupied,
                        min_dist=cfg.final_observe_distance / self._voxel_size,
                        max_dist=(cfg.final_observe_distance + 1.5) / self._voxel_size,
                    )
                if target_navigable_point is None:
                    # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                    # so we just temporarily use pathfinder to find a navigable point around it
                    target_point_normal = (
                        target_point * self._voxel_size + self._vol_origin[:2]
                    )
                    target_point_normal = np.append(target_point_normal, pts[-1])
                    target_point_habitat = pos_normal_to_habitat(target_point_normal)

                    target_navigable_point_habitat = (
                        get_proper_observe_point_with_pathfinder(
                            target_point_habitat, pathfinder, height=pts[-1]
                        )
                    )
                    if target_navigable_point_habitat is None:
                        logging.error(
                            f"Error in set_next_navigation_point: cannot find a proper navigable point around the target object"
                        )
                        return False

                    target_navigable_point = self.habitat2voxel(
                        target_navigable_point_habitat
                    )[:2]
                self.target_point = target_navigable_point
                return True
            else:
                if not random_position:
                    target_point = get_proper_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=choice.obs_point,
                        unoccupied_map=self.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / self._voxel_size - 1,
                        max_obs_dist=cfg.final_observe_distance / self._voxel_size + 1,
                    )
                else:
                    target_point = get_random_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=choice.obs_point,
                        unoccupied_map=self.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / self._voxel_size - 1,
                        max_obs_dist=cfg.final_observe_distance / self._voxel_size + 1,
                    )
                if target_point is None:
                    logging.error(
                        f"Error in set_next_navigation_point: cannot find a proper observation point for the snapshot"
                    )
                    return False

                self.target_point = target_point
                return True
        elif type(choice) == Frontier:
            # find the direction into unexplored
            ft_direction = self.max_point.orientation

            # find an unoccupied point between the agent and the frontier
            next_point = np.array(self.max_point.position, dtype=float)
            try_count = 0
            while (
                not self.check_within_bnds(next_point.astype(int))
                or self.occupied[int(next_point[0]), int(next_point[1])]
                or not self.island[int(next_point[0]), int(next_point[1])]
            ):
                next_point -= ft_direction
                try_count += 1
                if try_count > 1000:
                    logging.error(
                        f"Error in set_next_navigation_point: cannot find a proper next point"
                    )
                    return False

            self.target_point = next_point.astype(int)
            return True
        else:
            logging.error(
                f"Error in find_next_pose_with_path: wrong choice type: {type(choice)}"
            )
            return False

    def agent_step(
        self,
        pts,
        angle,
        objects: MapObjectDict[int, Dict],
        snapshots: Dict[str, SnapShot],
        pathfinder,
        cfg,
        path_points=None,
        save_visualization=True,
    ):
        if self.max_point is None or self.target_point is None:
            logging.error(
                f"Error in agent_step: max_point or next_point is None: {self.max_point}, {self.target_point}"
            )
            return (None,)
        # check the distance to next navigation point
        # if the target navigation point is too far
        # then just go to a point between the current point and the target point
        pts = pos_habitat_to_normal(pts)
        cur_point = self.normal2voxel(pts)
        max_dist_from_cur = (
            cfg.max_dist_from_cur_phase_1
            if type(self.max_point) == Frontier
            else cfg.max_dist_from_cur_phase_2
        )  # in phase 2, the step size should be smaller
        dist, path_to_target = self.get_distance(
            cur_point[:2], self.target_point, height=pts[2], pathfinder=pathfinder
        )

        if dist > max_dist_from_cur:
            target_arrived = False
            if path_to_target is not None:
                # drop the y value of the path to avoid errors when calculating seg_length
                path_to_target = [np.asarray([p[0], 0.0, p[2]]) for p in path_to_target]
                # if the pathfinder find a path, then just walk along the path for max_dist_from_cur distance
                dist_to_travel = max_dist_from_cur
                next_point = None
                for i in range(len(path_to_target) - 1):
                    seg_length = np.linalg.norm(
                        path_to_target[i + 1] - path_to_target[i]
                    )
                    if seg_length < dist_to_travel:
                        dist_to_travel -= seg_length
                    else:
                        # find the point on the segment according to the length ratio
                        next_point_habitat = (
                            path_to_target[i]
                            + (path_to_target[i + 1] - path_to_target[i])
                            * dist_to_travel
                            / seg_length
                        )
                        next_point = self.normal2voxel(
                            pos_habitat_to_normal(next_point_habitat)
                        )[:2]
                        break
                if next_point is None:
                    # this is a very rare case that, the sum of the segment lengths is smaller than the dist returned by the pathfinder
                    # and meanwhile the max_dist_from_cur larger than the sum of the segment lengths
                    # resulting that the previous code cannot find a proper point in the middle of the path
                    # in this case, just go to the target point
                    next_point = self.target_point.copy()
                    target_arrived = True
            else:
                # if the pathfinder cannot find a path, then just go to a point between the current point and the target point
                logging.info(
                    f"pathfinder cannot find a path from {cur_point[:2]} to {self.target_point}, just go to a point between them"
                )
                walk_dir = self.target_point - cur_point[:2]
                walk_dir = walk_dir / np.linalg.norm(walk_dir)
                next_point = (
                    cur_point[:2] + walk_dir * max_dist_from_cur / self._voxel_size
                )
                # ensure next point is valid, otherwise go backward a bit
                try_count = 0
                while (
                    not self.check_within_bnds(next_point)
                    or not self.island[
                        int(np.round(next_point[0])), int(np.round(next_point[1]))
                    ]
                    or self.occupied[
                        int(np.round(next_point[0])), int(np.round(next_point[1]))
                    ]
                ):
                    next_point -= walk_dir
                    try_count += 1
                    if try_count > 1000:
                        logging.error(
                            f"Error in agent_step: cannot find a proper next point"
                        )
                        return (None,)
                next_point = np.round(next_point).astype(int)
        else:
            target_arrived = True
            next_point = self.target_point.copy()

        next_point_old = next_point.copy()
        next_point = adjust_navigation_point(
            next_point,
            self.occupied,
            voxel_size=self._voxel_size,
            max_adjust_distance=0.1,
        )

        # determine the direction
        if target_arrived:  # if the next arriving position is the target point
            if type(self.max_point) == Frontier:
                # direction = self.rad2vector(angle)  # if the target is a frontier, then the agent's orientation does not change
                direction = (
                    self.target_point - cur_point[:2]
                )  # if the target is a frontier, then the agent should face the target point
            else:
                direction = (
                    self.max_point.position - cur_point[:2]
                )  # if the target is an object, then the agent should face the object
        else:  # the agent is still on the way to the target point
            direction = next_point - cur_point[:2]
        if (
            np.linalg.norm(direction) < 1e-6
        ):  # this is a rare case that next point is the same as the current point
            # usually this is a problem in the pathfinder
            logging.warning(
                f"Warning in agent_step: next point is the same as the current point when determining the direction"
            )
            direction = self.rad2vector(angle)
        direction = direction / np.linalg.norm(direction)

        # Plot
        fig = None
        if save_visualization:
            h, w = self._tsdf_vol_cpu.shape[:2]
            h = 8 * h / w
            arr_scale = (
                0.1 / self._voxel_size
            )  # when for default voxel size=0.1m, the unit length is 1

            fig, ax1 = plt.subplots(figsize=(8, h))

            ft_map = np.zeros(
                (self._tsdf_vol_cpu.shape[0], self._tsdf_vol_cpu.shape[1], 3),
                dtype=np.uint8,
            ) + np.asarray([[[255, 255, 255]]], dtype=np.uint8)

            _, unoccupied_high = self.get_island_around_pts(pts, height=1.8)
            obstacle_map = self.get_obstacle_map(height=1.8)
            # convolution to get the obstacles together with surroundings
            kernel_size = int(0.3 / self._voxel_size)
            kernel = np.ones((kernel_size, kernel_size))
            obstacle_map_convolved = ndimage.convolve(
                obstacle_map.astype(float), kernel, mode="constant", cval=0.0
            )

            # assign colors to the map
            ft_map[unoccupied_high > 0] = [200, 200, 200]
            ft_map[(self.unexplored == 0) & (unoccupied_high > 0)] = [194, 246, 198]
            ft_map[
                (obstacle_map_convolved > 0)
                & (obstacle_map_convolved < kernel_size**2 / 2)
            ] = [100, 100, 100]
            ft_map[(obstacle_map_convolved >= kernel_size**2 / 2)] = [0, 0, 0]

            ax1.imshow(ft_map)
            ax1.axis("off")

            agent_orientation = self.rad2vector(angle)

            ax1.scatter(
                cur_point[1],
                cur_point[0],
                c=(23 / 255, 188 / 255, 243 / 255),
                s=400,
                label="current",
            )
            end_x, end_y = (
                cur_point[1] + agent_orientation[1] * 5 * arr_scale,
                cur_point[0] + agent_orientation[0] * 5 * arr_scale,
            )
            ax1.plot(
                [cur_point[1], end_x],
                [cur_point[0], end_y],
                color="black",
                linewidth=5 * arr_scale,
            )

            for key, snapshot in snapshots.items():
                obs_point = snapshot.obs_point[:2]
                obj_points = [
                    self.habitat2voxel(objects[obj_id]["bbox"].center)[:2]
                    for obj_id in snapshot.cluster
                ]
                obj_center = np.mean(obj_points, axis=0)
                view_direction = obj_center - obs_point
                center_angle = (
                    np.arctan2(view_direction[0], view_direction[1]) * 180 / np.pi
                )
                obj_angles = [
                    np.arctan2(obj_point[0] - obs_point[0], obj_point[1] - obs_point[1])
                    * 180
                    / np.pi
                    for obj_point in obj_points
                ]
                # adjust the angles into proper range
                obj_angles = [
                    angle if angle > 0 else angle + 360 for angle in obj_angles
                ]  # range from 0 to 360
                if max(obj_angles) - min(obj_angles) > 180:
                    obj_angles = [
                        angle - 360 if angle > 180 else angle for angle in obj_angles
                    ]  # range from -180 to 180

                radius = np.linalg.norm(obj_points - obs_point, axis=1).max()
                wedge = Wedge(
                    center=(obs_point[1], obs_point[0]),
                    r=radius,
                    theta1=min(obj_angles) - 5,
                    theta2=max(obj_angles) + 5,
                    color=snapshot.color,
                    alpha=0.3,
                )

                # Add edge to the wedge
                if (
                    type(self.max_point) == SnapShot
                    and snapshot.image == self.max_point.image
                ):
                    edge_width = 7
                    wedge_edge = Wedge(
                        center=(obs_point[1], obs_point[0]),
                        r=radius,
                        theta1=min(obj_angles) - 5,
                        theta2=max(obj_angles) + 5,
                        facecolor="none",  # No face color for the edge wedge
                        edgecolor="red",
                        linewidth=edge_width,
                    )
                    ax1.add_patch(wedge_edge)

                ax1.add_patch(wedge)

                for obj_id in snapshot.cluster:
                    obj_vox = self.habitat2voxel(objects[obj_id]["bbox"].center)
                    ax1.scatter(obj_vox[1], obj_vox[0], color=snapshot.color, s=30)

            if type(self.max_point) == SnapShot:
                for obj_id in self.max_point.cluster:
                    obj_vox = self.habitat2voxel(objects[obj_id]["bbox"].center)
                    ax1.scatter(obj_vox[1], obj_vox[0], color="r", s=30)

            for frontier in self.frontiers:
                ax1.scatter(
                    frontier.position[1], frontier.position[0], color="m", s=30, alpha=1
                )
                normal = frontier.orientation
                dx, dy = normal * 10 * arr_scale
                arrow = FancyArrowPatch(
                    posA=(frontier.position[1], frontier.position[0]),
                    posB=(frontier.position[1] + dy, frontier.position[0] + dx),
                    arrowstyle=f"Simple, tail_width={0.5}, head_width={5}, head_length={5}",
                    linewidth=3,
                    color="m",
                    mutation_scale=1,
                )

                if type(self.max_point) == Frontier and frontier == self.max_point:
                    ax1.scatter(
                        frontier.position[1],
                        frontier.position[0],
                        color="r",
                        s=30,
                        alpha=1,
                    )
                    # Add edge to the arrow
                    arrow.set_path_effects(
                        [
                            pe.Stroke(
                                linewidth=5, foreground="red"
                            ),  # Edge with linewidth 3 and red color
                            pe.Normal(),  # Render the original arrow
                        ]
                    )

                ax1.add_patch(arrow)

        # Convert back to world coordinates
        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]

        # Find the yaw angle again
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2

        # update the path points
        if path_points is not None:
            updated_path_points = self.update_path_points(
                path_points, next_point_normal
            )
        else:
            updated_path_points = None

        # set the surrounding points of the next point as explored
        unoccupied_coords = np.argwhere(self.unoccupied)
        dists_unoccupied = np.linalg.norm(unoccupied_coords - next_point, axis=1)
        near_coords = unoccupied_coords[
            dists_unoccupied < cfg.surrounding_explored_radius / self._voxel_size
        ]
        self._explore_vol_cpu[near_coords[:, 0], near_coords[:, 1], :] = 1

        if target_arrived:
            self.max_point = None
            self.target_point = None

        return (
            self.normal2habitat(next_point_normal),
            next_yaw,
            next_point,
            fig,
            updated_path_points,
            target_arrived,
        )

    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        """Find the empty space around the point (x,y,z) in the world frame"""
        # Convert to voxel coordinates
        cur_point = self.normal2voxel(pts)

        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        # fill_size = int(fill_dim / self._voxel_size)
        # structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        # unoccupied = close_operation(unoccupied, structuring_element_close)

        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind

        assert (islands == 0).sum() == (
            unoccupied == 0
        ).sum(), f"{(islands == 0).sum()} != {(unoccupied == 0).sum()}"

        # also we need to include the island of all existing frontiers when calculating island at the same height as frontier
        if abs(height - self.occupancy_height) < 1e-3:
            for frontier in self.frontiers:
                frontier_inds = islands[frontier.region]
                # get the most common index
                mode_result = stats.mode(frontier_inds, axis=None)
                frontier_ind = mode_result.mode
                if frontier_ind != 0:
                    island = island | (islands == frontier_ind)

        return island, unoccupied

    def get_frontier_region_map(self, frontier_coordinates):
        # frontier_coordinates: [N, 2] ndarray of the coordinates covered by the frontier in voxel space
        region_map = np.zeros_like(self.frontier_map, dtype=bool)
        for coord in frontier_coordinates:
            region_map[coord[0], coord[1]] = True
        return region_map

    def create_frontier(
        self, ft_data: dict, frontier_edge_areas, cur_point
    ) -> Frontier:
        ft_direction = np.array([np.cos(ft_data["angle"]), np.sin(ft_data["angle"])])

        kernel = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        frontier_edge = ndimage.convolve(
            ft_data["region"].astype(int), kernel, mode="constant", cval=0
        )

        frontier_edge_areas_filtered = np.asarray(
            [p for p in frontier_edge_areas if 2 <= frontier_edge[p[0], p[1]] <= 12]
        )
        if len(frontier_edge_areas_filtered) > 0:
            frontier_edge_areas = frontier_edge_areas_filtered

        all_directions = frontier_edge_areas - cur_point[:2]
        all_direction_norm = np.linalg.norm(all_directions, axis=1, keepdims=True)
        all_direction_norm = np.where(
            all_direction_norm == 0, np.inf, all_direction_norm
        )
        all_directions = all_directions / all_direction_norm

        # the center is the closest point in the edge areas from current point that have close cosine angles
        cos_sim_rank = np.argsort(-np.dot(all_directions, ft_direction))
        center_candidates = np.asarray(
            [frontier_edge_areas[idx] for idx in cos_sim_rank[:5]]
        )
        center = center_candidates[
            np.argmin(np.linalg.norm(center_candidates - cur_point[:2], axis=1))
        ]
        center = adjust_navigation_point(
            center,
            self.occupied,
            max_dist=0.5,
            max_adjust_distance=0.3,
            voxel_size=self._voxel_size,
        )

        region = ft_data["region"]

        # allocate an id for the frontier
        # assert np.all(self.frontier_map[region] == 0)
        frontier_id = self.frontier_counter
        self.frontier_map[region] = frontier_id
        self.frontier_counter += 1

        return Frontier(
            position=center,
            orientation=ft_direction,
            region=region,
            frontier_id=frontier_id,
        )

    def free_frontier(self, frontier: Frontier):
        self.frontier_map[self.frontier_map == frontier.frontier_id] = 0

    # ── Room Segmentation Methods ───────────────────────────────────────

    @staticmethod
    def _cfg_get(cfg, key, default):
        if cfg is None:
            return default
        try:
            return getattr(cfg, key)
        except Exception:
            try:
                return cfg[key]
            except Exception:
                return default

    def get_room_id_at(self, point: np.ndarray) -> int:
        if self.room_map is None or point is None:
            return -1
        if len(self.room_regions) == 0 or not np.any(self.room_map > 0):
            return -1
        point = np.asarray(point).astype(int)
        if point.shape[0] > 2:
            point = point[:2]
        if (point[0] < 0 or point[0] >= self.room_map.shape[0]
            or point[1] < 0 or point[1] >= self.room_map.shape[1]):
            return -1
        room_id = int(self.room_map[point[0], point[1]])
        if room_id != 0:
            return room_id
        nearest = get_nearest_true_point(point, self.room_map > 0)
        if nearest is None:
            return -1
        return int(self.room_map[nearest[0], nearest[1]])

    def _find_geometric_room_seed_masks(
        self, room_area, min_room_area, seed_radius, seed_min_distance,
    ) -> List[np.ndarray]:
        seeds = []
        labels = measure.label(room_area, connectivity=1)
        kernel_size = max(3, int(seed_min_distance) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        for label in np.unique(labels):
            if label == 0:
                continue
            component = labels == label
            if np.count_nonzero(component) < min_room_area:
                continue
            dist = cv2.distanceTransform(
                component.astype(np.uint8), cv2.DIST_L2, maskSize=5)
            local_max = (dist == cv2.dilate(dist, kernel)) & component & (dist > 0)
            peak_labels = measure.label(local_max, connectivity=1)
            peak_centers = []
            for peak_label in np.unique(peak_labels):
                if peak_label == 0:
                    continue
                peak = peak_labels == peak_label
                peak_coords = np.argwhere(peak)
                if len(peak_coords) == 0:
                    continue
                peak_values = dist[peak]
                best = peak_coords[int(np.argmax(peak_values))]
                peak_centers.append(best)
            if len(peak_centers) == 0:
                coords = np.argwhere(component)
                center = get_nearest_true_point(
                    np.round(coords.mean(axis=0)).astype(int), component)
                if center is None:
                    center = coords[0]
                peak_centers = [center]
            for center in peak_centers:
                seed = self._disk_seed_mask(
                    center=np.asarray(center, dtype=int),
                    room_area=room_area,
                    radius=max(1, seed_radius),
                )
                if np.any(seed):
                    seeds.append(seed)
        return seeds

    @staticmethod
    def _disk_seed_mask(center: np.ndarray, room_area: np.ndarray, radius: int) -> np.ndarray:
        seed = np.zeros_like(room_area, dtype=bool)
        y_min = max(0, center[0] - radius)
        y_max = min(room_area.shape[0], center[0] + radius + 1)
        x_min = max(0, center[1] - radius)
        x_max = min(room_area.shape[1], center[1] + radius + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        disk = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
        seed[y_min:y_max, x_min:x_max] = disk & room_area[y_min:y_max, x_min:x_max]
        return seed

    @staticmethod
    def _grow_regions_from_seeds(room_area: np.ndarray, seeds: List[np.ndarray]) -> np.ndarray:
        labels = np.zeros(room_area.shape, dtype=np.int32)
        queue = deque()
        for label, seed in enumerate(seeds, start=1):
            for point in np.argwhere(seed & room_area):
                row, col = int(point[0]), int(point[1])
                if labels[row, col] == 0:
                    labels[row, col] = label
                    queue.append((row, col, label))
        while queue:
            row, col, label = queue.popleft()
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if (next_row < 0 or next_row >= room_area.shape[0]
                    or next_col < 0 or next_col >= room_area.shape[1]):
                    continue
                if not room_area[next_row, next_col] or labels[next_row, next_col] != 0:
                    continue
                labels[next_row, next_col] = label
                queue.append((next_row, next_col, label))
        return labels

    @staticmethod
    def _make_watershed_seeds_from_grown_regions(grown_regions: List[np.ndarray], seed_area_min: int) -> List[np.ndarray]:
        seeds = []
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for region in grown_regions:
            eroded = cv2.erode(region.astype(np.uint8), kernel, iterations=1) > 0
            if np.count_nonzero(eroded) < seed_area_min:
                seeds.append(region)
            else:
                seeds.append(eroded)
        return seeds

    @staticmethod
    def _fill_watershed_boundary_gaps(
        refined_regions: List[np.ndarray],
        grown_labels: np.ndarray,
        room_area: np.ndarray,
        min_room_area: int,
    ) -> List[np.ndarray]:
        labels = np.zeros(room_area.shape, dtype=np.int32)
        for label, region in enumerate(refined_regions, start=1):
            labels[region] = label
        missing = room_area & (labels == 0)
        if np.any(missing):
            labels[missing] = grown_labels[missing]
        return [labels == label for label in range(1, len(refined_regions) + 1)
                if np.count_nonzero(labels == label) >= min_room_area]

    def _watershed_room_regions(self, room_area, seeds, min_room_area):
        markers = np.zeros(room_area.shape, dtype=np.int32)
        for idx, seed in enumerate(seeds, start=1):
            markers[seed] = idx
        background_id = len(seeds) + 1
        markers[np.logical_not(room_area)] = background_id
        watershed_img = np.repeat(
            (room_area.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        cv2.watershed(watershed_img, markers)
        regions = []
        for label in range(1, background_id):
            region = (markers == label) & room_area
            if np.count_nonzero(region) >= min_room_area:
                regions.append(region)
        return regions

    def _filter_room_regions_by_observed_adjacency(
        self, regions, observed_threshold=0.30, max_unobserved_hops=1, adjacency_dilation=1,
    ):
        if max_unobserved_hops < 0 or self.unexplored is None:
            return regions
        if len(regions) == 0:
            return []
        explored = self.unexplored == 0
        for point in getattr(self, "init_points", []):
            if (0 <= point[0] < explored.shape[0]
                and 0 <= point[1] < explored.shape[1]):
                explored[point[0], point[1]] = True
        observed_indices = {idx for idx, region in enumerate(regions)
                            if np.any(region) and float(np.mean(explored[region])) >= observed_threshold}
        if len(observed_indices) == 0:
            return []
        allowed_indices = set(observed_indices)
        frontier_indices = set(observed_indices)
        for _ in range(max_unobserved_hops):
            next_indices = set()
            for idx, region in enumerate(regions):
                if idx in allowed_indices:
                    continue
                if any(self._room_regions_are_adjacent(region, regions[fi], dilation_iterations=adjacency_dilation)
                       for fi in frontier_indices):
                    next_indices.add(idx)
            if len(next_indices) == 0:
                break
            allowed_indices.update(next_indices)
            frontier_indices = next_indices
        return [region for idx, region in enumerate(regions) if idx in allowed_indices]

    @staticmethod
    def _room_regions_are_adjacent(region_a, region_b, dilation_iterations=1) -> bool:
        if not np.any(region_a) or not np.any(region_b):
            return False
        if np.any(region_a & region_b):
            return True
        iterations = max(1, int(dilation_iterations))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(region_a.astype(np.uint8), kernel, iterations=iterations) > 0
        return bool(np.any(dilated & region_b))

    def _commit_room_regions(self, regions, stable_iou_threshold, observed_threshold=0.30):
        previous_regions = self.room_regions
        used_previous_ids = set()
        room_map = np.zeros(self._vol_dim[:2], dtype=int)
        room_regions = []
        regions = sorted(regions, key=lambda r: np.count_nonzero(r), reverse=True)
        for region in regions:
            room_id = self._match_previous_room_id(
                region, previous_regions, used_previous_ids, stable_iou_threshold)
            if room_id is None:
                room_id = self.room_counter
                self.room_counter += 1
            used_previous_ids.add(room_id)
            coords = np.argwhere(region)
            center = coords.mean(axis=0)
            center = get_nearest_true_point(np.round(center).astype(int), region)
            if center is None:
                center = coords[0]
            room_map[region] = room_id
            room_regions.append(RoomRegion(
                room_id=room_id, center=np.asarray(center, dtype=int),
                region=region, area=int(np.count_nonzero(region)),
            ))
        self.room_map = room_map
        self.room_regions = sorted(room_regions, key=lambda r: r.room_id)
        self._refresh_room_observation_states(observed_threshold=observed_threshold)

    @staticmethod
    def _match_previous_room_id(region, previous_regions, used_previous_ids, stable_iou_threshold):
        best_iou = 0.0
        best_room_id = None
        for previous in previous_regions:
            if previous.room_id in used_previous_ids:
                continue
            intersection = np.count_nonzero(region & previous.region)
            union = np.count_nonzero(region | previous.region)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_room_id = previous.room_id
        if best_iou >= stable_iou_threshold:
            return best_room_id
        return None

    def _refresh_room_observation_states(self, observed_threshold=0.30):
        if self.unexplored is None:
            return
        explored = self.unexplored == 0
        for room in self.room_regions:
            if not np.any(room.region):
                room.observed_ratio = 0.0
                room.room_state = "unknown"
                continue
            room.observed_ratio = float(np.mean(explored[room.region]))
            room.room_state = (
                "observed" if room.observed_ratio >= observed_threshold else "hypothesis"
            )

    def _assign_frontiers_to_rooms(self):
        for room in self.room_regions:
            room.frontier_ids = []
        room_by_id = {room.room_id: room for room in self.room_regions}
        for frontier in self.frontiers:
            frontier.room_id = self.get_room_id_at(frontier.position)
            if frontier.room_id in room_by_id:
                room_by_id[frontier.room_id].frontier_ids.append(frontier.frontier_id)

    def _get_room_segmentation_envelope(self, area_source, unoccupied_high):
        navigable = unoccupied_high.astype(bool)
        explored = self.unexplored == 0
        for point in getattr(self, "init_points", []):
            if (0 <= point[0] < explored.shape[0]
                and 0 <= point[1] < explored.shape[1]):
                explored[point[0], point[1]] = True
        if area_source in {"navigable", "all", "explored_and_unexplored"}:
            return navigable
        if area_source == "unexplored":
            return navigable & np.logical_not(explored)
        if area_source != "explored":
            logging.warning(
                "Unknown room_segmentation.area_source=%s; falling back to explored.",
                area_source)
        return navigable & explored

    @staticmethod
    def _connected_room_regions(room_area, min_room_area):
        labels = measure.label(room_area, connectivity=1)
        return [labels == label for label in np.unique(labels)
                if label != 0 and np.count_nonzero(labels == label) >= min_room_area]

    def _find_room_seed_masks(self, room_area, min_area, max_area, max_iterations):
        boundary = np.logical_not(room_area).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        seeds = []
        for _ in range(max_iterations):
            boundary = cv2.dilate(boundary, kernel, iterations=1)
            remaining_free = cv2.bitwise_not(boundary)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                remaining_free, connectivity=8)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    mask = labels == label
                    if np.any(mask):
                        seeds.append(mask)
                        boundary[mask] = 255
            if np.all(boundary == 255):
                break
        return seeds

    def _region_growing_watershed_room_regions(
        self, room_area, min_room_area, seed_area_min, seed_radius, seed_min_distance,
    ):
        seeds = self._find_geometric_room_seed_masks(
            room_area=room_area, min_room_area=min_room_area,
            seed_radius=seed_radius, seed_min_distance=seed_min_distance,
        )
        if len(seeds) == 0:
            return self._connected_room_regions(room_area, min_room_area)
        grown_labels = self._grow_regions_from_seeds(room_area, seeds)
        grown_regions = [grown_labels == label
                         for label in range(1, len(seeds) + 1)
                         if np.count_nonzero(grown_labels == label) >= min_room_area]
        if len(grown_regions) == 0:
            return self._connected_room_regions(room_area, min_room_area)
        watershed_seeds = self._make_watershed_seeds_from_grown_regions(
            grown_regions=grown_regions, seed_area_min=seed_area_min,
        )
        refined_regions = self._watershed_room_regions(
            room_area, watershed_seeds, min_room_area)
        if len(refined_regions) == 0:
            return grown_regions
        return self._fill_watershed_boundary_gaps(
            refined_regions=refined_regions, grown_labels=grown_labels,
            room_area=room_area, min_room_area=min_room_area,
        )

    def update_room_map(self, cfg=None, pts=None) -> bool:
        if self.unoccupied is None or self.island is None or pts is None:
            return False

        min_room_area = self._cfg_get(cfg, "min_room_area", 16)
        seed_area_min = self._cfg_get(cfg, "seed_area_min", 6)
        seed_area_max = self._cfg_get(cfg, "seed_area_max", 400)
        max_dilation_iter = self._cfg_get(cfg, "max_dilation_iter", 80)
        stable_iou_threshold = self._cfg_get(cfg, "stable_iou_threshold", 0.2)
        close_iterations = self._cfg_get(cfg, "close_iterations", 1)
        room_height = self._cfg_get(cfg, "room_height", 1.8)
        observed_threshold = float(self._cfg_get(cfg, "observed_ratio_threshold", 0.30))
        max_unobserved_hops = int(self._cfg_get(cfg, "max_unobserved_room_hops", 1))
        adjacency_dilation = int(self._cfg_get(cfg, "room_adjacency_dilation", 1))
        area_source = str(self._cfg_get(cfg, "area_source", "explored")).lower()
        segmentation_method = str(
            self._cfg_get(cfg, "segmentation_method", "dilation_watershed")).lower()

        _, unoccupied_high = self.get_island_around_pts(pts, height=room_height)
        room_envelope = self._get_room_segmentation_envelope(
            area_source=area_source, unoccupied_high=unoccupied_high)
        room_area = room_envelope.copy()
        if np.count_nonzero(room_area) < min_room_area:
            self.clear_room_map()
            return False

        if close_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            room_area_u8 = (room_area.astype(np.uint8) * 255)
            room_area = (cv2.morphologyEx(
                room_area_u8, cv2.MORPH_CLOSE, kernel,
                iterations=close_iterations) > 0)
            room_area &= room_envelope

        if segmentation_method == "region_growing_watershed":
            regions = self._region_growing_watershed_room_regions(
                room_area=room_area, min_room_area=min_room_area,
                seed_area_min=seed_area_min,
                seed_radius=int(self._cfg_get(cfg, "region_seed_radius", 2)),
                seed_min_distance=int(
                    self._cfg_get(cfg, "region_seed_min_distance", 12)),
            )
        else:
            if segmentation_method != "dilation_watershed":
                logging.warning(
                    "Unknown room_segmentation.segmentation_method=%s; "
                    "falling back to dilation_watershed.", segmentation_method)
            seeds = self._find_room_seed_masks(
                room_area, min_area=seed_area_min, max_area=seed_area_max,
                max_iterations=max_dilation_iter)
            if len(seeds) == 0:
                regions = self._connected_room_regions(room_area, min_room_area)
            else:
                regions = self._watershed_room_regions(room_area, seeds, min_room_area)

        if len(regions) == 0:
            self.clear_room_map()
            return False

        regions = self._filter_room_regions_by_observed_adjacency(
            regions=regions, observed_threshold=observed_threshold,
            max_unobserved_hops=max_unobserved_hops,
            adjacency_dilation=adjacency_dilation)
        if len(regions) == 0:
            self.clear_room_map()
            return False

        self._commit_room_regions(regions, stable_iou_threshold,
                                  observed_threshold=observed_threshold)
        self._assign_frontiers_to_rooms()
        return True

    def clear_room_map(self):
        self.room_map = np.zeros(self._vol_dim[:2], dtype=int)
        self.room_regions = []
        for frontier in self.frontiers:
            frontier.room_id = -1

    # ── Spiral search for nearest navigable point ──────────────────────

    def spiral_search_navigable_point(
        self, pathfinder, target_voxel_xy, agent_habitat,
        max_radius_voxels=80, floor_height=0.07,
    ):
        """Clockwise spiral search on the 2D voxel grid starting from target.

        Returns the *nearest* navigable habitat point on the agent's reachable
        island, or None if nothing found within max_radius_voxels.

        Returns dict: {habitat_pos, voxel_xy, search_steps, spiral_dist} or None.
        """
        from src.habitat import pos_normal_to_habitat
        cy, cx = int(target_voxel_xy[0]), int(target_voxel_xy[1])
        h, w = self._tsdf_vol_cpu.shape[:2]
        island = self.island

        def _check_voxel(vy, vx):
            if vy < 0 or vy >= h or vx < 0 or vx >= w:
                return None
            if not island[vy, vx]:
                return None
            normal_pos = self.voxel2normal(np.array([vy, vx]))
            normal_3d = np.array([normal_pos[0], normal_pos[1], floor_height])
            hab_pos = pos_normal_to_habitat(normal_3d)
            snapped = pathfinder.snap_point(hab_pos[:3])
            if snapped is None or np.isnan(snapped).any():
                return None
            if abs(snapped[1] - floor_height) > 0.3:
                return None
            snapped_voxel = self.habitat2voxel(snapped)[:2]
            if not self.check_within_bnds(snapped_voxel):
                return None
            sv_y, sv_x = int(snapped_voxel[0]), int(snapped_voxel[1])
            if sv_y < 0 or sv_y >= h or sv_x < 0 or sv_x >= w:
                return None
            if not island[sv_y, sv_x]:
                return None
            if not pathfinder.is_navigable(snapped):
                return None
            return {"habitat_pos": snapped, "voxel_xy": (sv_y, sv_x)}

        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        total_checked = 0
        vy, vx = cy, cx

        result = _check_voxel(vy, vx)
        total_checked += 1
        if result is not None:
            result["search_steps"] = total_checked
            result["spiral_dist"] = 0
            return result

        step_len = 1
        steps_done = 0
        dir_idx = 0
        segs_at_len = 0

        for _ in range(max_radius_voxels * max_radius_voxels * 4):
            dy, dx = DIRS[dir_idx]
            vy += dy
            vx += dx
            steps_done += 1

            result = _check_voxel(vy, vx)
            total_checked += 1
            if result is not None:
                dist = abs(vy - cy) + abs(vx - cx)
                result["search_steps"] = total_checked
                result["spiral_dist"] = dist
                return result

            if steps_done == step_len:
                steps_done = 0
                dir_idx = (dir_idx + 1) % 4
                segs_at_len += 1
                if segs_at_len == 2:
                    segs_at_len = 0
                    step_len += 1

            if abs(vy - cy) > max_radius_voxels or abs(vx - cx) > max_radius_voxels:
                break

        return None

    def refresh_planner_grids(self, pts):
        """Re-derive island/unoccupied/unexplored/occupied from current TSDF.

        Mirrors update_frontier_map's grid setup. Call after integrate()
        to make spiral_search see the latest navigable area.
        """
        from scipy import ndimage
        from src.habitat import pos_habitat_to_normal
        pts_normal = pos_habitat_to_normal(pts)
        island, unoccupied = self.get_island_around_pts(
            pts_normal, height=self.occupancy_height)
        self.unoccupied = unoccupied
        self.island = island
        self.unexplored = (
            np.sum(self._explore_vol_cpu, axis=-1) == 0
        ).astype(int)
        for point in self.init_points:
            self.unexplored[point[0], point[1]] = 0
        self.occupied = np.logical_not(self.unoccupied).astype(int)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.unexplored_neighbors = ndimage.convolve(
            self.unexplored, kernel, mode="constant", cval=0.0)
        self.occupied_map_camera = np.logical_not(
            self.get_island_around_pts(
                pts_normal, height=self.vision_height)[0]
        )
