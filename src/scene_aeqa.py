import os
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
from quaternion import as_float_array
import supervision as sv
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
import copy
from PIL import Image

from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from src.habitat import (
    make_semantic_cfg,
    make_simple_cfg,
    get_quaternion,
    get_navigable_point_to,
    pos_normal_to_habitat,
)
from src.geom import get_cam_intr, IoU
from src.utils import rgba2rgb
from src.tsdf_planner import SnapShot
from src.hierarchy_clustering import SceneHierarchicalClustering

# Local application/library specific imports
from src.conceptgraph.utils.ious import mask_subtract_contained
from src.conceptgraph.utils.general_utils import (
    ObjectClasses,
    measure_time,
    filter_detections,
)
from src.conceptgraph.slam.slam_classes import MapObjectDict, DetectionDict, to_tensor
from src.conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
)
from src.conceptgraph.utils.model_utils import compute_clip_features_batched


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.cfg = cfg
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # about the loading the scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(
            cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".basis.navmesh",
        )
        semantic_texture_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.glb",
        )
        scene_semantic_annotation_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.txt",
        )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        if not os.path.exists(semantic_texture_path) or not os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.warning(
                f"semantic_texture_path: {semantic_texture_path} or scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"
            )

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            sim_cfg = make_semantic_cfg(sim_settings)
        else:
            sim_cfg = make_simple_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # load object classes
        # maintain a list of object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=scene_semantic_annotation_path,
            bg_classes=self.cfg_cg["bg_classes"],
            skip_bg=self.cfg_cg["skip_bg"],
            class_set=self.cfg["class_set"],
        )

        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.info(f"Load scene {scene_id} successfully with semantic texture")
        else:
            logging.info(f"Load scene {scene_id} successfully without semantic texture")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.img_width, cfg.img_height, cfg.hfov)

        # about scene graph
        self.objects: MapObjectDict[int, Dict] = (
            MapObjectDict()
        )  # object_id -> object item
        self.object_id_counter = 1

        self.snapshots: Dict[str, SnapShot] = {}  # image_path -> snapshot
        self.frames: Dict[str, SnapShot] = {}  # image_path -> all frames
        self.all_observations: Dict[str, np.ndarray] = (
            {}
        )  # image_path -> image, stores all actual observations at each step, used for querying vlm

        self.clustering = SceneHierarchicalClustering(
            min_sample_split=0,
            random_state=66,
        )

        # setup detection and segmentation models
        self.detection_model = detection_model
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())

        self.sam_predictor = sam_predictor

        self.clip_model = clip_model.to(self.device)
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

    def __del__(self):
        try:
            self.simulator.close()
        except:
            pass

    def clear_up_detections(self):
        self.objects = MapObjectDict()
        self.object_id_counter = 1

        self.snapshots = {}
        self.frames = {}
        self.all_observations = {}

    def get_observation(self, pts, angle):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0)
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs

    def get_frontier_observation_and_detect_target(
        self,
        pts,
        view_dir,
        detection_model,
        target_obj_id,
        target_obj_class,
        camera_tilt=0.0,
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        detection_model.set_classes([target_obj_class])
        results = detection_model.infer(
            rgb[..., :3], confidence=self.cfg.scene_graph.confidence
        )
        detections = sv.Detections.from_inference(results).with_nms(
            threshold=self.cfg.scene_graph.nms_threshold
        )

        target_detected = False
        if target_obj_id in np.unique(semantic_obs):
            for i in range(len(detections)):
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True

                target_x_start, target_y_start = np.argwhere(
                    semantic_obs == target_obj_id
                ).min(axis=0)
                target_x_end, target_y_end = np.argwhere(
                    semantic_obs == target_obj_id
                ).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[target_x_start:target_x_end, target_y_start:target_y_end] = (
                    True
                )
                if IoU(bbox_mask, obj_mask) > self.cfg.scene_graph.iou_threshold:
                    target_detected = True
                    break

        return obs, target_detected

    def get_navigable_point_to(
        self,
        target_position,
        max_search=1000,
        min_dist=6.0,
        max_dist=999.0,
        prev_start_positions=None,
    ):
        self.pathfinder.seed(random.randint(0, 1000000))
        return get_navigable_point_to(
            target_position,
            self.pathfinder,
            max_search,
            min_dist,
            max_dist,
            prev_start_positions,
        )

    def update_scene_graph(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path,
        frame_idx,
        target_obj_mask=None,  # the boolean mask of target object generated from the semantic sensor. If given, return the object id of the target object
    ) -> Tuple[np.ndarray, List[int], Optional[int]]:
        # return annotated image; the detected object ids in current frame; the object id of the target object (if detected)

        # set up object_classes first
        obj_classes = self.obj_classes

        # Detect objects
        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # if there are detections,
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        if xyxy_tensor.numel() != 0:
            sam_out = self.sam_predictor.predict(
                image_rgb, bboxes=xyxy_tensor, verbose=False
            )
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        # Create a detections object that we will save later
        curr_det = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )

        if len(curr_det) == 0:  # no detections, skip
            logging.debug("No detections in this frame")
            return image_rgb, [], None

        # filter the detection by removing overlapping detections
        curr_det, labels = filter_detections(
            image=image_rgb,
            detections=curr_det,
            classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=self.cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=self.cfg_cg.min_mask_size_ratio,
            confidence_threshold=self.cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            logging.debug("No detections left after filter_detections")
            return image_rgb, [], None

        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            curr_det,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            obj_classes.get_classes_arr(),
            self.device,
        )

        raw_gobs = {
            # add new uuid for each detection
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
        }

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(
            resized_gobs,
            image_rgb,
            skip_bg=self.cfg_cg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=self.cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=self.cfg_cg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs["mask"]) == 0:  # no detections in this frame
            logging.debug("No detections left after filter_gobs")
            return image_rgb, [], None

        # this helps make sure things like pillows on couches are separate objects
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg_cg.min_points_threshold,
            spatial_sim_type=self.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=self.cfg_cg.obj_pcd_max_points,
            device=self.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )
        # if the list is all None, then skip
        if all([obj is None for obj in obj_pcds_and_bboxes]):
            logging.debug("All objects are None in obj_pcds_and_bboxes")
            return image_rgb, [], None

        # add pcds and bboxes to gobs
        gobs["bbox"] = [
            obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]
        gobs["pcd"] = [
            obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]

        # filter out objects that are far away
        gobs = self.filter_gobs_with_distance(pts, gobs)

        detection_list = self.make_detection_list_from_pcd_and_gobs(
            gobs, img_path, obj_classes
        )

        if len(detection_list) == 0:  # no detections, skip
            logging.debug(
                "No detections left after make_detection_list_from_pcd_and_gobs"
            )
            return image_rgb, [], None

        # compare the detections with the target object mask to see whether the target object is detected
        target_obj_id = None
        if (
            target_obj_mask is not None
            and np.sum(target_obj_mask)
            / (target_obj_mask.shape[0] * target_obj_mask.shape[1])
            > 0.0001
        ):
            assert len(detection_list) == len(
                gobs["mask"]
            ), f"Error in update_scene_graph: {len(detection_list)} != {len(gobs['mask'])}"  # sanity check

            # loop through the detected objects to find the highest IoU with the target object
            max_iou = -1
            max_iou_obj_id = None
            for idx, obj_id in enumerate(detection_list.keys()):
                detected_mask = gobs["mask"][idx]
                iou_score = IoU(detected_mask, target_obj_mask)
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_obj_id = obj_id
            if max_iou > self.cfg.scene_graph.target_obj_iou_threshold:
                target_obj_id = max_iou_obj_id
                logging.info(
                    f"Target object {target_obj_id} {detection_list[target_obj_id]['class_name']} detected with IoU {max_iou} in {img_path}!!!"
                )

        # if there exists object detected in this frame, create a snapshot
        frame = SnapShot(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
        )
        # add all detected objects into the snapshot
        frame.full_obj_list = {
            obj_id: detection_list[obj_id]["conf"] for obj_id in detection_list.keys()
        }

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(self.objects) == 0:
            logging.debug(
                f"No objects in the map yet, adding all detections of length {len(detection_list)}"
            )
            self.objects.update(detection_list)

            self.frames[img_path] = frame

            annotated_image = image_rgb
            added_obj_ids = list(detection_list.keys())
        else:
            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                detection_list=detection_list,
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
            )

            visual_sim = compute_visual_similarities(detection_list, self.objects)

            agg_sim = aggregate_similarities(
                match_method=self.cfg_cg["match_method"],
                phys_bias=self.cfg_cg["phys_bias"],
                spatial_sim=spatial_sim,
                visual_sim=visual_sim,
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=self.cfg_cg[
                    "sim_threshold"
                ],  # Use the sim_threshold from the configuration
                existing_obj_ids=list(self.objects.keys()),
                detected_obj_ids=list(detection_list.keys()),
            )

            # Now merge the detected objects into the existing objects based on the match indices
            visualize_captions, target_obj_id, added_obj_ids = self.merge_obj_matches(
                detection_list=detection_list,
                match_indices=match_indices,
                obj_classes=obj_classes,
                snapshot=frame,
                target_obj_id=target_obj_id,
            )

            # add the snapshot into the snapshot list
            self.frames[img_path] = frame

            # create a Detection object for visualization
            det_visualize = sv.Detections(
                xyxy=gobs["xyxy"],
                confidence=gobs["confidence"],
                class_id=gobs["class_id"],
            )
            det_visualize.data["class_name"] = visualize_captions
            annotated_image = image_rgb.copy()
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
            LABEL_ANNOTATOR = sv.LabelAnnotator(
                text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK
            )
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
                annotated_image, det_visualize
            )
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, det_visualize)

        return annotated_image, added_obj_ids, target_obj_id

    def filter_gobs_with_distance(self, pts, gobs):
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # get the distance between the object and the current observation point
            if (
                np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]])
                > self.cfg.scene_graph.obj_include_dist
            ):
                logging.debug(
                    f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping"
                )
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs

    def merge_obj_matches(
        self,
        detection_list: DetectionDict,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        snapshot: SnapShot,
        target_obj_id: Optional[
            int
        ] = None,  # if given, then track whether the target object is merged into a previous object (so the id would change)
    ) -> Tuple[List[str], Optional[int], List[int]]:
        visualize_captions = []
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )
                added_obj_ids.append(detected_obj_id)
            else:
                # merge detected object into existing object
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]

                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    device=self.device,
                    run_dbscan=False,
                )
                # fix the class name by adopting the most popular class name
                class_id_counter = Counter(merged_obj["class_id"])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[
                    most_common_class_id
                ]
                merged_obj["class_name"] = most_common_class_name

                # adjust the full detected list of the current snapshot: remove the detected object and add the merged object
                snapshot.full_obj_list[existing_obj_match_id] = detected_obj["conf"]
                snapshot.full_obj_list.pop(detected_obj_id)

                self.objects[existing_obj_match_id] = merged_obj
                visualize_captions.append(
                    f"{existing_obj_match_id} {self.objects[existing_obj_match_id]['class_name']} {detected_obj['conf']:.3f} {merged_obj['num_detections']}"
                )

                # if current object is the target object, and it is merged into an existing object, then change the target object id to the existing object id
                if target_obj_id == detected_obj_id:
                    target_obj_id = existing_obj_match_id

        return visualize_captions, target_obj_id, added_obj_ids

    def make_detection_list_from_pcd_and_gobs(
        self, gobs, image_path, obj_classes
    ) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:  # point cloud was discarded
                continue

            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)

            detected_object = {
                "id": self.object_id_counter,  # unique id for this object
                "class_name": curr_class_name,  # global class id for this detection
                "class_id": [curr_class_idx],  # global class id for this detection
                "num_detections": 1,  # number of detections in this object
                "conf": gobs["confidence"][mask_idx],
                # These are for the entire 3D object
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                # the snapshot name it belongs to
                "image": None,
            }

            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1

        return detection_list

    def cleanup_empty_frames_snapshots(self):
        # remove the frame that have empty detected objects
        filtered_frames = {}
        for file_name, frame in self.frames.items():
            if len(frame.full_obj_list) > 0:
                filtered_frames[file_name] = frame
        self.frames = filtered_frames

        # remove the snapshots that have no cluster
        filtered_snapshots = {}
        for file_name, snapshot in self.snapshots.items():
            if len(snapshot.cluster) > 0:
                filtered_snapshots[file_name] = snapshot
        self.snapshots = filtered_snapshots

    def update_snapshots(
        self,
        obj_ids,
        min_detection=2,
    ):
        self.cleanup_empty_frames_snapshots()

        prev_snapshots = copy.deepcopy(self.snapshots)

        obj_ids_temp = obj_ids.copy()
        for filename, snapshot in self.snapshots.items():
            cluster = snapshot.cluster
            if any([obj_id in obj_ids_temp for obj_id in cluster]):
                obj_ids = obj_ids.union(set(cluster))
                prev_snapshots.pop(filename)
        obj_ids = list(set(obj_ids))

        # find and exclude the objects that have only one observation
        obj_exclude = [
            obj_id
            for obj_id in self.objects.keys()
            if self.objects[obj_id]["num_detections"] < min_detection
        ]
        obj_ids = [obj_id for obj_id in obj_ids if obj_id not in obj_exclude]

        obj_centers = np.zeros((len(obj_ids), 2))
        for i, obj_id in enumerate(obj_ids):
            obj_centers[i] = self.objects[obj_id]["bbox"].center[[0, 2]]

        if len(obj_centers) == 0:
            return

        new_snapshots = self.clustering.fit(obj_centers, obj_ids, self.frames)

        prev_snapshot_objs = [
            obj_id
            for snapshot in prev_snapshots.values()
            for obj_id in snapshot.cluster
        ]
        assert set(
            [
                obj_id
                for snapshot in new_snapshots.values()
                for obj_id in snapshot.cluster
            ]
        ) == set(
            obj_ids
        ), f"{set([obj_id for snapshot in new_snapshots.values() for obj_id in snapshot.cluster])} != {set(obj_ids)}"
        assert (
            set(obj_ids) & set(prev_snapshot_objs)
        ) == set(), f"{set(obj_ids)} & {set(prev_snapshot_objs)} != empty"
        assert (set(obj_ids) | set(prev_snapshot_objs) | set(obj_exclude)) == set(
            self.objects.keys()
        ), f"{set(obj_ids)} | {set(prev_snapshot_objs)} | {set(obj_exclude)} != {set(self.objects.keys())}"

        for key, snapshot in new_snapshots.items():
            if key in prev_snapshots.keys():
                prev_snapshots[key].cluster += snapshot.cluster
            else:
                prev_snapshots[key] = snapshot
        self.snapshots = prev_snapshots

        # update the snapshot belonging of each object
        for file_name, snapshot in self.snapshots.items():
            for obj_id in snapshot.cluster:
                self.objects[obj_id]["image"] = file_name

        # remove the duplicates caused by copying snapshots: self.frames and self.snapshots should point to the same object
        for file_name, snapshot in self.snapshots.items():
            self.frames[file_name] = snapshot

        # sanity check
        for obj_id, obj in self.objects.items():
            if obj["num_detections"] < min_detection:
                assert (
                    obj["image"] is None
                ), f"{obj_id} has only one detection but has image"
            else:
                assert obj["image"] is not None, f"{obj_id} has no image"

    def periodic_cleanup_objects(self, frame_idx, pts):
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            self.cfg_cg["denoise_interval"],
            self.cfg_cg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                objects=self.objects,
            )

        # Filtering
        if processing_needed(
            self.cfg_cg["filter_interval"],
            self.cfg_cg["run_filter_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg["obj_min_points"],
                obj_min_detections=self.cfg_cg["obj_min_detections"],
                min_distance=self.cfg.scene_graph.obj_include_dist,
                objects=self.objects,
                pts=pts,
            )

        # temporarily we do not merge close objects, since handling which snapshot the merged object belongs to is a bit tricky

        # Merging
        if processing_needed(
            self.cfg_cg["merge_interval"],
            self.cfg_cg["run_merge_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
            )

        # update the object list in snapshots, since some objects may have been removed
        frame_to_pop = []
        for (
            filename,
            ss,
        ) in (
            self.frames.items()
        ):  # TODO: check whether content in snapshots are also changed, and see whether need to remove snapshot that have empty cluster
            ss.cluster = [
                obj_id for obj_id in ss.cluster if obj_id in self.objects.keys()
            ]
            ss.full_obj_list = {
                obj_id: conf
                for obj_id, conf in ss.full_obj_list.items()
                if obj_id in self.objects.keys()
            }
            if len(ss.full_obj_list) == 0:
                frame_to_pop.append(filename)
        for filename in frame_to_pop:
            self.frames.pop(filename)

    def sanity_check(self, cfg):
        obj_exclude_count = sum(
            [
                1 if obj["num_detections"] < cfg.min_detection else 0
                for obj in self.objects.values()
            ]
        )
        total_objs_count = sum(
            [len(snapshot.cluster) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        total_objs_count = sum(
            [len(set(snapshot.cluster)) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        for obj_id in self.objects.keys():
            exist_count = 0
            for ss in self.snapshots.values():
                if obj_id in ss.cluster:
                    exist_count += 1
            if self.objects[obj_id]["num_detections"] < cfg.min_detection:
                assert (
                    exist_count == 0
                ), f"{exist_count} != 0 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
            else:
                assert (
                    exist_count == 1
                ), f"{exist_count} != 1 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
        for ss in self.snapshots.values():
            assert len(ss.cluster) == len(
                set(ss.cluster)
            ), f"{ss.cluster} has duplicates"
            assert len(ss.full_obj_list.keys()) == len(
                set(ss.full_obj_list.keys())
            ), f"{ss.full_obj_list.keys()} has duplicates"
            for obj_id in ss.cluster:
                assert (
                    obj_id in ss.full_obj_list
                ), f"{obj_id} not in {ss.full_obj_list.keys()}"
            for obj_id in ss.full_obj_list.keys():
                assert obj_id in self.objects, f"{obj_id} not in scene objects"
        # check whether the snapshots in scene.snapshots and scene.frames are the same
        for file_name, ss in self.snapshots.items():
            assert (
                ss.cluster == self.frames[file_name].cluster
            ), f"{ss}\n!=\n{self.frames[file_name]}"
            assert (
                ss.full_obj_list == self.frames[file_name].full_obj_list
            ), f"{ss}\n==\n{self.frames[file_name]}"

    def print_scene_graph(self):
        snapshot_dict = {}
        for obj_id, obj in self.objects.items():
            if obj["image"] not in snapshot_dict:
                snapshot_dict[obj["image"]] = []
            snapshot_dict[obj["image"]].append(
                f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
            )
        for snapshot_id, obj_list in snapshot_dict.items():
            logging.info(f"{snapshot_id}:")
            for obj_str in obj_list:
                logging.info(f"\t{obj_str}")


# ── GroundingDINO Navigation Chain ──────────────────────────────────────

_gd_model = None
_gd_transform = None


def _load_gd_model(gd_dir=None, device="cuda"):
    """Lazy-load GroundingDINO Swin-T model and transform."""
    global _gd_model, _gd_transform
    if _gd_model is not None:
        return _gd_model, _gd_transform

    if gd_dir is None:
        from src.const import GROUNDINGDINO_DIR as gd_dir

    import sys
    if gd_dir not in sys.path:
        sys.path.insert(0, gd_dir)

    import groundingdino.datasets.transforms as gd_transforms
    from groundingdino.util.inference import load_model as gd_load_model

    from src.const import GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS
    config_path = os.path.join(gd_dir, GROUNDINGDINO_CONFIG)
    weights_path = os.path.join(gd_dir, GROUNDINGDINO_WEIGHTS)

    logging.info(f"Loading GroundingDINO from {config_path}, weights={weights_path}")
    _gd_model = gd_load_model(config_path, weights_path, device=device)

    _gd_transform = gd_transforms.Compose([
        gd_transforms.RandomResize([800], max_size=1333),
        gd_transforms.ToTensor(),
        gd_transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
    ])
    return _gd_model, _gd_transform


def _gd_detect(rgb, prompt, box_threshold=0.30, text_threshold=0.25):
    """Run GroundingDINO on a single RGB image.

    Returns: (best_bbox_xyxy, best_phrase, best_score, image_pil) or
             (None, None, 0, None) if nothing detected.
    """
    from groundingdino.util.inference import predict as gd_predict

    gd_model, gd_transform = _load_gd_model()
    image_pil = Image.fromarray(rgb)
    W, H = image_pil.size
    image_tensor, _ = gd_transform(image_pil, None)

    with torch.no_grad():
        boxes, logits, phrases = gd_predict(
            model=gd_model, image=image_tensor, caption=prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
            device="cuda",
        )

    if len(boxes) == 0:
        return None, None, 0, image_pil

    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
    boxes_xyxy *= torch.tensor([W, H, W, H])

    top1 = int(torch.argmax(logits).item())
    score = float(logits[top1].item())
    bbox = boxes_xyxy[top1].cpu().numpy()
    phrase = phrases[top1]

    return bbox, phrase, score, image_pil


def grounded_navigate_to_object(
    scene, tsdf_planner, pts, angle,
    view_idx, view_angle, view_cam_pose,  # VLM-selected view
    object_desc,
    max_steps=20, gd_dir=None,
    max_consecutive_failures=5,
    max_iterations=5, converge_dist_voxels=5,
    max_nav_steps_per_iter=15,
    memory_store=None, cam_intr_ext=None, cfg_ext=None,
    detection_model=None, sam_predictor=None,
    clip_model=None, clip_preprocess=None, clip_tokenizer=None,
    cnt_step_base=0, step_budget=None,
    gd_model=None,
):
    """GD 导航链：VLM 选定视角 → GD 检测 → 3D 反投影 → 迭代螺旋搜索导航。

    视角由 VLM 在 Stage 2 选定（view_idx + view_angle + view_cam_pose）。
    代码不做视角扫描，只用 VLM 选定的那张图做 GD 检测。

    Key fixes from HM-GE stage-3 notes:
    - trans_pose = pose_habitat_to_tsdf(cam_pose)  (NOT raw Habitat cam_pose)
    - camera_convention = "z_forward"  (matches TSDF pose)
    - Result pcd is already in normal coords; no pos_habitat_to_normal conversion
    - Z-clip: normal[2] > 3.0m → pin to floor
    - Target = pcd mean (x,y), pin normal[2] to floor

    Returns: (new_pts, new_angle, success_bool, status_text, images_list)
    """
    from PIL import Image
    from src.habitat import pos_habitat_to_normal, pos_normal_to_habitat, pose_habitat_to_tsdf
    from src.conceptgraph.slam.utils import (
        detections_to_obj_pcd_and_bbox,
        init_process_pcd,
        get_bounding_box,
    )
    import open3d as o3d

    images = []
    cam_intr = cam_intr_ext if cam_intr_ext is not None else scene.cam_intrinsic
    cfg_cg = scene.cfg_cg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    floor_height = float(pts[1])
    cfg = cfg_ext if cfg_ext is not None else scene.cfg

    # ── Phase A: GD detect + SAM + 3D back-project → target_normal ──
    # Use VLM-selected view (view_angle / view_cam_pose)
    obs, cam_pose_habitat = scene.get_observation(pts, view_angle)
    rgb = obs["color_sensor"]
    depth = obs["depth_sensor"]

    bbox, phrase, score = _gd_detect(rgb, object_desc)
    if bbox is None:
        return pts, angle, False, f"GD no detection for '{object_desc}'", images

    logging.info(f"  GD: '{phrase}' score={score:.3f}")
    images.append(("gd_detection", rgb.copy()))

    # SAM mask
    try:
        sam_out = scene.sam_predictor.predict(
            rgb, bboxes=[bbox.tolist()], verbose=False)
        mask = sam_out[0].masks.data.cpu().numpy()[0].astype(bool)
    except Exception as e:
        logging.warning(f"  GD: SAM failed: {e}, using bbox as mask")
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x1, y1, x2, y2 = bbox.astype(int)
        mask[y1:y2, x1:x2] = True

    # 3D back-projection — STRICTLY follow debug_iterative_spiral_navigate.py
    # CRITICAL FIX: use TSDF pose + z_forward convention (not raw Habitat pose)
    cam_pose_tsdf = pose_habitat_to_tsdf(cam_pose_habitat)
    try:
        obj_list = detections_to_obj_pcd_and_bbox(
            depth_array=depth,
            masks=mask[None, :, :].astype(np.float32),
            cam_K=cam_intr,
            image_rgb=rgb,
            trans_pose=cam_pose_tsdf,            # TSDF pose (FIXED)
            camera_convention="z_forward",        # matches TSDF pose (FIXED)
            min_points_threshold=5,
            spatial_sim_type=cfg_cg.spatial_sim_type,
            obj_pcd_max_points=cfg_cg.obj_pcd_max_points,
            downsample_voxel_size=cfg_cg.get('downsample_voxel_size',
                cfg_cg.get('downsample_voxcel_size', 0.02))
                if isinstance(cfg_cg, dict)
                else getattr(cfg_cg, 'downsample_voxel_size',
                    getattr(cfg_cg, 'downsample_voxcel_size', 0.02)),
            device=device,
        )
    except Exception as e:
        logging.warning(f"  GD: back-project failed: {e}")
        return pts, angle, False, f"Back-projection failed: {e}", images

    if not obj_list or obj_list[0] is None:
        return pts, angle, False, "Back-projection returned None", images

    obj = obj_list[0]
    pcd = obj["pcd"]
    if len(pcd.points) == 0:
        return pts, angle, False, "Empty point cloud", images

    # pcd_np is already in normal coords (because we used TSDF pose + z_forward)
    pcd_np = np.asarray(pcd.points)
    logging.info(f"  GD: back-projected {pcd_np.shape[0]} points")

    # Z-clip: pin points above 3.0m to floor height (normal[2] is height)
    z_clip = cfg.tsdf_grid_size * 30  # 3.0m
    pcd_np_clipped = pcd_np.copy()
    over_z = pcd_np_clipped[:, 2] > z_clip
    if over_z.any():
        logging.info(f"  GD: z-clipping {int(over_z.sum())}/{len(pcd_np)} points")
        pcd_np_clipped[over_z, 2] = floor_height

    # Target = pcd mean (x,y) — OBB center gets pulled by z-clip
    target_normal = pcd_np_clipped.mean(axis=0)
    target_normal[2] = floor_height  # pin height to floor

    target_voxel = tsdf_planner.normal2voxel(target_normal)
    target_voxel_xy = (int(target_voxel[0]), int(target_voxel[1]))
    logging.info(
        f"  GD: target normal={target_normal.tolist()} "
        f"voxel={target_voxel.tolist()}")

    # ── Phase B: Iterative spiral search + per-step navigation ──
    # Port from debug_iterative_spiral_navigate.py lines 661-906
    # Per HM-GE stage-3 notes:
    #   1. Refresh grids, check if target on island
    #   2. Spiral search nearest navigable point to target
    #   3. set_next_navigation_point(target_type="image", ...)
    #   4. Per-step: agent_step → silent_perception → refresh → update_frontier
    #   5. On arrival: if converged → done; if not converged → re-spiral (continue)
    from src.agent_tools import silent_perception_step as gd_silent_step

    cur_pts, cur_angle = pts.copy(), angle
    converged = False
    arrived_any = False
    map_h, map_w = tsdf_planner._tsdf_vol_cpu.shape[:2]
    max_spiral_radius = max(map_h, map_w)

    for iteration in range(1, max_iterations + 1):
        if step_budget is not None and gd_silent_step._step_counter >= step_budget:
            logging.info(f"  GD iter {iteration}: step budget exhausted, stopping")
            break

        logging.info(f"  GD iter {iteration}/{max_iterations}, pts={cur_pts.tolist()}")

        # 1. Refresh grids from current TSDF state
        tsdf_planner.refresh_planner_grids(cur_pts)
        island_sum = int(tsdf_planner.island.sum()) if tsdf_planner.island is not None else 0
        logging.info(f"  GD iter {iteration}: island={island_sum} voxels")

        # 2. Check if target is already navigable (on island)
        spiral_result = None
        tv_y, tv_x = target_voxel_xy
        if (0 <= tv_y < map_h and 0 <= tv_x < map_w
                and tsdf_planner.island[tv_y, tv_x]):
            logging.info(f"  GD iter {iteration}: target on island, converging")
            normal_pos = tsdf_planner.voxel2normal(np.array([tv_y, tv_x]))
            normal_3d = np.array([normal_pos[0], normal_pos[1], floor_height])
            hab_pos = pos_normal_to_habitat(normal_3d)
            snapped = scene.pathfinder.snap_point(hab_pos[:3])
            if snapped is not None and not np.isnan(snapped).any():
                spiral_result = {
                    "habitat_pos": snapped,
                    "voxel_xy": target_voxel_xy,
                    "search_steps": 0,
                    "spiral_dist": 0,
                }
                converged = True

        # 3. Spiral search (if not already on island)
        if spiral_result is None:
            spiral_result = tsdf_planner.spiral_search_navigable_point(
                pathfinder=scene.pathfinder,
                target_voxel_xy=target_voxel_xy,
                agent_habitat=cur_pts,
                max_radius_voxels=max_spiral_radius,
                floor_height=floor_height,
            )
            if spiral_result is None:
                logging.warning(f"  GD iter {iteration}: spiral found nothing")
                break

        # 4. Convergence check
        spiral_dist = spiral_result["spiral_dist"]
        logging.info(
            f"  GD iter {iteration}: spiral dist={spiral_dist} "
            f"voxel={spiral_result['voxel_xy']}")
        if spiral_dist <= converge_dist_voxels:
            converged = True

        # 5. Set navigation point — use production code's target_type="image"
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
        pathfinder_target = np.array(spiral_result["habitat_pos"])
        set_ok = tsdf_planner.set_next_navigation_point(
            target_type="image",
            choice=pathfinder_target,
            pts=cur_pts.tolist(),
            objects=None, obs_points=None,
            cfg=cfg.planner, pathfinder=scene.pathfinder,
            random_position=False, observe_snapshot=False,
        )
        if not set_ok:
            logging.error(f"  GD iter {iteration}: set_next_navigation_point failed")
            break

        # 6. Per-step navigation loop
        arrived = False
        for nav_step in range(1, max_nav_steps_per_iter + 1):
            if step_budget is not None and gd_silent_step._step_counter >= step_budget:
                break

            result = tsdf_planner.agent_step(
                pts=cur_pts, angle=cur_angle,
                objects=scene.objects, snapshots=scene.snapshots,
                pathfinder=scene.pathfinder, cfg=cfg.planner,
                save_visualization=False,
            )
            if result[0] is None:
                logging.warning(f"  GD iter {iteration} step {nav_step}: agent_step failed")
                break

            cur_pts, cur_angle, _, _, _, target_arrived = result

            # Per-step silent perception (scene graph + TSDF + snapshot)
            gd_silent_step(
                scene, tsdf_planner, cur_pts, cur_angle,
                cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
                memory_store, cam_intr, cfg,
                detection_model, sam_predictor,
                clip_model, clip_preprocess, clip_tokenizer,
            )

            # Refresh grids (map grows with new observations)
            tsdf_planner.refresh_planner_grids(cur_pts)

            # Update frontier map (includes room segmentation)
            try:
                tsdf_planner.update_frontier_map(
                    cur_pts, cfg.planner, scene,
                    cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
                    save_frontier_image=False,
                )
            except Exception as e:
                logging.warning(f"  GD iter {iteration} step {nav_step}: update_frontier_map failed: {e}")

            logging.info(
                f"  GD iter {iteration} step {nav_step}: "
                f"voxel={tsdf_planner.habitat2voxel(cur_pts)[:2].tolist()} "
                f"arrived={target_arrived}")

            if target_arrived:
                arrived = True
                break

        arrived_any = arrived or arrived_any

        # Clear navigation state for next iteration
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None

        # 7. Iteration termination logic
        if converged and arrived:
            logging.info(f"  GD: converged and arrived at iteration {iteration}")
            break
        if arrived and not converged:
            logging.info(f"  GD iter {iteration}: arrived but not converged, re-spiraling")
            continue
        logging.info(f"  GD iter {iteration}: didn't arrive, continuing from current pos")

    status = f"GD nav: {'converged' if converged else 'not converged'}, arrived={arrived_any}"
    return cur_pts, cur_angle, arrived_any, status, images
