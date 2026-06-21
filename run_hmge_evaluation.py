import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.agent_workflow import run_episode
from src.logger_aeqa import Logger
from src.const import *


def main(cfg, start_ratio=0.0, end_ratio=1.0):
    logging.info(f"***** Running HM-GE AEQA Evaluation *****")

    # Load data
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    questions_list = questions_list[
        int(start_ratio * total_questions):int(end_ratio * total_questions)
    ]
    logging.info(f"Number of questions after splitting: {len(questions_list)}")

    # Load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio,
        len(questions_list), voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]

        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
        if any([invalid_scene_id in scene_id
                for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue

        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")
        question = question_data["question"]
        answer = question_data["answer"]

        # Initialize episode directory
        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=(0, 0),  # Workflow handles this
            )
        )

        logging.info(f"Question id {question_id} initialization successful!")
        logging.info(f"Question: {question}")
        logging.info(f"Ground truth answer: {answer}")

        # ── Run HM-GE Workflow ────────────────────────────────────────
        try:
            result = run_episode(
                scene_id=scene_id,
                question=question,
                question_id=question_id,
                cfg=cfg,
                detection_model=detection_model,
                sam_predictor=sam_predictor,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                output_dir=os.path.join(cfg.output_dir, "episodes"),
                max_total_steps=cfg.get("hmge_max_steps", 50),
            )
        except Exception as e:
            logging.error(f"Episode {question_id} failed with error: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "scene_id": scene_id,
                "question_id": question_id,
                "question": question,
                "answer": "",
                "success": False,
                "steps_taken": 0,
                "stages_completed": 0,
                "error": str(e),
            }

        # Log result
        task_success = result.get("success", False)
        gpt_answer = result.get("answer", "")
        steps_taken = result.get("steps_taken", 0)
        stages_completed = result.get("stages_completed", 0)
        error = result.get("error", "")

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=0.0,
            gpt_answer=gpt_answer,
            n_filtered_snapshots=0,
            n_total_snapshots=0,
            n_total_frames=0,
        )

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        logging.info(f"Steps: {steps_taken}, Stages: {stages_completed}")
        if error:
            logging.info(f"Error: {error}")

        # Save results after each episode
        logger.save_results()

        if not cfg.get("save_visualization", False):
            os.system(f"rm -r {episode_dir}")

    logger.save_results()
    logger.aggregate_results()
    logging.info(f"All scenes finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path",
                       default="cfg/eval_aeqa.yaml", type=str)
    parser.add_argument("--start_ratio", help="start ratio",
                       default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio",
                       default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name + "_hmge")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log")

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    logging.info(f"***** Running HM-GE AEQA Evaluation *****")
    main(cfg, args.start_ratio, args.end_ratio)
