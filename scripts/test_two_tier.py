"""End-to-end test for two-tier Planner-Executor refactor."""
import os, sys, json, time, logging

# ── Fix 1: HuggingFace offline mode ──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np, torch, random
from omegaconf import OmegaConf
import open_clip
from ultralytics import SAM, YOLOWorld

sys.path.insert(0, "/root/MyAgent")
from src.agent_workflow import run_episode_two_tier
from src.utils import get_pts_angle_aeqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

cfg = OmegaConf.load("/root/MyAgent/cfg/eval_aeqa.yaml")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Fix 2: Correct start position from A-EQA question data ──
questions_file = "data/aeqa_questions-41.json"
question_id = "00c2be2a-1377-4fae-a889-30936b7890c3"
question_data = next(
    q for q in json.load(open(questions_file, "r"))
    if q["question_id"] == question_id
)
scene_id = question_data["episode_history"]
question = question_data["question"]

pts, angle = get_pts_angle_aeqa(
    question_data["position"], question_data["rotation"]
)
logger.info(f"A-EQA start position: pts={pts}, angle={angle:.4f}")

# Load models
logger.info("Loading detection models...")
detection_model = YOLOWorld(cfg.yolo_model_name)
sam_predictor = SAM(cfg.sam_model_name)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

output_dir = "/root/MyAgent/results_two_tier/test_001"

logger.info(f"Starting two-tier test: scene={scene_id}, question={question}")
start_time = time.time()

result = run_episode_two_tier(
    scene_id=scene_id,
    question=question,
    question_id=question_id,
    cfg=cfg,
    detection_model=detection_model,
    sam_predictor=sam_predictor,
    clip_model=clip_model,
    clip_preprocess=clip_preprocess,
    clip_tokenizer=clip_tokenizer,
    output_dir=output_dir,
    max_planner_rounds=10,
    max_total_steps=50,
    start_pts=pts,
    start_angle=angle,
)

elapsed = time.time() - start_time
result["elapsed_seconds"] = elapsed
result["start_position"] = pts.tolist()
result["start_angle"] = float(angle)
logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
print(json.dumps(result, indent=2, default=str))
