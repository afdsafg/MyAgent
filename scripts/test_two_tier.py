"""End-to-end test for two-tier Planner-Executor refactor."""
import os, sys, json, time, logging
import numpy as np, torch, random
from omegaconf import OmegaConf
import open_clip
from ultralytics import SAM, YOLOWorld

sys.path.insert(0, "/root/MyAgent")
from src.agent_workflow import run_episode_two_tier

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

cfg = OmegaConf.load("/root/MyAgent/cfg/eval_aeqa.yaml")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load models
logger.info("Loading detection models...")
detection_model = YOLOWorld(cfg.yolo_model_name)
sam_predictor = SAM(cfg.sam_model_name)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

scene_id = "00824-Dd4bFSTQ8gi"
question = "What is hanging from the oven handle?"
question_id = "00c2be2a-1377-4fae-a889-30936b7890c3"
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
)

elapsed = time.time() - start_time
result["elapsed_seconds"] = elapsed
logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
print(json.dumps(result, indent=2, default=str))
