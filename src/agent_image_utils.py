"""图像工具：编码、拼接、转换。"""
import base64
import io
import numpy as np
from PIL import Image


def numpy_to_base64(img: np.ndarray, fmt="JPEG") -> str:
    """RGB uint8 numpy array → base64 string."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_mosaic(images, cols=None, target_h=240, padding=4):
    """拼接多张 RGB 图像为一张网格图。

    Args:
        images: list of (H,W,3) numpy uint8 arrays
        cols: 列数，None 则自动计算最接近正方形
        target_h: 每行统一高度
        padding: 图像间距像素
    Returns: (H,W,3) numpy uint8 mosaic
    """
    if not images:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)
    n = len(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    scaled = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(target_h * w / h)
        pil = Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS)
        scaled.append(np.array(pil))
    max_w = max(s.shape[1] for s in scaled)
    canvas_h = rows * (target_h + padding) + padding
    canvas_w = cols * (max_w + padding) + padding
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
    for i, img in enumerate(scaled):
        r, c = i // cols, i % cols
        y = r * (target_h + padding) + padding
        x = c * (max_w + padding) + padding
        h, w = img.shape[:2]
        x_offset = (max_w - w) // 2
        canvas[y:y + h, x + x_offset:x + x_offset + w] = img
    return canvas


def fig_to_base64(fig) -> str:
    """Matplotlib figure → base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
