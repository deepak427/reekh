"""RF-DETR Medium inference wrapper — loaded once at process startup."""
from __future__ import annotations

import os
import time

import numpy as np

# Must check before importing cv2.
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    raise RuntimeError("Unset OPENCV_FORCE_DNN_ENGINE before starting the service.")

import cv2

# Support both ENGINE_OPENCV (5.x mainline) and ENGINE_NEW (5.0.0 wheel).
if hasattr(cv2.dnn, "ENGINE_OPENCV"):
    _DNN_ENGINE  = cv2.dnn.ENGINE_OPENCV
    _ENGINE_NAME = "ENGINE_OPENCV"
elif hasattr(cv2.dnn, "ENGINE_NEW"):
    _DNN_ENGINE  = cv2.dnn.ENGINE_NEW
    _ENGINE_NAME = "ENGINE_NEW"
else:
    raise RuntimeError("Requires OpenCV 5's built-in DNN engine.")

CLASS_NAMES = [
    "AmurTiger", "Badger", "BlackBear", "Cow", "Dog", "Hare",
    "Leopard", "LeopardCat", "MuskDeer", "RaccoonDog", "RedFox",
    "RoeDeer", "Sable", "SikaDeer", "Weasel", "WildBoar", "Y-T-Marten",
]
NUM_CLASSES = len(CLASS_NAMES)   # 17
NUM_LOGITS  = NUM_CLASSES + 1    # 18 — index 17 is the unused background slot
INPUT_SIZE  = 576
NUM_QUERIES = 300


class RFDETRInference:
    """Wraps a loaded OpenCV DNN net.

    OpenCV DNN is NOT thread-safe for concurrent forward calls on the same
    net object.  This is not an issue here: the single background worker
    processes frames sequentially, so only one forward pass is ever in
    flight at a time.
    """

    def __init__(self, model_path: str) -> None:
        self._net = cv2.dnn.readNetFromONNX(model_path, _DNN_ENGINE)
        self.engine_name = _ENGINE_NAME
        # Warmup: one dummy forward pass so the first real frame isn't slow.
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self._net.setInput(dummy)
        self._net.forward(["dets", "labels"])

    def run(
        self,
        image: np.ndarray,
        confidence: float = 0.45,
    ) -> tuple[list[dict], float]:
        """Run inference on a BGR image (as returned by cv2.imdecode).

        Returns:
            detections:   list of dicts — class_id, class_name, score,
                          bbox [x1,y1,x2,y2], bbox_center [cx,cy].
            inference_ms: wall-clock time of the forward pass in ms.
        """
        blob = self._preprocess(image)
        t0   = time.perf_counter()
        self._net.setInput(blob)
        dets, logits = self._net.forward(["dets", "labels"])
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return self._decode(dets, logits, image.shape, confidence), inference_ms

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Direct square resize + ImageNet normalisation.  No letterbox."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
              np.array([0.229, 0.224, 0.225], np.float32)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    @staticmethod
    def _decode(
        dets:        np.ndarray,
        logits:      np.ndarray,
        image_shape: tuple,
        threshold:   float,
    ) -> list[dict]:
        """Decode RF-DETR outputs into detection dicts."""
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -80, 80)))  # (300, 18)
        flat   = scores.ravel()
        chosen = np.argsort(-flat)[:NUM_QUERIES]

        h, w  = image_shape[:2]
        scale = np.array([w, h, w, h], np.float32)
        results: list[dict] = []

        for index in chosen:
            score = float(flat[index])
            if score < threshold:
                break  # argsorted descending — all remaining are also below threshold
            query, logit_id = divmod(int(index), NUM_LOGITS)
            if logit_id >= NUM_CLASSES:
                continue  # background slot
            cx, cy, bw, bh = dets[0, query]
            box = np.array([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]) * scale
            box = np.clip(box, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "class_id":    logit_id,
                "class_name":  CLASS_NAMES[logit_id],
                "score":       score,
                "bbox":        [x1, y1, x2, y2],
                "bbox_center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            })

        return results  # RF-DETR one-to-one head: no NMS needed
