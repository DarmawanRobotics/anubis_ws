#!/usr/bin/env python3
import math
import time
import json

import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  -- inisialisasi CUDA context otomatis

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
import os

try:
    DEFAULT_ENGINE_PATH = os.path.join(
        get_package_share_directory('anubis_perception'),
        'models', 'yolo11s.engine'
    )
except Exception:
    DEFAULT_ENGINE_PATH = "yolo11s.engine"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4

PERSON_CLASS = 0
BAG_CLASSES = {24, 26, 28}
TARGET_CLASSES = {PERSON_CLASS, *BAG_CLASSES}
CLASS_NAMES = {0: "Person", 24: "Backpack", 26: "Handbag", 28: "Suitcase"}

MAX_MATCH_DIST = 80
TRACK_TIMEOUT = 15

DIST_THRESHOLD = 200
TIME_THRESHOLD = 2
ALARM_HOLD_TIME = 5

COLOR_PERSON = (255, 200, 0)
COLOR_BAG_NORMAL = (0, 255, 0)
COLOR_ALARM = (0, 0, 255)

# ============================================================
#  Preprocess / Postprocess (sama seperti versi ONNX)
# ============================================================


def letterbox_resize(img, target_size=640):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


def preprocess_frame(frame, target_size=640):
    orig_h, orig_w = frame.shape[:2]
    frame_resized, scale, pad_left, pad_top = letterbox_resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_input = np.transpose(frame_normalized, (2, 0, 1))
    frame_input = np.ascontiguousarray(np.expand_dims(frame_input, axis=0))
    return frame_input, scale, pad_left, pad_top, orig_h, orig_w


def postprocess(
    raw_output,
    scale,
    pad_left,
    pad_top,
    orig_h,
    orig_w,
    conf_thresh=CONF_THRESHOLD,
    nms_thresh=0.45,
    allowed_classes=TARGET_CLASSES,
):
    """Untuk engine RAW output [1, 84, 8400] (bukan nms=True export).
    84 = 4 (xywh) + 80 (class scores COCO)."""
    output = np.squeeze(raw_output, axis=0)  # [84, 8400]
    output = output.T  # [8400, 84]

    boxes_xywh = output[:, :4]
    scores_all = output[:, 4:84]  # 80 class scores

    class_ids = np.argmax(scores_all, axis=1)
    max_scores = np.max(scores_all, axis=1)

    mask = (max_scores >= conf_thresh) & np.isin(class_ids, list(allowed_classes))
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # xywh (skala INPUT_SIZE) -> xyxy skala original image
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_left) / scale
    boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_top) / scale
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_left) / scale
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_top) / scale

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    # NMS (cv2.dnn.NMSBoxes butuh format [x, y, w, h])
    boxes_wh = boxes_xyxy.copy()
    boxes_wh[:, 2] -= boxes_wh[:, 0]
    boxes_wh[:, 3] -= boxes_wh[:, 1]

    indices = cv2.dnn.NMSBoxes(
        boxes_wh.tolist(), max_scores.tolist(), conf_thresh, nms_thresh
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten() if not isinstance(indices, tuple) else indices[0]

    detections = []
    for idx in indices:
        detections.append(
            {
                "box": boxes_xyxy[idx].astype(np.int32),
                "score": float(max_scores[idx]),
                "class_id": int(class_ids[idx]),
            }
        )
    return detections


class SimpleTracker:
    def __init__(self, max_dist=MAX_MATCH_DIST, timeout=TRACK_TIMEOUT):
        self.max_dist = max_dist
        self.timeout = timeout
        self.tracks = {}
        self.next_id = 0

    def update(self, detections, frame_id):
        unmatched = list(range(len(detections)))
        used = set()

        for idx in list(unmatched):
            det = detections[idx]
            x1, y1, x2, y2 = det["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            best_tid, best_dist = None, self.max_dist + 1
            for tid, tr in self.tracks.items():
                if tid in used or tr["cid"] != det["class_id"]:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid].update(
                    {
                        "cx": cx,
                        "cy": cy,
                        "box": det["box"],
                        "cid": det["class_id"],
                        "last_seen": frame_id,
                    }
                )
                used.add(best_tid)
                det["track_id"] = best_tid
                unmatched.remove(idx)

        for idx in unmatched:
            det = detections[idx]
            x1, y1, x2, y2 = det["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "cx": cx,
                "cy": cy,
                "box": det["box"],
                "cid": det["class_id"],
                "last_seen": frame_id,
            }
            det["track_id"] = tid

        for tid in list(self.tracks.keys()):
            if frame_id - self.tracks[tid]["last_seen"] > self.timeout:
                del self.tracks[tid]

        return detections


def draw_box(frame, xyxy, color, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


# ============================================================
#  TensorRT engine wrapper
# ============================================================
class TRTEngine:
    def __init__(self, engine_path, input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE)):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]

        # cari nama input, set shape (untuk dynamic-shape engine)
        input_name = next(
            n
            for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.context.set_input_shape(input_name, input_shape)

        self.inputs, self.outputs = [], []
        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            entry = {
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype,
            }

            # daftarkan address device ke context (wajib di v3 API)
            self.context.set_tensor_address(name, int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

        print(
            f"[TRT] Input '{self.inputs[0]['name']}' shape: {self.inputs[0]['shape']}"
        )
        for o in self.outputs:
            print(
                f"[TRT] Output '{o['name']}' shape: {o['shape']}, dtype: {o['dtype']}"
            )

    def infer(self, input_data):
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        return [out["host"].reshape(out["shape"]) for out in self.outputs]


# ============================================================
#  ROS2 Node
# ============================================================
class AbandonedDetectionNode(Node):
    def __init__(self):
        super().__init__("abandoned_detection_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("engine_path", DEFAULT_ENGINE_PATH)
        self.declare_parameter("dist_threshold", float(DIST_THRESHOLD))
        self.declare_parameter("time_threshold", float(TIME_THRESHOLD))

        image_topic = self.get_parameter("image_topic").value
        engine_path_param = self.get_parameter('engine_path').value
        engine_path = engine_path_param if engine_path_param else DEFAULT_ENGINE_PATH
        self.dist_threshold = self.get_parameter("dist_threshold").value
        self.time_threshold = self.get_parameter("time_threshold").value

        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        self.trt_engine = TRTEngine(engine_path)
        self.bridge = CvBridge()

        self.tracker = SimpleTracker()
        self.away_start = {}
        self.abandoned_flags = {}
        self.alarm_hold_time = {}
        self.frame_id = 0
        self.total_infer = 0.0

        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.pub_annotated = self.create_publisher(
            Image, "/abandoned_detection/image_annotated", 10
        )
        self.pub_alarm = self.create_publisher(Bool, "/abandoned_detection/alarm", 10)
        self.pub_detail = self.create_publisher(
            String, "/abandoned_detection/detail", 10
        )

        self.get_logger().info(f"Subscribed to: {image_topic}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        self.frame_id += 1
        display = frame.copy()

        frame_input, scale, pad_left, pad_top, orig_h, orig_w = preprocess_frame(
            frame, INPUT_SIZE
        )

        t0 = time.time()
        outputs = self.trt_engine.infer(frame_input)
        self.total_infer += time.time() - t0

        detections = postprocess(outputs[0], scale, pad_left, pad_top, orig_h, orig_w)
        detections = self.tracker.update(detections, self.frame_id)

        persons = [d for d in detections if d["class_id"] == PERSON_CLASS]
        bags = [d for d in detections if d["class_id"] in BAG_CLASSES]

        def centroid(d):
            x1, y1, x2, y2 = d["box"]
            return (x1 + x2) / 2, (y1 + y2) / 2

        now = time.time()

        for bag in bags:
            bid = bag["track_id"]
            bx, by = centroid(bag)

            if bid in self.alarm_hold_time and now < self.alarm_hold_time[bid]:
                self.abandoned_flags[bid] = True
                continue

            if persons:
                nearest_dist = min(
                    math.hypot(centroid(p)[0] - bx, centroid(p)[1] - by)
                    for p in persons
                )
            else:
                nearest_dist = float("inf")

            is_abandoned = False
            if nearest_dist > self.dist_threshold:
                self.away_start.setdefault(bid, now)
                if now - self.away_start[bid] > self.time_threshold:
                    is_abandoned = True
            else:
                self.away_start.pop(bid, None)

            if is_abandoned and not self.abandoned_flags.get(bid, False):
                self.alarm_hold_time[bid] = now + ALARM_HOLD_TIME
            self.abandoned_flags[bid] = is_abandoned

        for p in persons:
            draw_box(display, p["box"], COLOR_PERSON, f"Person {p['track_id']}")

        abandoned_ids = []
        for bag in bags:
            bid = bag["track_id"]
            name = CLASS_NAMES.get(bag["class_id"], "Bag")
            if self.abandoned_flags.get(bid, False):
                draw_box(display, bag["box"], COLOR_ALARM, "ABANDONED!")
                abandoned_ids.append(bid)
            else:
                draw_box(display, bag["box"], COLOR_BAG_NORMAL, f"{name} {bid}")

        abandoned_count = len(abandoned_ids)
        avg_fps = self.frame_id / self.total_infer if self.total_infer > 0 else 0
        cv2.putText(
            display,
            f"FPS: {avg_fps:.1f} | Abandoned: {abandoned_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
        )

        # publish annotated frame
        out_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_annotated.publish(out_msg)

        # publish alarm
        alarm_msg = Bool()
        alarm_msg.data = abandoned_count > 0
        self.pub_alarm.publish(alarm_msg)

        if abandoned_count > 0:
            detail_msg = String()
            detail_msg.data = json.dumps(
                {"abandoned_bag_ids": abandoned_ids, "count": abandoned_count}
            )
            self.pub_detail.publish(detail_msg)

        self.get_logger().info(
            f"Frame {self.frame_id} | FPS {avg_fps:.1f} | Persons {len(persons)} | "
            f"Bags {len(bags)} | Abandoned {abandoned_count}",
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = AbandonedDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
