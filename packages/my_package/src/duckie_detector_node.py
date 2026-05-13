#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import onnxruntime as ort
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge


# ── Detectieparameters ───────────────────────────────────────────────────────

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PACKAGE_PATH, "best.onnx")
CONFIDENCE_THRESHOLD = 0.5    # minimale zekerheid om een duckie te tellen
INPUT_SIZE           = (640, 640)  # YOLOv11 standaard inputgrootte

import os
rospy.loginfo(f"Bestaat model? {os.path.exists(MODEL_PATH)}")

class DuckieDetectorNode(DTROS):
    """
    Detecteert duckies met een YOLOv11 ONNX model via onnxruntime.
    Publiceert True op /{vehicle}/duckie_stop wanneer een duckie gezien wordt,
    zodat de lane_following_node de robot kan stoppen.
    """

    def __init__(self, node_name: str):
        super(DuckieDetectorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        vehicle_name = os.environ["VEHICLE_NAME"]

        # ── Topics ───────────────────────────────────────────────────────────
        camera_topic     = f"/{vehicle_name}/camera_node/image/compressed"
        self._stop_topic = f"/{vehicle_name}/duckie_stop"

        # ── Publisher & Subscriber (eerst aanmaken zodat on_shutdown werkt) ──
        self._stop_pub = rospy.Publisher(self._stop_topic, Bool, queue_size=1)
        self._bridge   = CvBridge()

        # ── ONNX model laden via onnxruntime ──────────────────────────────────
        rospy.loginfo(f"Model laden van: {MODEL_PATH}")
        self._session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        rospy.loginfo("ONNX model geladen via onnxruntime!")

        # ── Subscriber ────────────────────────────────────────────────────────
        self._sub = rospy.Subscriber(
            camera_topic, CompressedImage, self._cb_image, queue_size=1
        )

        rospy.loginfo(f"[{node_name}] Gestart — publiceert op {self._stop_topic}")

    # ── Beeldvoorbereiding ────────────────────────────────────────────────────

    def _preprocess(self, bgr):
        """BGR → RGB → resize → normalize → NCHW float32 blob."""
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE)
        blob  = resized.astype(np.float32) / 255.0
        blob  = np.transpose(blob, (2, 0, 1))   # HWC → CHW
        blob  = np.expand_dims(blob, axis=0)     # CHW → NCHW
        return blob

    # ── Hoofdcallback ─────────────────────────────────────────────────────────

    def _cb_image(self, msg: CompressedImage):
        # 1. Decoderen & voorbewerken
        bgr  = self._bridge.compressed_imgmsg_to_cv2(msg)
        blob = self._preprocess(bgr)

        # 2. Inferentie
        outputs = self._session.run(None, {self._input_name: blob})

        # 3. Resultaten verwerken
        # YOLOv11 output shape: [1, 4+num_classes, 8400]
        predictions = outputs[0][0].T   # → [8400, 4+num_classes]

        duckie_gevonden = False
        best_confidence = 0.0

        for pred in predictions:
            conf = float(np.max(pred[4:]))
            if conf >= CONFIDENCE_THRESHOLD:
                duckie_gevonden = True
                best_confidence = max(best_confidence, conf)

        # 4. Publiceren
        self._stop_pub.publish(Bool(data=duckie_gevonden))

        # 5. Logging
        if duckie_gevonden:
            rospy.logwarn_throttle(
                1.0, f"DUCKIE GEDETECTEERD! Zekerheid: {best_confidence:.0%} — stop signaal gestuurd."
            )
        else:
            rospy.loginfo_throttle(3.0, "Geen duckie gevonden, rijden mag.")

    def on_shutdown(self):
        self._stop_pub.publish(Bool(data=False))
        rospy.loginfo("Duckie detector gestopt.")


# ── Startpunt ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = DuckieDetectorNode(node_name="duckie_detector_node")
    rospy.spin()
