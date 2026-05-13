#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from typing import Optional
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from std_msgs.msg import Bool


# ── Rijparameters ────────────────────────────────────────────────────────────
VELOCITY      = 0.25   # voorwaartse snelheid (m/s)
KP            = 0.035  # proportionele gain
KD            = 0.010  # afgeleide gain  (dempt oscillaties)

# ── ROI: onderste deel van het frame gebruiken ───────────────────────────────
ROI_FRACTION  = 0.5   # gebruik de onderste 50 % van het beeld

# ── HSV-kleurdrempels ────────────────────────────────────────────────────────
# Geel (linkerrijstrook)
YELLOW_LOW  = np.array([ 20,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([ 35, 255, 255], dtype=np.uint8)

# Wit (rechterrijstrook)
WHITE_LOW   = np.array([  0,   0, 180], dtype=np.uint8)
WHITE_HIGH  = np.array([180,  60, 255], dtype=np.uint8)

# Minimaal contouroppervlak om ruis te negeren
MIN_CONTOUR_AREA = 300


class LaneFollowingNode(DTROS):
    """
    Lane-following node voor de Duckiebot.

    Detecteert de gele (links) en witte (rechts) rijstrooklijnen via
    HSV-kleurfiltering, berekent de laterale fout ten opzichte van het
    midden van de rijstrook en stuurt de robot bij via een PD-regelaar.
    """

    def __init__(self, node_name: str):
        super(LaneFollowingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        vehicle_name = os.environ["VEHICLE_NAME"]

        # ── Topics ───────────────────────────────────────────────────────────
        camera_topic = f"/{vehicle_name}/camera_node/image/compressed"
        twist_topic  = f"/{vehicle_name}/car_cmd_switch_node/cmd"

        # ── CV-hulpmiddelen ───────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Stop-vlag (gezet door duckie_detector_node) ──────────────────────────
        self._duckie_stop = False

        # ── PD-regelaar toestandsvariabelen ──────────────────────────────────
        self._prev_error = 0.0
        self._prev_time  = rospy.Time.now()

        # ── Publisher & Subscriber ───────────────────────────────────────────
        self._pub = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self._sub = rospy.Subscriber(
            camera_topic, CompressedImage, self._cb_image, queue_size=1
        )

        stop_topic = f"/{vehicle_name}/duckie_stop"
        self._stop_sub = rospy.Subscriber(
            stop_topic, Bool, self._cb_stop, queue_size=1
        )

        rospy.loginfo(f"[{node_name}] Gestart — luistert op {camera_topic}")

    # ── Beeldverwerking ───────────────────────────────────────────────────────

    def _preprocess(self, bgr_image: np.ndarray):
        """Verkleinen, Gaussiaans vervagen, converteren naar HSV."""
        small = cv2.resize(bgr_image, (320, 240))
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return small, hsv

    def _get_roi(self, image: np.ndarray) -> np.ndarray:
        """Geeft het onderste gedeelte van het beeld terug (rijstrookgebied)."""
        h = image.shape[0]
        start_row = int(h * (1.0 - ROI_FRACTION))
        return image[start_row:, :]

    def _largest_centroid_x(self, mask: np.ndarray) -> Optional[float]:
        """
        Zoekt het grootste contour in het masker en geeft de x-coördinaat
        van het zwaartepunt terug. Geeft None terug als er niets gevonden is.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"]

    # ── Stop callback ────────────────────────────────────────────────────────

    def _cb_stop(self, msg):
        self._duckie_stop = msg.data

    # ── Hoofdcallback ─────────────────────────────────────────────────────────

    def _cb_image(self, msg: CompressedImage):
        # 0. Stop als duckie_detector een duckie ziet
        if self._duckie_stop:
            self._publish(v=0.0, omega=0.0)
            rospy.logwarn_throttle(1.0, "DUCKIE GEDETECTEERD — robot gestopt!")
            return

        # 1. Decoderen & voorbewerken
        bgr  = self._bridge.compressed_imgmsg_to_cv2(msg)
        _, hsv = self._preprocess(bgr)

        # 2. ROI selecteren
        hsv_roi = self._get_roi(hsv)
        width   = hsv_roi.shape[1]
        cx      = width / 2.0  # beeldmidden

        # 3. Kleurmaskers
        mask_yellow = cv2.inRange(hsv_roi, YELLOW_LOW, YELLOW_HIGH)
        mask_white  = cv2.inRange(hsv_roi, WHITE_LOW,  WHITE_HIGH)

        # Tijdelijk: sla één frame op voor debugging
        if not hasattr(self, '_saved'):
            self._saved = True
            cv2.imwrite('/tmp/frame_bgr.png', bgr)
            cv2.imwrite('/tmp/mask_yellow.png', mask_yellow)
            cv2.imwrite('/tmp/mask_white.png', mask_white)
            rospy.loginfo("Debug frames opgeslagen in /tmp/")

        # 4. Zwaartepunten per lijn
        x_yellow = self._largest_centroid_x(mask_yellow)
        x_white  = self._largest_centroid_x(mask_white)

        # 5. Foutberekening
        #    Ideaal rijdpunt = midden tussen de twee lijnen.
        #    Ontbreekt een lijn, dan gebruiken we een vaste offset als schatting.
        if x_yellow is not None and x_white is not None:
            lane_center = (x_yellow + x_white) / 2.0
        elif x_yellow is not None:
            lane_center = x_yellow + width * 0.25  # wit ontbreekt → schat rechts
        elif x_white is not None:
            lane_center = x_white  - width * 0.25  # geel ontbreekt → schat links
        else:
            # Geen enkele lijn zichtbaar — stop en wacht
            self._publish(v=0.0, omega=0.0)
            rospy.logwarn_throttle(2.0, "Geen rijstrooklijnen gevonden — robot gestopt.")
            return

        # Fout: positief = robot is te ver naar links → stuur rechts (negatieve omega)
        error = lane_center - cx

        # 6. PD-regelaar
        now  = rospy.Time.now()
        dt   = (now - self._prev_time).to_sec()
        dt   = max(dt, 1e-3)  # voorkom deling door nul

        derivative   = (error - self._prev_error) / dt
        omega        = -(KP * error + KD * derivative)  # negatief: stuur naar midden

        self._prev_error = error
        self._prev_time  = now

        # 7. Publiceren
        self._publish(v=VELOCITY, omega=omega)

        # 8. Info logging (throttled zodat terminal leesbaar blijft)
        y_str    = f"{x_yellow:.0f}" if x_yellow is not None else "N/A"
        w_str    = f"{x_white:.0f}"  if x_white  is not None else "N/A"
        lines    = "geel+wit" if (x_yellow is not None and x_white is not None) \
                   else ("alleen geel" if x_yellow is not None else "alleen wit")
        stuur    = "LINKS" if omega > 0.05 else ("RECHTS" if omega < -0.05 else "RECHTUIT")
        rospy.loginfo_throttle(1.0,
            f"\n--- Lane Following ---\n"
            f"  Lijnen zichtbaar : {lines}\n"
            f"  Geel centroid    : {y_str} px\n"
            f"  Wit centroid     : {w_str} px\n"
            f"  Laterale fout    : {error:+.1f} px\n"
            f"  Rijsnelheid (v)  : {VELOCITY:.2f} m/s\n"
            f"  Stuurhoek (omega): {omega:+.3f} rad/s\n"
            f"  Richting         : {stuur}\n"
            f"----------------------"
        )

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    def _publish(self, v: float, omega: float):
        msg = Twist2DStamped()
        msg.v     = v
        msg.omega = omega
        self._pub.publish(msg)

    def on_shutdown(self):
        """Zorg dat de robot stopt bij afsluiten."""
        self._publish(v=0.0, omega=0.0)
        rospy.loginfo("Lane following gestopt — robot tot stilstand gebracht.")


# ── Startpunt ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = LaneFollowingNode(node_name="lane_following_node")
    rospy.spin()
