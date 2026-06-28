#!/usr/bin/env python3
"""
Intersection navigator (Taak 3 - Navigatie).

Werkt samen met de lane-following node (Taak 4 - PID) en de lokalisatie
node (Taak 2). Stroom:

  1. Lane-following (PID) rijdt de bot over de gele/witte lijnen.
  2. Deze node kijkt naar de camera en detecteert de RODE stoplijn
     die in Duckietown een kruispunt markeert.
  3. Bij een rode lijn: publiceer /intersection_active = True. De
     lane-following pauzeert dan (luistert naar dat topic).
  4. Lees /current_node en /next_node van de lokalisatie. Bereken uit
     de kaart-coordinaten of de volgende node LINKS, RECHTS of RECHTDOOR
     ligt ten opzichte van de richting waarin we binnenkwamen.
  5. Voer een GETIMEDE manoeuvre uit (oprijden + draaien). Getimed,
     niet op hoek, zodat we niet afhankelijk zijn van een exacte heading
     uit odometry.
  6. Klaar: publiceer /intersection_active = False. Lane-following neemt
     het weer over.

Alle tijden/snelheden staan als parameters bovenaan zodat je ze tijdens
het testen kunt afstellen zonder de logica aan te raken.
"""

import os
import math
import rospy
import cv2
import numpy as np
from typing import Optional
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

# ── Afstelbare parameters ─────────────────────────────────────────────────────
# Snelheden tijdens een manoeuvre
FORWARD_SPEED      = 0.20   # m/s, oprijden tot midden kruispunt
TURN_SPEED         = 0.15   # m/s, voorwaartse snelheid tijdens de draai
TURN_OMEGA         = 2.5    # rad/s, draaisnelheid (positief = links)

# Getimede duur van elk stuk van de manoeuvre (in seconden) - AFSTELLEN BIJ TEST
ENTER_TIME         = 0.8    # tijd om het kruispunt op te rijden
TURN_TIME_90       = 1.0    # tijd voor een bocht van 90 graden
STRAIGHT_TIME      = 1.2    # tijd om recht over het kruispunt te rijden

# Rood-detectie (HSV). Rood ligt aan beide uiteinden van de hue-schaal,
# dus we gebruiken twee bereiken en tellen ze samen.
RED_LOW_1   = np.array([  0, 100,  80], dtype=np.uint8)
RED_HIGH_1  = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LOW_2   = np.array([160, 100,  80], dtype=np.uint8)
RED_HIGH_2  = np.array([180, 255, 255], dtype=np.uint8)

# Hoeveel rode pixels (in onderste deel van het beeld) telt als "stoplijn"
RED_PIXEL_THRESHOLD = 1500
# Onderste deel van het beeld gebruiken (de stoplijn ligt vlak voor de bot)
ROI_FRACTION = 0.4
# ──────────────────────────────────────────────────────────────────────────────


class IntersectionNavigator(DTROS):

    def __init__(self, node_name: str):
        super(IntersectionNavigator, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        vehicle_name = os.environ["VEHICLE_NAME"]

        # Kaart laden (zelfde map.yaml als Dijkstra/lokalisatie gebruiken)
        map_file = rospy.get_param("~map_file")
        with open(map_file, "r") as f:
            data = __import__("yaml").safe_load(f)
        self.node_positions = {
            n: (float(c[0]), float(c[1])) for n, c in data["nodes"].items()
        }

        # ── Topics ───────────────────────────────────────────────────────────
        camera_topic = f"/{vehicle_name}/camera_node/image/compressed"
        twist_topic  = f"/{vehicle_name}/car_cmd_switch_node/cmd"

        self._bridge = CvBridge()

        # Status
        self._current_node = None
        self._next_node    = None
        self._busy         = False   # bezig met een manoeuvre?

        # ── Publishers & subscribers ─────────────────────────────────────────
        self._pub_cmd    = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self._pub_active = rospy.Publisher("/intersection_active", Bool, queue_size=1)

        rospy.Subscriber(camera_topic, CompressedImage, self._cb_image, queue_size=1)
        rospy.Subscriber("/current_node", String, self._cb_current, queue_size=1)
        rospy.Subscriber("/next_node",    String, self._cb_next,    queue_size=1)

        # Begin in normale (rij)modus
        self._pub_active.publish(Bool(data=False))

        rospy.loginfo(f"[{node_name}] Gestart — wacht op rode stoplijnen bij kruispunten.")

    # ── Lokalisatie callbacks ────────────────────────────────────────────────

    def _cb_current(self, msg):
        self._current_node = msg.data

    def _cb_next(self, msg):
        self._next_node = msg.data

    # ── Rood-detectie ────────────────────────────────────────────────────────

    def _red_pixel_count(self, bgr) -> int:
        """Tel rode pixels in het onderste deel van het beeld."""
        small = cv2.resize(bgr, (320, 240))
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = hsv.shape[0]
        roi = hsv[int(h * (1.0 - ROI_FRACTION)):, :]

        mask1 = cv2.inRange(roi, RED_LOW_1, RED_HIGH_1)
        mask2 = cv2.inRange(roi, RED_LOW_2, RED_HIGH_2)
        mask  = cv2.bitwise_or(mask1, mask2)
        return int(cv2.countNonZero(mask))

    # ── Hoofdcallback ────────────────────────────────────────────────────────

    def _cb_image(self, msg: CompressedImage):
        if self._busy:
            return  # tijdens een manoeuvre niet opnieuw triggeren

        bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        red = self._red_pixel_count(bgr)

        if red >= RED_PIXEL_THRESHOLD:
            rospy.loginfo(f"🔴 Rode stoplijn gezien ({red} px) — kruispunt!")
            self._handle_intersection()

    # ── Kruispunt afhandelen ─────────────────────────────────────────────────

    def _handle_intersection(self):
        self._busy = True
        # Lane-following pauzeren
        self._pub_active.publish(Bool(data=True))
        self._stop()
        rospy.sleep(0.3)

        turn = self._decide_turn()
        rospy.loginfo(f"➡ Manoeuvre: {turn}  (van {self._current_node} naar {self._next_node})")

        # Altijd eerst het kruispunt oprijden
        self._drive(FORWARD_SPEED, 0.0, ENTER_TIME)

        if turn == "LEFT":
            self._drive(TURN_SPEED,  abs(TURN_OMEGA), TURN_TIME_90)
        elif turn == "RIGHT":
            self._drive(TURN_SPEED, -abs(TURN_OMEGA), TURN_TIME_90)
        else:  # STRAIGHT of onbekend
            self._drive(FORWARD_SPEED, 0.0, STRAIGHT_TIME)

        self._stop()
        rospy.sleep(0.3)

        # Besturing teruggeven aan lane-following
        self._pub_active.publish(Bool(data=False))
        rospy.loginfo("✅ Manoeuvre klaar — lane-following neemt weer over.")

        # Korte pauze zodat we de net gepasseerde rode lijn niet meteen
        # opnieuw als kruispunt zien
        rospy.sleep(1.5)
        self._busy = False

    # ── Richtingsbeslissing ──────────────────────────────────────────────────

    def _decide_turn(self) -> str:
        """
        Bepaal LEFT / RIGHT / STRAIGHT op basis van de hoek tussen
        (vorige->huidige) en (huidige->volgende) node in de kaart.

        We hebben de vorige node niet expliciet, dus we schatten de
        binnenkomst-richting uit de twee bekende nodes. Als er te weinig
        info is, kiezen we STRAIGHT (veiligste default).
        """
        cur  = self._current_node
        nxt  = self._next_node

        if cur is None or nxt is None or nxt == "GOAL_REACHED":
            return "STRAIGHT"
        if cur not in self.node_positions or nxt not in self.node_positions:
            return "STRAIGHT"

        cx, cy = self.node_positions[cur]
        nx, ny = self.node_positions[nxt]

        # Richting van huidige naar volgende node
        dx = nx - cx
        dy = ny - cy

        # We nemen aan dat de bot "vooruit" in de richting van de vorige
        # beweging kijkt. Zonder expliciete heading gebruiken we een
        # eenvoudige vuistregel op basis van het grootste richtingsverschil.
        # Dit is een benadering — bij het testen afstellen of vervangen
        # door een echte heading uit /odometry als die betrouwbaar is.
        angle = math.degrees(math.atan2(dy, dx))

        # Normaliseer naar [-180, 180]
        if angle > 135 or angle < -135:
            return "STRAIGHT"   # terug/recht
        elif 45 < angle <= 135:
            return "LEFT"
        elif -135 <= angle < -45:
            return "RIGHT"
        else:
            return "STRAIGHT"

    # ── Lage-niveau besturing ────────────────────────────────────────────────

    def _drive(self, v, omega, duration):
        """Rij met (v, omega) gedurende 'duration' seconden."""
        rate = rospy.Rate(20)
        t_end = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < t_end and not rospy.is_shutdown():
            self._publish(v, omega)
            rate.sleep()

    def _stop(self):
        self._publish(0.0, 0.0)

    def _publish(self, v, omega):
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = omega
        self._pub_cmd.publish(msg)

    def on_shutdown(self):
        self._stop()
        self._pub_active.publish(Bool(data=False))


if __name__ == "__main__":
    node = IntersectionNavigator(node_name="intersection_navigator")
    rospy.spin()
