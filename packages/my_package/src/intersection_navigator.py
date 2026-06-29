#!/usr/bin/env python3
"""
Intersection navigator (Taak 3 - Navigatie) -- positie-gebaseerd.

In plaats van rode stoplijnen te detecteren, gebruikt deze node de
ODOMETRIE-positie en de geplande route (Dijkstra + lokalisatie) om te
weten wanneer de bot een kruispunt nadert en welke kant op te draaien.

Stroom:
  1. Lane-following (PID) rijdt de bot over de gele middenlijn.
  2. Deze node volgt de positie (/odometry) en de route (/current_node,
     /next_node) van de lokalisatie.
  3. Komt de bot dicht bij een KRUISPUNT-node (een junction uit de kaart),
     dan berekent hij uit de coordinaten of de volgende node LINKS, RECHTS
     of RECHTDOOR ligt -- op basis van de richting waaruit hij binnenkwam.
  4. Lane-following pauzeert (/intersection_active = True), de getimede
     draai wordt uitgevoerd, daarna neemt lane-following het weer over.

Geen kleurdetectie nodig. Alle tijden/drempels staan als parameters
bovenaan zodat je ze tijdens het testen kunt afstellen.
"""

import os
import math
import yaml
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool

# ── Afstelbare parameters ─────────────────────────────────────────────────────
# Hoe dicht (in kaart-eenheden) bij een kruispunt-node voor we gaan draaien
TRIGGER_DISTANCE   = 0.6

# Snelheden tijdens een manoeuvre
FORWARD_SPEED      = 0.20   # m/s, oprijden tot midden kruispunt
TURN_SPEED         = 0.15   # m/s, voorwaartse snelheid tijdens de draai
TURN_OMEGA         = 2.5    # rad/s, draaisnelheid (positief = links)

# Getimede duur van elk deel van de manoeuvre (sec) - AFSTELLEN BIJ TEST
ENTER_TIME         = 0.6    # tijd om het kruispunt op te rijden
TURN_TIME_90       = 1.0    # tijd voor een bocht van 90 graden
STRAIGHT_TIME      = 1.0    # tijd om recht over het kruispunt te rijden

# Welke nodes zijn kruispunten (junctions) waar gedraaid kan worden.
# Hoeken (TL, TR, BL, BR) zijn geen doorgaande kruispunten.
JUNCTION_NODES = {"TC", "MC", "MR", "ML", "BC"}
# ──────────────────────────────────────────────────────────────────────────────


class IntersectionNavigator(DTROS):

    def __init__(self, node_name: str):
        super(IntersectionNavigator, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
        )

        vehicle_name = os.environ["VEHICLE_NAME"]

        # Kaart laden (zelfde map.yaml als Dijkstra/lokalisatie)
        map_file = rospy.get_param("~map_file")
        with open(map_file, "r") as f:
            data = yaml.safe_load(f)
        self.node_positions = {
            n: (float(c[0]), float(c[1])) for n, c in data["nodes"].items()
        }

        twist_topic = f"/{vehicle_name}/car_cmd_switch_node/cmd"

        # Status
        self._x = None
        self._y = None
        self._current_node = None
        self._next_node    = None
        self._prev_node    = None   # om de binnenkomst-richting te bepalen
        self._busy         = False

        # Publishers & subscribers
        self._pub_cmd    = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self._pub_active = rospy.Publisher("/intersection_active", Bool, queue_size=1)

        rospy.Subscriber("/odometry",     Odometry, self._cb_odom,    queue_size=1)
        rospy.Subscriber("/current_node", String,   self._cb_current, queue_size=1)
        rospy.Subscriber("/next_node",    String,   self._cb_next,    queue_size=1)

        self._pub_active.publish(Bool(data=False))
        rospy.loginfo(f"[{node_name}] Gestart - draait bij kruispunten volgens de route.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_odom(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._check_intersection()

    def _cb_current(self, msg):
        # Onthoud de vorige node zodra de huidige verandert
        if msg.data != self._current_node:
            self._prev_node = self._current_node
            self._current_node = msg.data

    def _cb_next(self, msg):
        self._next_node = msg.data

    # ── Kruispunt-check ───────────────────────────────────────────────────────

    def _check_intersection(self):
        if self._busy:
            return
        if self._x is None or self._next_node is None:
            return
        if self._next_node == "GOAL_REACHED":
            return
        if self._next_node not in self.node_positions:
            return
        # Alleen draaien bij echte kruispunten (junctions)
        if self._next_node not in JUNCTION_NODES:
            return

        nx, ny = self.node_positions[self._next_node]
        dist = math.hypot(nx - self._x, ny - self._y)

        rospy.loginfo_throttle(
            1.0,
            f"[nav] pos=({self._x:.2f},{self._y:.2f}) -> volgende node "
            f"{self._next_node} op {dist:.2f}  (trigger < {TRIGGER_DISTANCE})"
        )

        if dist <= TRIGGER_DISTANCE:
            self._handle_intersection()

    # ── Kruispunt afhandelen ─────────────────────────────────────────────────

    def _handle_intersection(self):
        self._busy = True
        self._pub_active.publish(Bool(data=True))   # lane-following pauzeren
        self._stop()
        rospy.sleep(0.3)

        turn = self._decide_turn()
        rospy.loginfo(
            f"➡ Kruispunt {self._next_node}: manoeuvre {turn} "
            f"(van {self._prev_node} via {self._current_node} naar {self._next_node})"
        )

        self._drive(FORWARD_SPEED, 0.0, ENTER_TIME)   # kruispunt oprijden

        if turn == "LEFT":
            self._drive(TURN_SPEED,  abs(TURN_OMEGA), TURN_TIME_90)
        elif turn == "RIGHT":
            self._drive(TURN_SPEED, -abs(TURN_OMEGA), TURN_TIME_90)
        else:
            self._drive(FORWARD_SPEED, 0.0, STRAIGHT_TIME)

        self._stop()
        rospy.sleep(0.3)

        self._pub_active.publish(Bool(data=False))   # besturing teruggeven
        rospy.loginfo("✅ Manoeuvre klaar - lane-following neemt weer over.")

        rospy.sleep(1.5)   # niet meteen opnieuw triggeren
        self._busy = False

    # ── Richtingsbeslissing ──────────────────────────────────────────────────

    def _decide_turn(self) -> str:
        """
        Bepaal LEFT / RIGHT / STRAIGHT uit de hoek tussen de binnenkomst-
        richting (prev->current) en de uitgaande richting (current->next).
        """
        prev = self._prev_node
        cur  = self._current_node
        nxt  = self._next_node

        # Zonder vorige node kunnen we de binnenkomst-richting niet bepalen
        if prev is None or cur is None or nxt is None:
            return "STRAIGHT"
        for n in (prev, cur, nxt):
            if n not in self.node_positions:
                return "STRAIGHT"

        px, py = self.node_positions[prev]
        cx, cy = self.node_positions[cur]
        nx, ny = self.node_positions[nxt]

        # Binnenkomst-richting en uitgaande richting
        in_angle  = math.atan2(cy - py, cx - px)
        out_angle = math.atan2(ny - cy, nx - cx)

        # Verschil, genormaliseerd naar [-180, 180] graden
        diff = math.degrees(out_angle - in_angle)
        diff = (diff + 180) % 360 - 180

        if diff > 45:
            return "LEFT"
        elif diff < -45:
            return "RIGHT"
        else:
            return "STRAIGHT"

    # ── Lage-niveau besturing ────────────────────────────────────────────────

    def _drive(self, v, omega, duration):
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
