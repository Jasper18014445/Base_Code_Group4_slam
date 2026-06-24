#!/usr/bin/env python3
import numpy as np
 
class EKF:
    """
    Extended Kalman Filter for 2D robot pose estimation.
 
    State:  q = [x, y, theta]
    Motion: differential drive  (dX, dtheta)
    Measurement: range + bearing to a known point (tag_xy)
    """
 
    def __init__(self, q0, P0, Q, R):
        """
        q0 : np.array(3,)   initial state [x, y, theta]
        P0 : np.array(3,3)  initial covariance
        Q  : np.array(2,2)  process noise  (dX, dtheta)
        R  : np.array(2,2)  measurement noise (range, bearing)
        """
        self.q = q0.copy().astype(float)
        self.P = P0.copy().astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
 
    # ── predict ───────────────────────────────────────────────────────────────
 
    def predict(self, dX, dtheta):
        """
        Motion model: move forward dX metres, rotate dtheta radians.
        """
        x, y, th = self.q
        th_new = th + dtheta
 
        # State update
        self.q[0] += dX * np.cos(th)
        self.q[1] += dX * np.sin(th)
        self.q[2]  = self._wrap(th_new)
 
        # Jacobian of motion model w.r.t. state (F)
        F = np.array([
            [1, 0, -dX * np.sin(th)],
            [0, 1,  dX * np.cos(th)],
            [0, 0,  1              ],
        ])
 
        # Jacobian of motion model w.r.t. noise (G)
        G = np.array([
            [np.cos(th), 0],
            [np.sin(th), 0],
            [0,          1],
        ])
 
        # Covariance update
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T
 
    # ── update ────────────────────────────────────────────────────────────────
 
    def update(self, z, tag_xy):
        """
        Measurement update.
 
        z      : np.array(2,)  [range, bearing]  from sensor_fusion.py
        tag_xy : np.array(2,)  [x, y] of the observed landmark in world frame
        """
        x, y, th = self.q
 
        dx = tag_xy[0] - x
        dy = tag_xy[1] - y
        rng = np.sqrt(dx**2 + dy**2)
 
        if rng < 1e-6:          # landmark on top of robot — skip
            return
 
        # Expected measurement
        z_hat = np.array([rng, self._wrap(np.arctan2(dy, dx) - th)])
 
        # Innovation
        inn = z - z_hat
        inn[1] = self._wrap(inn[1])
 
        # Measurement Jacobian (H)
        H = np.array([
            [-dx/rng,        -dy/rng,         0],
            [ dy/rng**2,    -dx/rng**2,      -1],
        ])
 
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
 
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
 
        # State & covariance update
        self.q = self.q + K @ inn
        self.q[2] = self._wrap(self.q[2])
        self.P = (np.eye(3) - K @ H) @ self.P
 
    # ── helper ────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _wrap(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi
