#!/usr/bin/env python3
import numpy as np
import math

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class EKF:
    def __init__(self, q0, P0, Q, R):
        self.q = q0.copy().astype(float)
        self.P = P0.copy().astype(float)
        self.Q = Q.copy().astype(float)
        self.R = R.copy().astype(float)

    def predict(self, dX, dtheta):
        x, y, theta = self.q
        theta_mid = theta + 0.5 * dtheta
        self.q = np.array([
            x + dX * math.cos(theta_mid),
            y + dX * math.sin(theta_mid),
            wrap_angle(theta + dtheta)
        ])
        F = np.array([
            [1.0, 0.0, -dX * math.sin(theta_mid)],
            [0.0, 1.0,  dX * math.cos(theta_mid)],
            [0.0, 0.0,  1.0],
        ])
        G = np.array([
            [math.cos(theta_mid), -0.5 * dX * math.sin(theta_mid)],
            [math.sin(theta_mid),  0.5 * dX * math.cos(theta_mid)],
            [0.0,                  1.0],
        ])
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    def update(self, z, landmark_xy):
        x, y, theta = self.q
        lx, ly = landmark_xy
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2
        sq = math.sqrt(q) if q > 1e-6 else 1e-3
        z_pred = np.array([sq, wrap_angle(math.atan2(dy, dx) - theta)])
        innov = z - z_pred
        innov[1] = wrap_angle(innov[1])
        H = np.array([
            [-dx/sq, -dy/sq,  0.0],
            [ dy/q,  -dx/q,  -1.0],
        ])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ innov
        self.q = self.q + delta
        self.q[2] = wrap_angle(self.q[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
