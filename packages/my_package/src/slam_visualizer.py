#!/usr/bin/env python3

import rospy
import threading
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

app = Flask(__name__)

class WebSlamVisualizer:
    def __init__(self):
        rospy.init_node('slam_web_visualizer', anonymous=True)
        self.lock = threading.Lock()

        self.odom_x, self.odom_y = [], []
        self.slam_x, self.slam_y = [], []
        self.fused_x, self.fused_y = [], []

        rospy.Subscriber('/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/visual_pose', PoseStamped, self.slam_callback)
        rospy.Subscriber('/fused_pose', PoseStamped, self.fused_callback)

    def odom_callback(self, msg):
        with self.lock:
            self.odom_x.append(msg.pose.pose.position.x)
            self.odom_y.append(msg.pose.pose.position.y)

    def slam_callback(self, msg):
        with self.lock:
            self.slam_x.append(msg.pose.position.x)
            self.slam_y.append(msg.pose.position.y)

    def fused_callback(self, msg):
        with self.lock:
            self.fused_x.append(msg.pose.position.x)
            self.fused_y.append(msg.pose.position.y)

    def generate_plot(self):
        with self.lock:
            fig, ax = plt.subplots()
            ax.plot(self.odom_x, self.odom_y, label='Odometry')
            ax.plot(self.slam_x, self.slam_y, label='SLAM')
            ax.plot(self.fused_x, self.fused_y, label='Fused')
            ax.set_title('SLAM Web Visualization')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.legend()
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf

visualizer = WebSlamVisualizer()

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>SLAM Web Visualization</title>
        <meta http-equiv="refresh" content="1">
    </head>
    <body>
        <h1>SLAM Web Visualization</h1>
        <img src="/plot.png" />
    </body>
    </html>
    '''

@app.route('/plot.png')
def plot_png():
    buf = visualizer.generate_plot()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    thread.daemon = True
    thread.start()
    rospy.spin()
