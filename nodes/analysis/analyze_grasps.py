#! /home/playert/miniconda3/envs/ros/bin/python

from multiprocessing.sharedctypes import Value
import sys
sys.path.insert(0, "/home/rsa/testbed/tim_grasp_ws/devel/lib/python3/dist-packages/")

import rospy
from cl_tsgrasp.msg import Grasps
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import List
from threading import Lock
import os
import time

# Class to maintain a pd.DataFrame of grasp info and
# plot descriptive statistics
class GraspProcessor:

    def __init__(self, fig, axs):
        self.df = pd.DataFrame()
        self.fig, self.axs = fig, axs
        self.mtx = Lock() # mutex for analyzing and plotting one message

    def process_grasps(self, msg: Grasps):
        self.add_grasps_to_df(msg)

    def add_grasps_to_df(self, msg: Grasps):
        confs = self.retrieve_confs(msg)
        depths = self.retrieve_depths(msg)
        widths = self.retrieve_widths(msg)
        time = self.retrieve_time(msg)

        entries = []
        for conf, depth, width, pose in zip(confs, depths, widths, msg.poses):
            entries.append({
                "conf" : conf,
                "pose_x" : pose.position.x,
                "pose_y" : pose.position.y,
                "pose_z" : pose.position.z,
                "pose_qx" : pose.orientation.x,
                "pose_qy" : pose.orientation.y,
                "pose_qz" : pose.orientation.z,
                "depth" : depth,
                "width" : width,
                "time": time
            })

        self.df = pd.concat([self.df, pd.DataFrame(entries)])

    def retrieve_confs(self, msg: Grasps):
        return np.array(msg.confs)

    def retrieve_depths(self, msg: Grasps):
        if msg.header.frame_id not in ('left', 'camera_depth_optical_frame'):
            raise ValueError(f"Grasps in wrong frame: {msg.header.frame_id}")
        return np.array([
            p.position.z
            for p in msg.poses
        ])

    def retrieve_widths(self, msg: Grasps):
        return np.array(msg.widths)

    def retrieve_time(self, msg: Grasps):
        T = msg.header.stamp
        return datetime.fromtimestamp(T.to_sec())

    def plot_data(self):
        for row in self.axs: 
            for a in row:
                a.clear()
        self.plot_histogram(self.df["conf"], self.axs[0][0], "conf", range=(0,1))
        self.plot_histogram(self.df["depth"], self.axs[1][0], "depth (m)", "PDF of depth", range=(0,1.5))

        quantile = self.df['conf'].quantile(0.95)
        self.plot_histogram(
            self.df[self.df["conf"] > quantile]["depth"], 
            self.axs[2][0], 
            "depth (m)",
            title="PDF of depths with conf above 95th percentile",
            range=(0,1.5))

        self.plot_histogram(self.df["width"], self.axs[0][1], "width (m)", "PDF of width", range=(0,0.06))

        self.plot_histogram(
            self.df[self.df["conf"] > quantile]["width"],
            self.axs[1][1],
            "width (m)",
            title="PDF of widths with conf above 95th percentile",
            range=(0,0.06)
        )

        self.fig.tight_layout()

        os.makedirs('/home/rsa/testbed/tim_grasp_ws/figs/out/', exist_ok=True)

        # plt.savefig('/home/rsa/testbed/tim_grasp_ws/figs/out/hists.svg')
        plt.savefig('/home/rsa/testbed/tim_grasp_ws/figs/out/hists.png')
        self.df.to_csv('/home/rsa/testbed/tim_grasp_ws/figs/out/grasps.csv')

    @staticmethod
    def plot_histogram(data, ax, xlabel, title=None, bins=100, range=None):
        if title is None: title = f'PDF of {xlabel}'
        # hist, bin_edges = np.histogram(data, bins=bins, range=(0, 1), density=True)
        # ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        ax.hist(data, bins=bins, range=range, density=True)
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)

if __name__ == "__main__":
    rospy.init_node('analyze_grasps')
    fig, axs = plt.subplots(3, 2, figsize=(20,10))
    proc = GraspProcessor(fig, axs)
    proc.time = time.time()

    def grasp_cb(msg):
        with proc.mtx:
            print(f"Processing message! {time.time() - proc.time : 3f}")
            proc.time = time.time()
            proc.process_grasps(msg)
            proc.plot_data()

    print("Subscribing to grasps message.")
    grasps_sub = rospy.Subscriber('tsgrasp/grasps', Grasps, grasp_cb)

    rospy.spin()
