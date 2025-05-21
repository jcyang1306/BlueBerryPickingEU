import copy, csv, datetime, os
import cv2
import open3d as o3d
import numpy as np

class DataRecorder():
    def __init__(self, root_path='/home/hkclr/AGBot_ws/eu_robot_ws/refactored_eu_arm/tmp/picking_data'):
        ## Data saving config ##
        date_prefix = datetime.date.today().isoformat()
        grasping_data_p = os.path.join(root_path, "date_" + date_prefix)
        if not os.path.exists(grasping_data_p):
            os.makedirs(grasping_data_p)
        time_stamp = datetime.datetime.now().strftime("%H-%M-%S")
        grasping_data_p = os.path.join(grasping_data_p, time_stamp)
        if not os.path.exists(grasping_data_p):
            os.makedirs(grasping_data_p)
        self.save_path_ = grasping_data_p
        self.idx_ = 0
        print(f'===== Data saving path: {grasping_data_p}') 

    def save_img(self, img, idx):
        print(f'Saved {self.save_path_}/frame-{idx:06d}_color.jpg')
        cv2.imwrite(f'{self.save_path_}/frame-{idx:06d}_color.jpg', img)

    def save_depth(self, img, idx):
        print(f'Saved {self.save_path_}/frame-{idx:06d}_depth.png')
        cv2.imwrite(f'{self.save_path_}/frame-{idx:06d}_depth.png', img)

    def write_log(self, log, idx):
        print(f'Saved {self.save_path_}/frame-{idx:06d}_pose.txt')
        np.savetxt(f"{self.save_path_}/frame-{idx:06d}_pose.txt", log, fmt="%.5f")

    def save_ply(self, pc, idx):
        print(f'Saved {self.save_path_}/frame-{idx:06d}.ply')
        o3d.io.write_point_cloud(f'{self.save_path_}/frame-{idx:06d}.ply', pc)

    def save_data(self, img, pc, log):
        self.save_img(img, self.idx_)
        self.write_log(log, self.idx_)
        self.save_ply(pc, self.idx_)
        self.idx_ += 1