
import os
import open3d as o3d
import cv2

from sensing_algo import GraspingAlgo

def test_infer(algo, img):
    print('Testing inference')
    masks, bboxes, scores = algo.infer_img(img)
    return masks, bboxes, scores

def test_gen_grasp_pose(algo, img, pc, masks):
    print('Testing gen_grasp_pose')
    grasp_poses, grasp_uv = algo.gen_grasp_pose(img, pc, masks)
    return grasp_poses, grasp_uv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", "-r", default='/home/hkclr/AGBot_ws/eu_robot_ws/refactored_eu_arm/tmp/picking_data/date_2025-04-02/15-24-31', help="Path to the data")
    parser.add_argument("--index", "-i", default=0,type=int, help="i th frame data to process")
    args = parser.parse_args()
    
    # Process data path stuff
    r_path = args.root_path
    img_path = os.path.join(args.root_path, 'frame-{:06d}_color.jpg'.format(args.index))
    ply_path = os.path.join(args.root_path, 'frame-{:06d}.ply'.format(args.index))

    # Load model
    grasp_algo = GraspingAlgo()

    # Load test data
    print(f'<Load data> imgpath: {img_path}\nply path: {ply_path}')
    color_image = cv2.imread(img_path)
    pc = o3d.io.read_point_cloud(ply_path)

    # Test inference
    masks, _, _ = test_infer(grasp_algo, color_image)

    # Test grasp pose generation
    test_gen_grasp_pose(grasp_algo, color_image, pc, masks)