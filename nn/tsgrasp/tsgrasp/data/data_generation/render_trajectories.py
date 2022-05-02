"""
trajectory_vid.py
A module for creating RGB-D videos annotated with grasps from the ACRONYM dataset.
Tim Player playert@oregonstate.edu July 22, 2021
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # for headless

import trimesh
import numpy as np
import os
import h5py
import csv
from omegaconf import DictConfig
from functools import partial

import sys

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def make_trajectory(trimesh_camera, obj_loc, num_frames : int, seed=None):
    """Create a "random" trajectory that views the object.
    Initially, these trajectories will be circular orbits that always look directly at the object, at different elevation angles.

    More creative distributions of trajectories, including ones in which
    the image-frame object center is not in the middle of the path, are 
    future work.

    Args:
        trimesh_camera (obj): camera object from Trimesh
        obj_loc (np.ndarray): location of the object. Origin? Center? Not sure.
        num_frames (int): number of positions along trajectory
        seed (float, optional): RNG seed. Defaults to None.

    Returns:
        np.ndarray: array of 4x4 poses
    """
    
    np.random.seed(seed)

    # The pitch angle is a uniform random angle 
    avg_pitch = np.pi/3
    pitch_range = 0.16 # roughly 10 degrees
    pitch = np.random.uniform(avg_pitch-pitch_range/2, avg_pitch+pitch_range/2)
    
    # A full circle is swept out, with a random initial yaw angle.
    yaw0 = np.random.uniform(0, 2*np.pi)
    name = f"pitch:{pitch:.2f},yaw0:{yaw0:.2f}" # quick description
    
    d0 = 2 # Orbital distance
    
    yaws = np.linspace(yaw0, yaw0+2*np.pi, num_frames)
    ds = d0 * np.ones_like(yaws)
    poses = []

    for i, (yaw, d) in enumerate(zip(yaws, ds)):
        camera_pose = trimesh_camera.look_at(
            points=[obj_loc],
            rotation=trimesh.transformations.euler_matrix(
                pitch,
                0,
                yaw,
            ),
            distance=d,
        )
        poses.append(camera_pose)
    return poses, name

def transform(pts, tf):
    """Transform a set of row vectors with a homogeneous matrix.

    pts: (B,N,3) or (1, N, 3) stack of row vectors
    tf: (B, 4,4) or (4, 4) homogeneous transformation matrix
    """
    ones = np.ones(shape=(*pts.shape[:-1], 1))
    pts = np.concatenate([pts, ones], axis=-1)
    
    return (tf @ pts.swapaxes(-1, -2)).swapaxes(-1, -2)[:,:,:3]

def inverse_homo(tf):
    """Compute inverse of homogeneous transformation matrix.

    The matrix should have entries
    [[R,       Rt]
     [0, 0, 0, 1]].
    """
    R = tf[0:3, 0:3]
    t = R.T @ tf[0:3, 3].reshape(3, 1)
    return np.block([
        [R.T, -t],
        [0, 0, 0, 1]
    ])

def make_trajectories_for_path(h5_path, traj_per_object, frames_per_traj, mesh_dir, load_mesh, PyrenderScene, SceneRenderer):
    """Open the grasp dataset h5 file at `h5_path`, load its OBJ mesh, render random video trajectories, and append the trajectory information to the file's dataset.

    Args:
        h5_path (str): absolute path to h5 dataset with ACRONYM info
        traj_per_object (int): how many trajectories to generate for this object
        frames_per_traj (int): how many frames per trajectory
        mesh_dir (str): folder containing OBJ meshes
        load_mesh (function): mesh-loading function from ACRONYM
        PyrenderScene (class): scene-rendering class from ACRONYM
        SceneRenderer (class): scene-rendering class from ACRONYM

    Returns:
        bool: success/failure
    """
    try:
        ## Load mesh corresponding to this grasp file
        m = load_mesh(h5_path, mesh_root_dir=mesh_dir)


        with h5py.File(h5_path, 'r+') as traj_ds:
            if sum('pitch' in k for k in traj_ds.keys()) >= traj_per_object: # already done
                return True

            for i in range(traj_per_object):
                ## Define floor and object
                support_mesh = trimesh.creation.cylinder(radius=100, height=1)

                ## Create Scene object and add the meshes
                scene = PyrenderScene()
                scene.add_object("support_object", support_mesh, pose=np.eye(4), support=True)

                ## Define object location and add object to scene
                obj_pose = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.527],
                    [0, 0, 0, 1]
                ])

                scene.add_object("obj0", m, pose=obj_pose, support=False)

                ## Create rendering machinery: SceneRenderer and camera
                renderer = SceneRenderer(scene, width=300, height=300)
                trimesh_camera = renderer.get_trimesh_camera()

                ## Create random trajectory
                poses, traj_name = make_trajectory(trimesh_camera, obj_pose[:3, 3], num_frames=frames_per_traj)
                # TODO: save a relative pose, not the camera pose. This will make it possible to do a simple multiplication to transform.

                ## Step through trajectory and render images
                color_ims = []
                depth_ims = []
                pcs = []
                seg_ims = []
                for i, camera_pose in enumerate(poses):
                    color, depth, pc, segmentation = renderer.render(camera_pose=camera_pose, target_id="obj0")

                    color_ims.append(color)
                    depth_ims.append(depth)
                    pcs.append(pc)
                    seg_ims.append(segmentation)

                # Turn lists of images into ndarrays
                color_ims = np.stack(color_ims)
                depth_ims = np.stack(depth_ims)
                pcs = np.stack(pcs)
                seg_ims = np.stack(seg_ims)

                # Turn list of poses into ndarray
                poses = np.stack(poses)

                # Get transformation from camera frame to object frame
                tf_from_cam_to_scene = poses.dot(
                    trimesh.transformations.euler_matrix(np.pi, 0, 0)
                ) # camera_pose has a flipped z axis
                tf_from_scene_to_obj = inverse_homo(obj_pose)
                tf_from_cam_to_obj = tf_from_scene_to_obj @ tf_from_cam_to_scene

                # Read the successful contact points from the h5 file
                contact_pts = np.asarray(traj_ds['grasps/contact_points']).astype('float')
                contact_pts = np.concatenate(contact_pts)
                success = np.asarray(traj_ds['grasps/qualities/flex/object_in_gripper'])

                ## Store images and poses into h5py group
                # We name the group with a short desription of the trajectory, like "/pitch:1.09,yaw0:0.82"

                traj_ds.create_group(traj_name)
                # TODO: make a name we can iterate over more easily

                cds = traj_ds[traj_name].create_dataset(
                    "color", np.shape(color_ims), h5py.h5t.STD_U8BE, data=color_ims, compression="gzip", compression_opts=9, chunks=np.shape(color_ims)
                )

                dds = traj_ds[traj_name].create_dataset(
                    "depth", np.shape(depth_ims), h5py.h5t.IEEE_F32BE, data=depth_ims, compression="gzip", compression_opts=9, chunks=np.shape(depth_ims)
                )
                    
                sds = traj_ds[traj_name].create_dataset(
                    "segmentation", np.shape(seg_ims), h5py.h5t.STD_U8BE, data=depth_ims, compression="gzip", compression_opts=9, chunks=np.shape(seg_ims)
                )

                pds = traj_ds[traj_name].create_dataset(
                    "poses", np.shape(poses), h5py.h5t.IEEE_F64BE, data=poses, compression="gzip", compression_opts=9, chunks=np.shape(poses)
                )
                p_co_ds = traj_ds[traj_name].create_dataset(
                    "tf_from_cam_to_obj", np.shape(tf_from_cam_to_obj), h5py.h5t.IEEE_F64BE, data=tf_from_cam_to_obj, compression="gzip", compression_opts=9, chunks=np.shape(tf_from_cam_to_obj)
                )
            return True

    except Exception as e:
        print(e)
        return False

def render_trajectories(cfg : DictConfig):

    # Dynamically import tools from ACRONYM 
    # for loading meshes and rendering perspectives
    sys.path.insert(0, cfg.ACRONYM_REPO)
    from acronym_tools import load_mesh
    from scripts.acronym_render_observations import PyrenderScene, SceneRenderer

    h5_paths = [os.path.join(cfg.DS_DIR, p) for p in os.listdir(cfg.DS_DIR) if p.endswith('.h5')]
    
    make_trajectories = partial(make_trajectories_for_path,
        traj_per_object = cfg.TRAJ_PER_OBJECT,
        mesh_dir = cfg.MESH_DIR,
        load_mesh = load_mesh,
        PyrenderScene = PyrenderScene,
        SceneRenderer = SceneRenderer,
        frames_per_traj = cfg.FRAMES_PER_TRAJ
    )

    ## DEBUG
    # h5_paths = h5_paths[:2]
    # Save trajectory data for each .h5 grasp file, in parallel.
    # successes = make_trajectories(h5_paths[0])
    with Pool(cfg.PROCESSES) as p:
        successes = list(
            tqdm(
                p.imap_unordered(make_trajectories, h5_paths, chunksize=1),
                total=len(h5_paths)
            )
        )
    
    print("Successes:")
    for s in successes: print(s)

    with open('trajectory_vid_out.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(h5_paths, successes))
