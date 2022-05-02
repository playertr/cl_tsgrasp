"""
trajectory_vid.py
A module for creating RGB-D videos annotated with grasps from the ACRONYM dataset.
Tim Player playert@oregonstate.edu July 22, 2021
"""

from posixpath import split
import trimesh
import numpy as np
import os
import h5py
from . import params
from shutil import copyfile
import torch
from pytorch3d.ops import knn_points
import csv

import sys

sys.path.insert(0, "/home/playert/Research/acronym")
from scripts.acronym_render_observations import SceneRenderer, PyrenderScene

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def make_trajectory(trimesh_camera, obj_loc, seed=None, N=params.FRAMES_PER_TRAJ):
    # Create a "random" trajectory that views the object.
    # Initially, these trajectories will be circular orbits that always look directly at the object, at different elevation angles.

    # More creative distributions of trajectories, including ones in which
    # the image-frame object center is not in the middle of the path, are 
    # future work.
    
    np.random.seed(seed)

    # The pitch angle is a uniform random angle 
    avg_pitch = np.pi/3
    pitch_range = 0.16 # roughly 10 degrees
    pitch = np.random.uniform(avg_pitch-pitch_range/2, avg_pitch+pitch_range/2)
    
    # A full circle is swept out, with a random initial yaw angle.
    yaw0 = np.random.uniform(0, 2*np.pi)
    name = f"pitch:{pitch:.2f},yaw0:{yaw0:.2f}" # quick description
    
    d0 = 2 # Orbital distance
    
    yaws = np.linspace(yaw0, yaw0+2*np.pi, N)
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

def load_mesh(filepath, mesh_root_dir, scale=None):
    """Load a mesh from a HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    Args:
        filepath (str): HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    if filepath.endswith(".h5"):
        data = h5py.File(filepath, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        # meshes/ChestOfDrawers/9a1281e8357bdd629f92caeb1d84eac1.obj
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filepath)
    
    mesh_fname = os.path.basename(mesh_fname)

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh

def grasp_labels(obj_pts, pos_contact_pts, epsilon, use_GPU=False):
    """ Generate binary labels for camera-frame points in `pts` based on proximity to object-frame points in `pos_contact_pts`"""

    # Convert object and depth camera points to batched Torch tensors
    if use_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    batch_size = obj_pts.shape[0]
    obj_pts = torch.Tensor(obj_pts).to(device=device)
    pos_contact_pts = torch.tile(torch.Tensor(pos_contact_pts).to(device=device), (batch_size, 1, 1))

    dists, idx, nn = knn_points(obj_pts, pos_contact_pts, K=1)
    success = (dists < epsilon)

    return idx.cpu().numpy(), success.cpu().numpy()

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

def make_trajectories_for_path(gp):
    """Generate videos along a trajectory for the grasp file at path `gp` and save them to disk."""
    try:
        ## Copy this h5py file into the grasp directory, so we can append the trajectories to the h5 file
        traj_file = os.path.join(params.TRAJDIR, os.path.basename(gp))
        if not os.path.exists(traj_file):
            os.makedirs(params.TRAJDIR, exist_ok=True)
            copyfile(gp, traj_file)
        else:
            return True

        ## Load mesh corresponding to this grasp file
        m = load_mesh(gp, mesh_root_dir=params.OBJDIR)


        traj_ds = h5py.File(traj_file, 'r+')

        for i in range(params.TRAJ_PER_OBJECT):
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
            poses, traj_name = make_trajectory(trimesh_camera, obj_pose[:3, 3])
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
            good_contacts = contact_pts[np.where(success==1)]

            # Use KNN to create grasp quality labels
            obj_frame_pcs = transform(pcs[:,:,:3], tf_from_cam_to_obj)
            nearest_grasp_idx, labels = grasp_labels(obj_frame_pcs,good_contacts, epsilon=params.SUCCESS_RADIUS)
            nearest_grasp_idx = nearest_grasp_idx.reshape(-1, 300, 300)
            label_ims = labels.reshape(-1, 300, 300)
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

            lds = traj_ds[traj_name].create_dataset(
                "grasp_labels", np.shape(label_ims), h5py.h5t.STD_U8BE, data=label_ims, compression="gzip", compression_opts=9, chunks=np.shape(label_ims)
            )

            ngds = traj_ds[traj_name].create_dataset(
                "nearest_grasp_idx", np.shape(nearest_grasp_idx), h5py.h5t.STD_U16BE, data=nearest_grasp_idx, compression="gzip", compression_opts=9
            )

        traj_ds.close()
        return True

    except Exception as e:
        print(e)
        return False

def main():

    ## Get files containing .obj meshes
    # files are like "a238b2f6f5e2cf66c77016426fd9cf48.obj" and are separated into a hash and an absolute path.
    obj_paths = []
    hashes = []
    for fname in os.listdir(params.OBJDIR):
        if '.obj' in fname:
            h, _ = os.path.splitext(fname)
            hashes.append(h)
            obj_paths.append(os.path.join(params.OBJDIR, fname))

    ## Get files containing corresponding .h5 grasp files
    # files are like "Book_a238b2f6f5e2cf66c77016426fd9cf48_0.000621589326212787.h5", and the last token is the scale of the mesh (multiple scales may be given for a single object). We create a list of tuples (obj_path, [grasp_paths]) containing file paths relevant to a given object.

    grasp_paths_with_obj = []
    grasp_paths = [os.path.join(params.GRASPDIR, p) for p in os.listdir(params.GRASPDIR) if '.h5' in p]

    for p in grasp_paths:
        for h in hashes:
            if h in p:
                grasp_paths_with_obj.append(p)


    if params.DEBUG:
        make_trajectories_for_path(grasp_paths_with_obj[1])
    else:

        split1 = int(0.7 * len(grasp_paths_with_obj))
        split2 = int(0.85 * len(grasp_paths_with_obj))

        # grasp_paths_with_obj = grasp_paths_with_obj[0:split1]
        grasp_paths_with_obj = grasp_paths_with_obj[split2:]

        # make_trajectories_for_path(grasp_paths_with_obj[0])
        
        # Save trajectory data for each .h5 grasp file, in parallel.
        with Pool(cpu_count()-2) as p:
            successes = list(
                tqdm(
                    p.imap_unordered(make_trajectories_for_path, grasp_paths_with_obj),
                    total=len(grasp_paths_with_obj)
                )
            )
        
        print("Successes:")
        for s in successes: print(s)

        with open('trajectory_vid_out.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(grasp_paths, successes))

if __name__ == '__main__':
    main()