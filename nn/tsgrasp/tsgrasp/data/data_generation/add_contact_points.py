"""
add_contact_points.py
Script for determining the two 3D contact points of 6DOF grasps from the ACRONYM dataset. Uses utilities from Contact GraspNet.
Tim Player, playert@oregonstate.edu, July 29 2021
"""

import os
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import csv
from omegaconf import DictConfig
from functools import partial
import sys
from rtree.exceptions import RTreeError

def append_grasp_info(h5_path, mesh_root, cfg):
    """Generate contact points and append them to a single file."""
    if cfg.CONTACT_GRASPNET_REPO not in sys.path:
        sys.path.insert(0, cfg.CONTACT_GRASPNET_REPO)
    from tools.create_contact_infos import grasps_contact_info
    if cfg.ACRONYM_REPO not in sys.path:
        sys.path.insert(0, cfg.ACRONYM_REPO)
    from acronym_tools import load_mesh, load_grasps

    # Check to see whether this file has been modified yet
    with h5py.File(h5_path) as ds:
        if 'grasps/contact_points' in ds: # Already did this one
            return True

    # load mesh. Not all meshes were successfully made, so wrap in Try/Exc
    try:
        obj_mesh = load_mesh(filename=h5_path, mesh_root_dir=mesh_root)
    except ValueError as e:
        print(e)
        # breakpoint()
        return False

    T, success = load_grasps(filename=h5_path)

    try:
        contact_dicts = grasps_contact_info(grasp_tfs=T, successfuls=success, obj_mesh=obj_mesh, check_collisions=False)
    except AttributeError as e: # NoneType object has no attribute 'faces'
        print(e)
        # breakpoint()
        return False
    except RTreeError as e: # obscure, maybe memory-related error?
        print(e)
        # breakpoint()
        return False

    # `contact_dicts` is a list of dictionaries, with one dict per grasp.
    # Not every grasp successfully generated a dictionary (maybe because of buggy collision checking?)

    # However, we want to associate every grasp with a set of contact points in the h5 file.

    # The following trickery with `transform_idxs` constructs a dictionary of the indices corresponding to each transform. For speed, dictionaries are used to look up the correspondences. Because ndarrays are an unhashable type, the `.tobytes()` method is called on them, creating a byte representation. This technique is potentially brittle: it assumes that `grasps_contact_info` will not break the bitwise representation of the grasp transformation, which might not be the case with all versions of ContactGraspNet or all systems.
    print(f"{len(contact_dicts)} contact points identified.")

    transform_idxs = {}
    for i, d in enumerate(contact_dicts):
        transform_idxs[d['grasp_transform'].tobytes()] = i
    
    # Create an ndarray containing the two 3D grasp contacts for each grasp.
    contact_points = np.nan * np.ones(shape=(T.shape[0], 2, 3))
    for i, t in enumerate(T):
        if t.tobytes() in transform_idxs:
            idx_of_points = transform_idxs[t.tobytes()]
            contact_points[i] = contact_dicts[idx_of_points]['contact_points']

    # Add the `contact_points` ndarray to this file.
    with h5py.File(h5_path, 'r+') as ds:
        ds["grasps/"].create_dataset(
                "contact_points", np.shape(contact_points), h5py.h5t.IEEE_F32BE, data=contact_points
        )

    return True

def add_contact_points(cfg: DictConfig):

    # Dynamically import tools from ACRONYM and Contact-Graspnet
    # for loading meshes and adding contact points.
    # We import this way so that we can specify the path to their repos in yaml.
    # We make them global so we're not passing expensive arguments between processes.


    h5_paths = [os.path.join(cfg.DS_DIR, f) for f in os.listdir(cfg.DS_DIR) if f.endswith(".h5")]

    append_grasp = partial(append_grasp_info, 
        mesh_root=cfg.MESH_DIR,
        cfg = cfg
    )

    ## DEBUG
    # h5_paths = h5_paths[:50]

    if cfg.PROCESSES == 0:
        successes = [append_grasp(path) for path in tqdm(h5_paths)]
    else:
        with Pool(cfg.PROCESSES) as p:
            successes = list(
                tqdm(
                    p.imap_unordered(append_grasp, h5_paths),
                    total=len(h5_paths)
                )
        )

    print("Successful:")
    print(successes)

    with open('add_contacts_out.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(h5_paths, successes))
