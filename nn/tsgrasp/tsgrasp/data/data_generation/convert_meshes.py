"""
convert_meshes.py
Process the .obj files and make them waterproof and simplified.
"""
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from omegaconf import DictConfig
from functools import partial
import h5py
import shutil

def process_hash(paths, obj_dir: str, manifold_path: str, simplify_path: str, out_dir: str, BAD_MESH_JAIL: str):
    """Process a single object file by calling a subshell with the mesh processing script.

    Args:
        paths (Tuple[str,str])): relative path like
            meshes/Lamp/14cbef4f4a67e57a4cf9c858305a22f8.obj, and full h5 file path
        obj_dir (str): root directory of all the meshes
        manifold_path (str): path to `manifold` executable
        simplify_path (str): path to `simplify` executable
        out_dir (str): parent directory for output mesh
        BAD_MESH_JAIL (str): place to but the .h5 files whose meshes didn't simplify
    """
    path, h5_path = paths
    outfile = out_dir + path
    if os.path.isfile(outfile): # already done
        return

    tokens = path.split("/")
    h = tokens[-1][:-4] # hash

    obj_path = obj_dir + 'models/' + h + ".obj"
    if not os.path.isfile(obj_path):
        print("Could not find original obj file.")
        return
    
    # Waterproof the object
    temp_name = f"temp.{h}.watertight.obj"
    completed = subprocess.run(["timeout", "-sKILL", "60", manifold_path, obj_path, temp_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if completed.returncode != 0:
        print(f"Skipping object (manifold failed): {h}")
        return
            
    # Simplify the object
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    completed = subprocess.run(["timeout", "-sKILL", "60", simplify_path, "-i", temp_name, "-o", outfile, "-m", "-r", "0.02"],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if completed.returncode != 0:
        print(f"Skipping object (simplify failed): {h}")
        print(f"Moving this .h5 grasp file ({h5_path}) to {BAD_MESH_JAIL}.")
        os.makedirs(os.path.dirname(BAD_MESH_JAIL), exist_ok=True)
        shutil.move(h5_path, BAD_MESH_JAIL)
        return
    
    try:
        os.remove(temp_name)
    except FileNotFoundError:
        print(f"That's weird! {temp_name} was not created!")

def convert_meshes(cfg: DictConfig):
    """Waterproof and simplify the OBJ meshes in cfg.GRASP_DIR."""

    ## Grab the hash names and desired paths from the grasp files
    paths = []
    h5_names = [n for n in os.listdir(cfg.GRASP_DIR) if n.endswith('.h5')]
    for fname in h5_names:
        h5_path = os.path.join(cfg.GRASP_DIR, fname)
        with h5py.File(h5_path, 'r') as ds:
            output_path = ds['object/file'][()].decode('utf-8')
            # meshes/Lamp/14cbef4f4a67e57a4cf9c858305a22f8.obj
            paths.append((output_path, h5_path))

    ## Wrap `process_hash` in a partial to remove arguments
    process_obj = partial(process_hash, 
        obj_dir = cfg.MESH_DIR,
        manifold_path = cfg.MANIFOLD_PATH,
        simplify_path = cfg.SIMPLIFY_PATH,
        out_dir = cfg.OUTPUT_DIR,
        BAD_MESH_JAIL = cfg.BAD_MESH_JAIL
    )

    ## DEBUG
    # paths = paths[:50]

    ## Issue the commands in a multiprocessing pool
    with Pool(cfg.PROCESSES) as p:
        examples = list(
            tqdm(
                p.imap_unordered(process_obj, paths),
                total=len(paths)
            )
        )