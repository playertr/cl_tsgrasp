
LOCAL_INTERACTIVE_DEBUG = False
if not LOCAL_INTERACTIVE_DEBUG:
    import pyglet
    pyglet.options['headless'] = True

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import trimesh
from PIL import Image
import io
import numpy as np
from tsgrasp.net.minkowski_graspnet import build_6dof_grasps
import imageio
import os

class GraspAnimationLogger(Callback):
    def __init__(self, cfg, example_batch: dict):
        super().__init__()
        self.save_outputs=cfg.save_outputs
        self.output_dir=cfg.output_dir
        self.batch = example_batch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        ## Run forward inference on the example batch
        pts = self.batch['positions'].to(pl_module.device)
        outputs = pl_module.forward(pts)

        gif_paths = animate_grasps_from_outputs(outputs)

        ## Send to C L O U D
        # wandb.log({
        #     "val/examples": [wandb.Image(path) 
        #                       for path in gif_paths]
        #     })

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        ## Run forward inference on this batch
        pts = batch['positions'].to(pl_module.device)
        outputs = pl_module.forward(pts)

        os.makedirs(self.output_dir, exist_ok=True)
        animate_grasps_from_outputs(outputs,
            name=os.path.join(self.output_dir, str(batch_idx))
        )

def animate_grasps_from_outputs(outputs, name=""):
    """
    Save GIFs overlaying the predicted grasps on the points clouds.
    Returns a list of CPU tensor animations.

    outputs: raw grasp parameters from module. (N_BATCH*N_TIME*N_PT, W_i)
    pts: point cloud. (N_BATCH, N_TIME, N_PT, 3)
    """

    class_logits, baseline_dir, approach_dir, grasp_offset, pts = outputs

    contact_pts = pts
    # ## Select the top 50 most likely grasps from each time step, for each 
    # # batch.
    confs = torch.sigmoid(class_logits)
    # _, idxs = torch.topk(confs, k=100, dim=2)

    # contact_pts = torch.gather(pts, dim=2, index=idxs.repeat(1, 1, 1, 3))
    # baseline_dir = torch.gather(baseline_dir, dim=2, index=idxs.repeat(1, 1, 1, 3))
    # approach_dir = torch.gather(approach_dir, dim=2, index=idxs.repeat(1, 1, 1, 3))
    # grasp_offset = torch.gather(grasp_offset, dim=2, index=idxs)


    ## Construct the 4x4 grasp poses
    grasp_tfs = build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_offset)

    gif_paths = []
    for batch_dim in range(len(pts)):
        ims = animate_grasps(
            pts[batch_dim].cpu().numpy(), grasp_tfs[batch_dim].cpu().numpy(), 
            confs[batch_dim].cpu().numpy()
        )
        path = f"{name}_{batch_dim}.gif"
        imageio.mimsave(path, ims)
        gif_paths.append(path)
        
    return gif_paths

def animate_grasps(pts, grasp_tfs, confs, pitches=None, res=(1080, 1080)):
    """
    Animate frames from multiple grasp predictions, raising the camera up.
    
    pts: (T, M, 3) point cloud
    grasp_tfs: list of T (N_t, 4, 4) grasp transforms
    pitches: list of T pitch angles of camera perspective. 0.55*2*np.pi works well.
    
    Returns a list of image ndarrays.
    """
    ims = []

    if pitches is None:
        # pitches = np.linspace(0.55*2*np.pi, 0.56*2*np.pi, len(pts))[::-1]
        pitches = [0.55*2*np.pi]*len(pts)

    for t, pitch in enumerate(pitches):
        ims.append(draw_grasps(pts[t], grasp_tfs[t], confs[t], pitch, res=res))

    return ims

def draw_grasps(pts, grasp_tfs, confs, pitch=0.55*2*np.pi, res=(1080, 1080)):
    """
    Draw the grasps on the point cloud.
    
    pts: (M, 3) point cloud in depth cam frame
    grasp_tfs: (N,4,4) grasp transforms in depth cam frame
    pitch: pitch angle of new camera perspective. 0.55*2*np.pi works well.
    """

    grasp_tfs = [grasp_tfs[i] for i in range(len(grasp_tfs)) if confs[i] > 0.5]

    pcl = trimesh.points.PointCloud(vertices=pts, r=100)
    pcl.visual.vertex_colors = trimesh.visual.interpolate(confs, color_map='viridis') # or color by depth: pts[:,2]

    grasp_markers = [create_gripper_marker().apply_transform(t) for t in       grasp_tfs]

    scene = trimesh.Scene([pcl, grasp_markers])

    roll, yaw = 0, 0
    cam_pose = scene.camera.look_at(
        points=pts,
        rotation=trimesh.transformations.euler_matrix(
                    pitch,
                    roll,
                    yaw,
        )
    )
    scene.camera_transform = cam_pose
    if LOCAL_INTERACTIVE_DEBUG: scene.show(viewer='gl')
    data = scene.save_image(resolution=res, visible=False) 
    im = Image.open(io.BytesIO(data))
    return np.asarray(im)

def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp
