from contact_torchnet.net.contact_torchnet import ContactTorchNet, add_s_loss, compute_labels, unbuild_6dof_grasps, build_6dof_grasps
from contact_torchnet.utils.config_utils import load_config
import pytest

import torch

@pytest.fixture
def ctn():
    global_config = load_config("conf/cgn_config.yaml")
    return ContactTorchNet(global_config)


def test_ContactTorchNet_forward(ctn):
    xyz = torch.rand((3, 20000, 3))
    cat = ctn.forward(xyz)

def test_transform_vec():
    from contact_torchnet.net.contact_torchnet import transform_vec

    x = torch.arange(12.0).reshape(2, 2, 3)
    tf = torch.stack([
        torch.eye(4), torch.eye(4)
    ])

    x_new = transform_vec(x, tf)

    assert torch.equal(x, x_new)

    tf2 = torch.stack([
        torch.Tensor([
            [1, 0, 0, 1.],
            [0, 1, 0, 2.],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        torch.Tensor([
            [1, 0, 0, 3.],
            [0, 1, 0, 4.],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    ])

    x_new2 = transform_vec(x, tf2)
    diff = x_new2 - x
    assert torch.equal(
        diff,  
        torch.Tensor([
            [[ 1.,  2.,  0.],
            [ 1.,  2.,  0.]],

            [[ 3.,  4.,  0.],
            [ 3.,  4.,  0.]]])
    )

    x2 = torch.arange(6.0).reshape(2, 1, 3)
    x_new3 = transform_vec(x2, tf2)
    diff2 = x_new3 - x2
    assert torch.equal(
        diff2,  
        torch.Tensor([
            [[ 1.,  2.,  0.]],

            [[ 3.,  4.,  0.]]])
    )

def test_transform_mat():
    from contact_torchnet.net.contact_torchnet import transform_mat

    pose = torch.stack([
        torch.eye(4), torch.eye(4)
    ])
    pose = torch.stack([pose, pose])

    tf = torch.stack([
        torch.Tensor([
            [1, 0, 0, 1.],
            [0, 1, 0, 2.],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        torch.Tensor([
            [1, 0, 0, 3.],
            [0, 1, 0, 4.],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    ])

    new_pose = transform_mat(pose, tf)

    print("cat")

def test_build_unbuild_6dof_grasp():
    baseline_dir = torch.Tensor([
        [1, 0, 0]
    ])
    approach_dir = torch.Tensor([
        [0, 0, 1]
    ])
    contact_points = torch.Tensor([
        [0, 0, 0]
    ])
    grasp_width = torch.Tensor([
        [1]
    ])
    
    grasps = build_6dof_grasps(contact_points, baseline_dir, approach_dir, grasp_width)
    ad, gd = unbuild_6dof_grasps(grasps)

    assert torch.equal(ad, approach_dir)
    assert torch.equal(gd, baseline_dir)

    # Test random, low-dim case.
    baseline_dir = torch.rand(1, 3)
    approach_dir = torch.rand(1, 3)
    contact_points = torch.rand(1, 3)
    grasp_width = torch.rand(1, 1)
    
    grasps = build_6dof_grasps(contact_points, baseline_dir, approach_dir, grasp_width)
    ad, gd = unbuild_6dof_grasps(grasps)

    assert torch.allclose(ad, approach_dir)
    assert torch.allclose(gd, baseline_dir)

    # Test higher-dim case with broadcasting
    baseline_dir = torch.rand(5, 7, 2, 3)
    approach_dir = torch.rand(5, 7, 2, 3)
    contact_points = torch.rand(5, 7, 2, 3)
    grasp_width = torch.rand(5, 7, 2, 1)
    
    grasps = build_6dof_grasps(contact_points, baseline_dir, approach_dir, grasp_width)
    ad, gd = unbuild_6dof_grasps(grasps)

    assert torch.allclose(ad, approach_dir)
    assert torch.allclose(gd, baseline_dir)

def test_compute_labels():
    """ pred_points (torch.Tensor): (b, n_pred_grasp, 3) inference positions (camera frame)
        gt_grasp_tfs (torch.Tensor): (b, n_gt_grasp, 4, 4) ground truth grasp poses (scene frame)
        gt_contact_pts (torch.Tensor): (b, n_gt_grasp, 2, 3) ground truth mesh contact points (scene frame)
        gt_widths (torch.Tensor): (b, n_gt_grasp, 1) ground truth grasp widths
        cam_pose (torch.Tensor): (b, 4, 4) camera poses
    """

    pred_points = torch.rand(2, 2, 3)
    gt_grasp_tfs = torch.rand(2, 3, 4, 4)
    gt_contact_pts = torch.rand(2, 3, 2, 3)

    gt_widths = torch.rand(2, 3, 1)
    cam_pose = torch.stack([
        torch.eye(4),
        torch.eye(4)
    ])
    
    grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = compute_labels(
        pred_points, gt_grasp_tfs, gt_contact_pts, gt_widths, cam_pose)

def test_loss(ctn):

    ## Create grasp predictions
    device=torch.device('cuda')
    ctn = ctn.to(device)
    xyz = torch.rand((2, 256, 3), device=device)
    (baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points
    ) = ctn.forward(xyz)
    # dummy convert grasp offset to scalar, not binned
    grasp_offset = 0.1 * torch.argmax(grasp_offset, dim=-1).unsqueeze(-1)

    ## Compute grasp labels at the predicted grasp locations
    gt_grasp_tfs = torch.rand(2, 3, 4, 4, device=device)
    gt_contact_pts = torch.rand(2, 3, 2, 3, device=device)
    gt_widths = torch.rand(2, 3, 1, device=device)
    cam_pose = torch.stack([
        torch.eye(4, device=device),
        torch.eye(4, device=device)
    ])
    
    grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = compute_labels(
        pred_points, gt_grasp_tfs, gt_contact_pts, gt_widths, cam_pose)

    ## Test add-s loss
    cp_tensor = torch.rand(1, 5, 3, device=device)
    sym_cp_tensor = torch.rand(1, 5, 3, device=device)
    loss = add_s_loss(baseline_dir, binary_seg, grasp_offset, approach_dir, 
    pred_points, baseline_dir_label, grasp_width_label, approach_dir_label, grasp_success_label, cp_tensor, sym_cp_tensor)

from hydra import compose, initialize
from contact_torchnet.data.acronymvid import AcronymVidDataset
def test_cgn_on_data(ctn):
    device=torch.device('cuda')

    with initialize(config_path="../conf"):
        cfg = compose(config_name="config")

    avd = AcronymVidDataset(cfg.data.data_cfg)
    dl = torch.utils.data.DataLoader(avd, shuffle=True)


    data = {k: v.to(device) for k, v in next(iter(dl)).items()}

    ## Create grasp predictions
    ctn = ctn.to(device)
    xyz = data['positions'][0,:2,:,:]
    (baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points
    ) = ctn.forward(xyz)

    ## Compute grasp labels at the predicted grasp locations
    gt_grasp_tfs = data['cam_frame_pos_grasp_tfs'][0,:2,:,:]
    gt_contact_pts = data['pos_contact_pts_mesh'].squeeze().repeat(2,1,1,1) #TODO do this in compute_labels?
    gt_widths = data['pos_finger_diffs'].squeeze(0).repeat(2, 1, 1)
    cam_pose = data['camera_pose'][0][:2,:,:]
    
    grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = compute_labels(
        pred_points, gt_grasp_tfs, gt_contact_pts, gt_widths, cam_pose)

    ## Test add-s loss
    cp_tensor = data['single_gripper_points']
    sym_cp_tensor = data['single_gripper_points'] # TODO -- make symmetric
    loss = ctn.loss(baseline_dir, torch.sigmoid(binary_seg), grasp_offset, approach_dir, 
    pred_points, baseline_dir_label, grasp_width_label, approach_dir_label, grasp_success_label, cp_tensor, sym_cp_tensor)