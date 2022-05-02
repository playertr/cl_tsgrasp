import torch
from tsgrasp.net.tsgrasp_super import TSGraspSuper
import torch.nn.functional as F

def test_build_6dof_grasps():

    ## basic shape is as expected
    N = 2
    contact_pts = torch.rand((N, 3))
    baseline_dir = F.normalize(torch.rand(N, 3), dim=1)
    approach_dir = F.normalize(torch.rand(N, 3), dim=1)
    grasp_width = torch.rand(N, 1)

    grasps = TSGraspSuper.build_6dof_grasps(
        contact_pts, baseline_dir, approach_dir, grasp_width
    )

    assert grasps.shape == (2, 4, 4)

    ## unbuild undoes build

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
    
    grasps = TSGraspSuper.build_6dof_grasps(contact_points, baseline_dir, approach_dir, grasp_width)
    ad, gd = TSGraspSuper.unbuild_6dof_grasps(grasps)

    assert torch.equal(ad, approach_dir)
    assert torch.equal(gd, baseline_dir)

def test_control_point_tensor():
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

    grasp_tfs = TSGraspSuper.build_6dof_grasps(
            contact_pts=contact_points,
            baseline_dir=baseline_dir,
            approach_dir=approach_dir,
            grasp_width=grasp_width
    )
    cps = TSGraspSuper.control_point_tensors(grasp_tfs)

    assert cps.shape == (1, 5, 3)

def test_approx_min_dists():
    ## Test that shapes are okay
    N = 2
    M = 3
    pred_cp = torch.rand((N, 5, 3))
    gt_cp = torch.rand((M, 5, 3))

    dists = TSGraspSuper.approx_min_dists(pred_cp, gt_cp)

    assert dists.shape == (N,)

    # Test that identical tensors have difference of zero
    pred_cp = torch.arange(15.0).reshape(5, 3).repeat(N, 1, 1)
    gt_cp = torch.arange(15.0).reshape(5, 3).repeat(M, 1, 1)

    dists = TSGraspSuper.approx_min_dists(pred_cp, gt_cp)

    assert dists.sum() == 0

def test_adds_loss():

    ## Check that shapes are okay
    V = 2
    approach_dir = F.normalize(torch.rand((V, 3), requires_grad=True), dim=1)
    baseline_dir = F.normalize(torch.rand(V, 3), dim=1)
    positions = torch.rand((V, 3))
    grasp_width = torch.rand(V, 1)

    logits = torch.rand((V,1))
    labels = torch.rand((V,1)) > 0.5

    W = 3
    pos_grasp_tfs = torch.rand((W, 4, 4))


    adds_loss = TSGraspSuper.add_s_loss(approach_dir, baseline_dir, positions, grasp_width, logits, labels, pos_grasp_tfs)

    assert adds_loss.shape == torch.Size([])
    assert adds_loss > 0

def test_label_points():
    # xyz (torch.Tensor): (V, 3) point cloud from depth cam
    # contact_points (torch.Tensor): (W, 3) set of gripper contact points from positive grasps
    # radius (float): distance threshold to label points as positive
    V = 500
    W = 300

    # Create bimodal point cloud with two clusters
    xyz = torch.rand((V//2, 3))
    xyz = torch.cat([
        xyz, 
         torch.rand((V//2, 3)) + torch.Tensor([[500, 500, 500]])    
    ])

    contact_points = torch.rand((W, 3))

    labels = TSGraspSuper.label_points(xyz, contact_points, radius=5.0)

    assert torch.mean(labels.float()) == 0.5
    assert labels.shape == (V, 1)