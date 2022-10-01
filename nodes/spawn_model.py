#! /home/playert/miniconda3/envs/tsgrasp/bin/python

# Script to spawn models in gazebo given an OBJ path and inertial properties
# from dataclasses import dataclass
import h5py
import numpy as np
# from pathlib import Path
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, Quaternion, Point
import rospy
import os
from os.path import join

# @dataclass
class Color:
    # red, green, blue, alpha
    r = 0.0 # type: float
    g = 0.0 # type: float
    b = 0.0 # type: float
    a = 0.0 # type: float
    r_2 = 0.0
    g_2 = 0.0
    b_2 = 0.0

    def __init__(self, r, g, b, a):
        self.r, self.g, self.b, self.a = r, g, b, a
        self.r_2 = r/2
        self.g_2 = g/2
        self.b_2 = b/2

# @dataclass
class OBJObject:
    name = None # type: str
    mass = 0.0 # type: float
    inertia= None #: np.ndarray
    mesh_file = None # type: str
    scale = 0.0 # type: float
    color = 0.0 # type: Color
    friction = 0.0 # type: float
    center_of_mass = None # type: np.ndarray

    def __init__(self, name, mass, inertia, mesh_file, scale, color, friction, center_of_mass):
        self.name, self.mass, self.inertia, self.mesh_file, self.scale, self.color, self.friction, self.center_of_mass = name, mass, inertia, mesh_file, scale, color, friction, center_of_mass

    def to_sdf(self):
        return """
<?xml version='1.0'?>
<sdf version="1.4">
<model name="{name}">
    <pose>0 0 0 0 0 0</pose>
    <static>false</static>
    <self_collide>0</self_collide>
    <enable_wind>0</enable_wind>
    <link name="link">
    <inertial>
        <pose>{center_of_mass[0]} {center_of_mass[1]} {center_of_mass[2]} 0 0 0</pose>
        <mass>{mass}</mass>
        <inertia> 
        <ixx>{inertia[0][0]}</ixx>
        <ixy>{inertia[0][1]}</ixy> 
        <ixz>{inertia[0][2]}</ixz>
        <iyy>{inertia[1][1]}</iyy>
        <iyz>{inertia[1][2]}</iyz>
        <izz>{inertia[2][2]}</izz>
        </inertia>
    </inertial>
    <collision name="collision">
        <geometry>
            <mesh>
                <uri>file://{mesh_file}</uri>
                <scale>{scale} {scale} {scale}</scale>
            </mesh>
        </geometry>
        <max_contacts>100</max_contacts>
        <surface>
            <contact>
                <ode>
                <max_vel>0.0</max_vel>
                <min_depth>0.0</min_depth>
                </ode>
            </contact>
            <friction>
                <ode>
                <mu>{friction}</mu>
                <mu2>{friction}</mu2>
                </ode>
                <torsional>
                <ode/>
                </torsional>
            </friction>
            <bounce/>
        </surface>
    <physics>
        <ode>
        <limit>
            <cfm>0</cfm>
            <erp>0.2</erp>
        </limit>
        <implicit_spring_damper>1</implicit_spring_damper>
        </ode>
    </physics>
    </collision>
    <visual name="visual">
        <geometry>
            <mesh>
                <uri>file://{mesh_file}</uri>
                <scale>{scale} {scale} {scale}</scale>
            </mesh>
        </geometry>
        <material>
            <ambient>{color.r_2} {color.r_2} {color.r_2} {color.a}</ambient>
            <diffuse>{color.r} {color.g} {color.b} {color.a}</diffuse>
            <specular>{color.r} {color.g} {color.b} {color.a}</specular>
            <emissive>0 0 0 0</emissive>
        </material>
    </visual>
    </link>
</model>
</sdf>
""".format(name=self.name, mass=self.mass, inertia=self.inertia, mesh_file=self.mesh_file, scale=self.scale, friction=self.friction, color=self.color, center_of_mass=self.center_of_mass)

def get_obj(h5_path, mesh_dir):
    with h5py.File(h5_path, 'r') as ds:

        mass = ds['object/mass'][()]
        inertia = np.asarray(ds['object/inertia'])
        scale = ds['object/scale'][()]
        friction = ds['object/friction'][()]
        try:
            file = ds['object/file'][()].decode('utf-8')
        except AttributeError as e:
            file = ds['object/file'][()]
        center_of_mass = np.asarray(ds['object/com'])
        # file is like 
        # meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1.obj
    
    color = Color(215, 63, 9, 1)
    # file = Path(mesh_dir) / Path(*Path(file).parts[1:]) 
    # name = Path(file).stem
    file = join(mesh_dir, os.path.join(*(file.split(os.path.sep)[1:])))
    name = file.split(os.path.sep)[-2]

    print(file)
    return OBJObject(name=name, mass=mass, inertia=inertia, mesh_file=file, scale=scale, color=color, friction=friction, center_of_mass=center_of_mass)

class ObjectDataset():
    def __init__(self, dataset_dir, split):
        self.root = os.path.join(dataset_dir, split)
        self.mesh_dir = os.path.join(self.root, "meshes")
        self.h5_dir = os.path.join(self.root, "h5")
        self.split = split

        # Find the raw filepaths.
        if split in ["train", "val", "test"]:
            self._paths = [os.path.join(self.h5_dir, f) for f in sorted(os.listdir(self.h5_dir))]
        else:
            raise ValueError("Split %s not recognised" % split)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        h5_path = self._paths[idx]
        return get_obj(h5_path, self.mesh_dir)

if __name__ == "__main__":

    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    print("Got it.")


    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    obj = get_obj(
        '/home/playert/Research/tsgrasp/data/dataset/train/h5/Xbox_e9f575a666fc26aeeb6dfb094ca5145_0.00019886029112129727.h5',
        mesh_dir='/home/playert/Research/tsgrasp/data/dataset/train/meshes'
    )

    xml = obj.to_sdf()


    # print(f"Spawning model: {'foobar'}")
    item_pose   =   Pose(
        position=Point(x=0.2, y=0.2, z=0.5),
        orientation=Quaternion(x=0, y=0, z=0, w=1)
    )

    try:
        resp1 = spawn_model("foobar", xml, "", item_pose, "world")
        print(xml)
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))
        print(resp1)
    
    
