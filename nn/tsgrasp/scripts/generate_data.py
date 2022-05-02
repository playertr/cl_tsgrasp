import shutil
import hydra
from omegaconf import DictConfig
from tsgrasp.data.data_generation.convert_meshes import convert_meshes
from tsgrasp.data.data_generation.add_contact_points import add_contact_points
from tsgrasp.data.data_generation.render_trajectories import render_trajectories
import os
import random

@hydra.main(config_path="../conf", config_name="scripts/generate_data")
def main(cfg : DictConfig):

    # print("####################################################")
    # print(f"Simplifying the meshes in {cfg.MESH_DIR} "
    #     f"and writing the results to {cfg.convert_meshes.OUTPUT_DIR} .")
    # convert_meshes(cfg.convert_meshes)

    # print("####################################################")
    # print(f"Generating grasp contact points.")
    # # First, copy cfg.ACRONYM_DIR to cfg.OUTPUT_DATASET_DIR.
    # # Then, add contact points to the h5 files in the new dataset.
    # if not os.path.exists(cfg.OUTPUT_DATASET_DIR):
    #     copytree(cfg.ACRONYM_DIR, cfg.OUTPUT_DATASET_DIR)
    # add_contact_points(cfg.add_contact_points)

    # print("####################################################")
    # print(f"Rendering trajectories.")
    # render_trajectories(cfg.render_trajectories)

    print("####################################################")
    print(f"Separating into train and test sets.")

    ## Read all of the categories (folders) from the object directory and shuffle them
    random.seed(42)

    def get_h5_info(h5_path):
        return {
            "path": h5_path,
            "category": os.path.basename(h5_path).split("_")[0]
        }
    def get_obj_info(obj_path):
        return {
            "path": obj_path,
            "category": os.path.basename(obj_path)
        }
    def paths(dir):
        return (os.path.join(dir, d) for d in os.listdir(dir))

    mesh_infos = [get_obj_info(p) for p in paths(cfg.SIMPLIFIED_MESH_DIR)]
    h5_infos = [get_h5_info(p) for p in paths(cfg.OUTPUT_DATASET_DIR) if p.endswith('.h5')]

    object_categories = list(set(info['category'] for info in mesh_infos))
    random.shuffle(object_categories)

    ## Divide into train and test set by object category
    split = int(len(object_categories) * cfg.TRAIN_SPLIT)
    train_cats = object_categories[:split]
    test_cats = object_categories[split:]

    ## Create train and test directories, each with an "/obj" and "/h5" folder.
    train_dir = cfg.OUTPUT_DATASET_DIR + "train/"
    test_dir = cfg.OUTPUT_DATASET_DIR + "test/"
    mesh_train_dir = train_dir + "meshes/"
    mesh_test_dir = test_dir + "meshes/"
    h5_train_dir = train_dir + "h5/"
    h5_test_dir  = test_dir + "h5/"

    for dir in [mesh_train_dir, mesh_test_dir, h5_train_dir, h5_test_dir]:
        os.makedirs(dir, exist_ok=True)

    ## Move train and test objects into respective directories
    def move_item(info, dest):
        shutil.move(info['path'], dest)
        print(f"shutil.move({info['path']}, {dest})")

    for mesh_info in mesh_infos:
        if mesh_info['category'] in train_cats:
            move_item(mesh_info, mesh_train_dir)
        elif mesh_info['category'] in test_cats:
            move_item(mesh_info, mesh_test_dir)
        else:
            raise ValueError

    for h5_info in h5_infos:
        if h5_info['category'] in train_cats:
            move_item(h5_info, h5_train_dir)
        elif h5_info['category'] in test_cats:
            move_item(h5_info, h5_test_dir)
        else:
            raise ValueError
    
if __name__ == "__main__":
    main()