from hydra import initialize, compose

def test_top_level_keys_in_config():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config")

    for key in ["model", "data", "training"]:
        assert key in cfg.keys()