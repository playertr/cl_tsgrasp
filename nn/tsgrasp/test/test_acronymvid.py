from tsgrasp.data.acronymvid import AcronymVidDataset, minkowski_collate_fn
from test.fixtures import acronymvid_cfg, acronymvid_dataset

def test_acronymvid_dataset(acronymvid_cfg):
    ds = AcronymVidDataset(acronymvid_cfg)
    res = ds[0]
    print("done")


def test_minkowski_collate_fn(acronymvid_dataset):
    d0 = acronymvid_dataset[20]
    d1 = acronymvid_dataset[35]

    coll = minkowski_collate_fn([d0, d1])

    print(coll)
    print("done")