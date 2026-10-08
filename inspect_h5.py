import h5py

with h5py.File("W05_data/2D/W05.h5", "r") as f:
    print("Root keys:", list(f.keys()))
    group = f["W05"]
    print("W05 keys:", list(group.keys()))
    subgroup = group["W05_0"]
    print("W05_0 keys:", list(subgroup.keys()))
    print("W05_0/vector shape:", subgroup["vector"].shape, "dtype:", subgroup["vector"].dtype)
    print("Attributes in W05_0:", dict(subgroup.attrs))
    print("Attributes in W05_0/vector:", dict(subgroup["vector"].attrs))
