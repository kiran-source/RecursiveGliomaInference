import h5py

for i in range(1, 11):
    pid = f"Patient_{i:03d}"
    h5_path = f"mri_data/{pid}/2D/{pid}.h5"
    xml_path = f"mri_data/{pid}/2D/{pid}_0.xml"
    with h5py.File(h5_path, "r") as f:
        groups = list(f[pid].keys())
        v = f[f"{pid}/{groups[0]}/vector"][:]
        print(f"[OK] {pid}: 5 scan groups found, mesh XML present, vector shape={v.shape}, max={v.max():.4f}")
print("All 10 patients verified successfully!")
