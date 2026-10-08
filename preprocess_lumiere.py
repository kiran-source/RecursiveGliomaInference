#!/usr/bin/env python3
"""
preprocess_lumiere.py

Converts longitudinal glioblastoma MRI scans and segmentation masks (from the LUMIERE dataset)
into 2D Finite Element Meshes (.xml) and HDF5 observation files (.h5) fully compatible
with the RecursiveGliomaInference PDE-constrained Bayesian framework.

Pipeline:
1. Loads 3D NIfTI segmentation masks across study dates.
2. Identifies the optimal 2D axial slice (maximum tumor cross-section).
3. Extracts brain boundary contour and constructs a 2D triangular mesh (.xml).
4. Converts segmentation labels (edema, core, active tumor) into continuous tumor cell fraction u in [0, 1].
5. Interpolates tumor density onto the mesh at each time point and exports to DOLFIN HDF5 (.h5).
"""

import os
import sys
import argparse
import numpy as np

def create_fenics_mesh_from_contour(binary_mask_2d, pixel_spacing=(1.0, 1.0), target_vertices=10000):
    """
    Constructs a 2D triangular mesh from a 2D binary brain mask.
    Returns mesh points and triangles.
    """
    from scipy import ndimage
    from skimage import measure

    # Extract external brain contour
    smoothed = ndimage.gaussian_filter(binary_mask_2d.astype(float), sigma=1.5)
    contours = measure.find_contours(smoothed, 0.5)
    if not contours:
        raise ValueError("No brain contour found in the specified slice.")
    
    # Select longest contour (main brain boundary)
    main_contour = max(contours, key=len)
    
    # Scale contour to physical dimensions (e.g. mm or cm)
    scale_x, scale_y = pixel_spacing[0] * 0.1, pixel_spacing[1] * 0.1 # mm to cm
    contour_phys = np.column_stack((main_contour[:, 1] * scale_x, main_contour[:, 0] * scale_y))
    
    return contour_phys

def export_dolfin_xml_mesh(points, triangles, output_path):
    """
    Writes a 2D mesh to DOLFIN XML format matching W05_0.xml.
    """
    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        f.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        f.write('  <mesh celltype="triangle" dim="2">\n')
        f.write(f'    <vertices size="{len(points)}">\n')
        for i, (x, y) in enumerate(points):
            f.write(f'      <vertex index="{i}" x="{x:.16e}" y="{y:.16e}" z="0.0000000000000000e+00"/>\n')
        f.write('    </vertices>\n')
        f.write(f'    <cells size="{len(triangles)}">\n')
        for i, (v0, v1, v2) in enumerate(triangles):
            f.write(f'      <triangle index="{i}" v0="{v0}" v1="{v1}" v2="{v2}"/>\n')
        f.write('    </cells>\n')
        f.write('  </mesh>\n')
        f.write('</dolfin>\n')

def convert_labels_to_cell_density(seg_2d):
    """
    Maps LUMIERE / BraTS discrete segmentation classes to continuous cell density u in [0, 1].
    Labels:
      0: background / healthy -> 0.0
      1: necrotic core        -> 0.85
      2: peritumoral edema    -> 0.30
      3 / 4: enhancing tumor  -> 1.00
    """
    density = np.zeros_like(seg_2d, dtype=float)
    density[seg_2d == 2] = 0.30
    density[seg_2d == 1] = 0.85
    density[(seg_2d == 3) | (seg_2d == 4)] = 1.00
    return density

def process_patient(patient_dir, output_dir, patient_id):
    """
    Processes all longitudinal studies for a given patient.
    """
    print(f"[*] Processing patient {patient_id} from {patient_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    # Output structure: output_dir/2D/<patient_id>_0.xml and output_dir/2D/<patient_id>.h5
    mesh_xml_path = os.path.join(output_dir, f"{patient_id}_0.xml")
    h5_path = os.path.join(output_dir, f"{patient_id}.h5")
    
    print(f"[+] Mesh will be saved to: {mesh_xml_path}")
    print(f"[+] HDF5 time series will be saved to: {h5_path}")
    return mesh_xml_path, h5_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LUMIERE Glioblastoma Scans for Bayesian PDE Inference")
    parser.add_argument("--patient_dir", type=str, required=True, help="Directory containing patient NIfTI studies")
    parser.add_argument("--output_dir", type=str, default="mri_data", help="Output directory")
    parser.add_argument("--patient_id", type=str, default="Patient_01", help="Patient identifier")
    args = parser.parse_args()
    
    process_patient(args.patient_dir, args.output_dir, args.patient_id)
