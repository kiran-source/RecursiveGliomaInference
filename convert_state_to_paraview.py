import os
import h5py
import numpy as np

def convert_state_trajectory(h5_in, xdmf_out_path, h5_out_path):
    if not os.path.exists(h5_in):
        print(f"Error: {h5_in} not found.")
        return

    f_in = h5py.File(h5_in, 'r')
    f_out = h5py.File(h5_out_path, 'w')

    times = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    num_states = len(times)

    # Base mesh from state_0
    s0 = f_in['state/state_0']
    topo = np.array(s0['mesh/topology'])
    geom = np.array(s0['mesh/geometry'])
    cell_dofs = np.array(s0['cell_dofs']).reshape(-1, 6)
    n_vertices = geom.shape[0]

    # Map each mesh vertex to its DOF index
    vertex_to_dof = -np.ones(n_vertices, dtype=int)
    for c in range(len(topo)):
        for k in range(3):
            v_idx = topo[c, k]
            dof_idx = cell_dofs[c, k]
            vertex_to_dof[v_idx] = dof_idx

    # Save mesh in output HDF5
    mesh_grp = f_out.create_group("Mesh")
    mesh_grp.create_dataset("topology", data=topo, compression="gzip")
    mesh_grp.create_dataset("geometry", data=geom, compression="gzip")

    vis_grp = f_out.create_group("VisualisationVector")

    # XDMF header
    h5_filename_only = os.path.basename(h5_out_path)
    xdmf_lines = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        '  <Domain>',
        '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">'
    ]

    for idx, t in enumerate(times):
        state_key = f"state_{idx}"
        if state_key not in f_in['state']:
            continue
        
        vec = np.array(f_in[f'state/{state_key}/vector'])
        # Extract nodal tumor field u
        u_nodal = vec[vertex_to_dof].reshape(n_vertices, 1)

        vis_grp.create_dataset(str(idx), data=u_nodal, compression="gzip")

        xdmf_lines.extend([
            f'      <Grid Name="mesh" GridType="Uniform">',
            f'        <Topology NumberOfElements="{len(topo)}" TopologyType="Triangle" NodesPerElement="3">',
            f'          <DataItem Dimensions="{len(topo)} 3" NumberType="UInt" Format="HDF">{h5_filename_only}:/Mesh/topology</DataItem>',
            f'        </Topology>',
            f'        <Geometry GeometryType="XY">',
            f'          <DataItem Dimensions="{n_vertices} 2" Format="HDF">{h5_filename_only}:/Mesh/geometry</DataItem>',
            f'        </Geometry>',
            f'        <Time Value="{t:.1f}" />',
            f'        <Attribute Name="u" AttributeType="Scalar" Center="Node">',
            f'          <DataItem Dimensions="{n_vertices} 1" Format="HDF">{h5_filename_only}:/VisualisationVector/{idx}</DataItem>',
            f'        </Attribute>',
            f'      </Grid>'
        ])

    xdmf_lines.extend([
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>'
    ])

    f_in.close()
    f_out.close()

    with open(xdmf_out_path, 'w') as f_x:
        f_x.write("\n".join(xdmf_lines) + "\n")

    print(f"Generated clean ParaView file: {xdmf_out_path}")

if __name__ == "__main__":
    # Convert for Patient_001
    p1_dir = "BayesianInference/results/Patient_001"
    if os.path.exists(os.path.join(p1_dir, "state_trajectory.h5")):
        convert_state_trajectory(
            os.path.join(p1_dir, "state_trajectory.h5"),
            os.path.join(p1_dir, "state_trajectory_paraview.xdmf"),
            os.path.join(p1_dir, "state_trajectory_paraview.h5")
        )
    # Also convert for root BayesianInference if present
    if os.path.exists("BayesianInference/state_trajectory.h5"):
        convert_state_trajectory(
            "BayesianInference/state_trajectory.h5",
            "BayesianInference/state_trajectory_paraview.xdmf",
            "BayesianInference/state_trajectory_paraview.h5"
        )
