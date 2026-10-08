import sys
print("Python version:", sys.version)

try:
    import fenics
    print("FEniCS version:", fenics.__version__)
except Exception as e:
    print("FEniCS error:", e)

try:
    import dolfin as dl
    print("DOLFIN version:", dl.__version__)
except Exception as e:
    print("DOLFIN error:", e)

try:
    import ufl
    print("UFL version:", ufl.__version__)
except Exception as e:
    print("UFL error:", e)

try:
    import scipy
    print("SciPy version:", scipy.__version__)
except Exception as e:
    print("SciPy error:", e)

try:
    import numpy as np
    print("NumPy version:", np.__version__)
except Exception as e:
    print("NumPy error:", e)

try:
    sys.path.append('../hippylib')
    sys.path.append('../hippylib/hippylib')
    sys.path.append('/workspace/hippylib')
    sys.path.append('/workspace/hippylib/hippylib')
    import hippylib
    print("hIPPYlib imported successfully! Version:", getattr(hippylib, '__version__', 'unknown'))
    from hippylib import BiLaplacianPrior
    print("BiLaplacianPrior successfully imported!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("hIPPYlib error:", e)

print("Environment check completed.")

import dolfin as dl
mesh = dl.Mesh("/workspace/W05_data/2D/W05_0.xml")
print("Mesh successfully loaded! Num vertices:", mesh.num_vertices(), "Num cells:", mesh.num_cells())

Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
u = dl.Function(Vh2)
h5 = dl.HDF5File(mesh.mpi_comm(), "/workspace/W05_data/2D/W05.h5", "r")
for t, group in [(2, "/W05/W05_0"), (4, "/W05/W05_1"), (5, "/W05/W05_2"), (6, "/W05/W05_3"), (9, "/W05/W05_4")]:
    h5.read(u, group)
    print(f"Read time t={t} ({group}): norm = {u.vector().norm('l2'):.4f}, max = {u.vector().max():.4f}")
h5.close()
print("W05.h5 data read successfully!")

