# how to generated the meshes

```shell
max=7
for (( i=0; i <= $max; ++i ))
do
    n=$((2**$i))
    generateStructuredMesh -e quad -o square_$i.vtu --lx 1 --ly 1 --lz 0 --nx $n --ny $n --nz 1
done
```

# how to run the simulations

```python
from ogs6py import ogs

from ogstools.meshlib import MeshSeries

for i in range(8):
    model = ogs.OGS(
        INPUT_FILE="./square_neumann.prj",
        PROJECT_FILE=f"./square_neumann_{i}.prj",
    )
    model.replace_mesh("square.vtu", f"square_{i}.vtu")
    model.replace_text(f"square_1e0_neumann_{i}", "./time_loop/output/prefix")
    model.write_input()
    model.run_model(logfile=f"out_{i}.log")

base_id = 3
base_mesh = MeshSeries("./square_1e0_neumann_0.pvd").read(-1)
base_mesh.save("./square_neumann_convergence_study_res_0.vtu")
for i in range(base_id + 1, 8):
    mesh = MeshSeries(f"./square_1e0_neumann_{i-base_id}.pvd").read(-1)
    base_mesh = base_mesh.sample(mesh)
    base_mesh.save(f"./square_neumann_convergence_study_res_{i-base_id}.vtu")
```
