# %%
import glob
import subprocess

from lxml import etree
from tqdm.auto import tqdm

folders = {
    "SMALL_DEFORMATION": "Mechanics",
    "HYDRO_MECHANICS": "HydroMechanics",
}

for process, folder_name in tqdm(folders.items()):
    prj_files = glob.glob(
        f"/home/fzill/opengeosys/ogs/Tests/Data/{folder_name}/**/*.prj",
        recursive=True,
    )
    # print(*prj_files, sep="\n")
    xsd = process + "_definition.xsd"
    subprocess.run(
        [
            *("java -jar ../build/trang.jar -I xml -O xsd".split(" ")),
            *(prj_files),
            xsd,
        ],
        check=False,
    )
    subprocess.run(["code", xsd], check=False)
    schema = etree.XMLSchema(etree.parse(xsd))
    for prj in tqdm(prj_files):
        schema.assert_(etree.parse(prj))

# %%
prj_files = glob.glob(
    "/home/fzill/opengeosys/ogs/Tests/Data/**/*.prj",
    recursive=True,
)
excludes = ["TwoPhaseFlowPrho", "SmallDeformationNonLocal", "TES", "StokesFlow"]
for exclude in excludes:
    prj_files = [prj for prj in prj_files if exclude not in prj]

# %%
# print(*prj_files, sep="\n")
xsd = "full_definition.xsd"
subprocess.run(
    [
        *("java -jar ../build/trang.jar -I xml -O xsd".split(" ")),
        *(prj_files),
        xsd,
    ],
    check=False,
)
subprocess.run(["code", xsd], check=False)

# %%
schema = etree.XMLSchema(etree.parse(xsd))
count = 0
for prj in tqdm(prj_files):
    if not schema.validate(etree.parse(prj)):
        print(schema.error_log)
        count += 1
print(count)

# %%
