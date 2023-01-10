import subprocess


def test_cli():
    subprocess.run(["msh2vtu", "--help"], check=True)
