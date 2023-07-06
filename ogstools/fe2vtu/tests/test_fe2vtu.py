import subprocess


def test_cli():
    subprocess.run(["fe2vtu", "--help"], check=True)
