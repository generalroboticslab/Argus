"""Patch IsaacGymEnvs so it imports without `urdfpy`.

IsaacGymEnvs' `tasks/__init__.py` eagerly imports the IndustReal tasks, and
`industreal_algo_utils.py` does a module-level `from urdfpy import URDF`. urdfpy
0.0.22 hard-pins networkx 2.2, whose `graphml.py` uses the removed `np.int`
alias -- so merely importing isaacgymenvs crashes on any modern numpy.

Argus does not use the IndustReal tasks. This script makes that single import
lazy (moved inside the only function that uses it), so importing isaacgymenvs
no longer needs urdfpy at all. No version downgrades required.

Idempotent: safe to run any number of times. Run after installing IsaacGymEnvs:

    python patch_isaacgymenvs.py
"""
import os
import sys

TOP_LEVEL_IMPORT = "from urdfpy import URDF\n"
LAZY_COMMENT = "  # lazy import: only IndustReal tasks need urdfpy"


def find_target():
    try:
        import isaacgymenvs
    except ImportError:
        sys.exit("ERROR: isaacgymenvs is not installed (see README step 3)")
    path = os.path.join(
        os.path.dirname(isaacgymenvs.__file__),
        "tasks", "industreal", "industreal_algo_utils.py",
    )
    if not os.path.isfile(path):
        sys.exit(f"ERROR: expected file not found: {path}")
    return path


def main():
    path = find_target()
    with open(path) as f:
        lines = f.readlines()

    if TOP_LEVEL_IMPORT not in lines:
        already = any("from urdfpy import URDF" in ln for ln in lines)
        print(("Already patched: " if already else "Nothing to patch: ") + path)
        return

    out = []
    inserted = False
    for line in lines:
        if line == TOP_LEVEL_IMPORT:
            continue  # drop the crashing module-level import
        if not inserted and line.lstrip().startswith("urdf = URDF.load("):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}from urdfpy import URDF{LAZY_COMMENT}\n")
            inserted = True
        out.append(line)

    if not inserted:
        sys.exit("ERROR: could not locate `urdf = URDF.load(` to add lazy import")

    with open(path, "w") as f:
        f.writelines(out)
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
