"""Dependency regression checks for a fresh Argus environment.

Validates the dependency fixes that let the repo install cleanly on modern
package versions (no numpy-1.19 downgrade pinning):

  * numpy >= 1.20 is active and `np.int` is gone (the alias that crashed
    the old urdfpy -> networkx 2.2 import chain).
  * networkx imports without the `np.int` AttributeError.
  * yourdfpy imports and its URDF API (base_link / joint_map / get_transform)
    works -- this is the code path that replaced urdfpy in
    envs/tasks/legged_terrain.py.
  * urdfpy is no longer required (it should be absent; its mere import used
    to crash regardless of whether URDF was used).

These checks do NOT require Isaac Gym or a GPU. Full end-to-end training is a
separate, GPU-only tier handled by test_fresh_env.sh.

Exit code 0 = all required checks passed.
"""
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed = []
failed = []


def check(name, fn):
    try:
        detail = fn()
        passed.append((name, detail))
        print(f"[PASS] {name}" + (f" -- {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        failed.append((name, repr(e)))
        print(f"[FAIL] {name} -- {e!r}")


def numpy_modern():
    import numpy as np
    assert tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 20), \
        f"numpy too old: {np.__version__}"
    assert not hasattr(np, "int"), "np.int still present (numpy < 1.24?)"
    return f"numpy {np.__version__}, np.int removed"


def networkx_imports():
    # The original crash: `import networkx` -> graphml.py -> np.int AttributeError.
    import networkx as nx
    nx.DiGraph()
    return f"networkx {nx.__version__}"


def yourdfpy_imports():
    import yourdfpy
    return f"yourdfpy {getattr(yourdfpy, '__version__', '?')}"


def urdfpy_absent():
    import importlib.util
    spec = importlib.util.find_spec("urdfpy")
    assert spec is None, "urdfpy is still installed; it should be removed (it crashes on import)"
    return "urdfpy not installed (expected)"


def yourdfpy_api_path():
    # Mirrors the migrated asymmetry-design block in legged_terrain.py.
    import yourdfpy
    urdfs = sorted(glob.glob(os.path.join(REPO_ROOT, "assets", "**", "*.urdf"), recursive=True))
    if not urdfs:
        raise AssertionError("no .urdf assets found to exercise the API")
    # Prefer a urdf with non-fixed joints (a robot, not a static prop).
    for path in urdfs:
        urdf = yourdfpy.URDF.load(path)
        base = urdf.base_link
        moving = [j for j in urdf.joint_map.values()
                  if j.type != "fixed" and j.child in urdf.link_map]
        if not moving:
            continue
        for j in moving:
            T = urdf.get_transform(frame_to=j.child, frame_from=base)
            assert T.shape == (4, 4), f"bad transform shape {T.shape}"
        return f"{os.path.relpath(path, REPO_ROOT)}: {len(moving)} moving joints OK"
    raise AssertionError("no urdf with non-fixed joints found")


check("numpy modern (np.int removed)", numpy_modern)
check("networkx imports", networkx_imports)
check("yourdfpy imports", yourdfpy_imports)
check("urdfpy absent", urdfpy_absent)
check("yourdfpy URDF API (legged_terrain path)", yourdfpy_api_path)

# Informational only -- isaacgymenvs is a separate git install, not in requirements.txt.
try:
    import isaacgymenvs  # noqa: F401
    print("[INFO] isaacgymenvs present")
except Exception:
    print("[INFO] isaacgymenvs not installed (install separately; see README step 3)")

print()
print(f"Required checks: {len(passed)} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
