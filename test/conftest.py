"""Skip the tests that need the full geo2d package when only the stand-in is available.

`tools/geo2d_lite` draws domains and meshes them, but it does not implement everything
geo2d does (arc projection, mesh degree tables), so these tests run only against geo2d.
"""
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEEDS_FULL_GEO2D = (
    "test/test_envs/test_geo2d_bridge.py::TestGeo2DBridge::test_env_builds_with_par_zero_for_lattice_polygons",
    "test/test_envs/test_geo2d_bridge.py::TestGeo2DBridge::test_resmooth_keeps_corners_and_boundary",
    "test/test_envs/test_geo2d_bridge.py::TestGeo2DBridge::test_round_trip_to_geo2d_mesh",
    "test/test_game/test_curved_levels.py::TestClockwiseCurvedLoops::test_a_clockwise_loop_keeps_its_arcs_on_their_own_vertices",
)


def _using_stand_in():
    try:
        from src.geo2d_bridge import import_geo2d
        return "geo2d_lite" in (import_geo2d().__file__ or "")
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if not _using_stand_in():
        return
    skip = pytest.mark.skip(reason="needs the full geo2d package, not the tools/geo2d_lite stand-in")
    for item in items:
        if item.nodeid in NEEDS_FULL_GEO2D:
            item.add_marker(skip)
