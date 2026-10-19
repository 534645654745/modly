"""Built-in mesh operation definitions."""

from .inspection import inspect_mesh, suggest_fixes
from .operations import decimate_mesh, repair_mesh, smooth_mesh
from .types import MeshOp


REPAIR_PARAMS = (
    {
        "id": "remove_duplicates",
        "label": "Remove Duplicates",
        "type": "boolean",
        "default": True,
        "tooltip": "Remove duplicate vertices and faces.",
    },
    {
        "id": "remove_degenerate",
        "label": "Remove Degenerate Faces",
        "type": "boolean",
        "default": True,
        "tooltip": "Remove zero-area faces and collapsed edges.",
    },
    {
        "id": "fix_non_manifold",
        "label": "Fix Non-Manifold",
        "type": "boolean",
        "default": True,
        "tooltip": "Detach faces causing non-manifold edges.",
    },
    {
        "id": "fill_holes",
        "label": "Fill Holes",
        "type": "boolean",
        "default": True,
        "tooltip": (
            "Fill simple boundary holes. Structural holes from AI generation "
            "may not be fillable in post-processing."
        ),
    },
    {
        "id": "max_hole_size",
        "label": "Max Hole Size",
        "type": "int",
        "default": 2000,
        "min": 10,
        "max": 10000,
        "tooltip": (
            "Maximum number of boundary edges of a hole to be filled. "
            "Increase if large holes remain open."
        ),
    },
)

DECIMATE_PARAMS = (
    {
        "id": "target_faces",
        "label": "Target Triangles",
        "type": "int",
        "default": 10000,
        "min": 100,
        "max": 1000000,
        "tooltip": "Target number of triangles after simplification.",
    },
)

SMOOTH_PARAMS = (
    {
        "id": "iterations",
        "label": "Iterations",
        "type": "int",
        "default": 5,
        "min": 1,
        "max": 50,
        "tooltip": (
            "Number of smoothing passes. More iterations = smoother result "
            "but may lose fine details."
        ),
    },
    {
        "id": "lambda_",
        "label": "Smoothing Strength",
        "type": "float",
        "default": 0.5,
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "tooltip": (
            "Controls how far each vertex moves toward its neighbours per "
            "iteration. Lower = more conservative."
        ),
    },
    {
        "id": "mode",
        "label": "Mode",
        "type": "select",
        "default": "taubin",
        "options": [
            {"value": "taubin", "label": "Taubin (volume-preserving)"},
            {"value": "laplacian", "label": "Laplacian (stronger, may shrink)"},
        ],
        "tooltip": (
            "Taubin alternates positive/negative steps to prevent mesh "
            "shrinkage. Laplacian is simpler but tends to shrink the mesh over "
            "many iterations."
        ),
    },
)


INSPECT_PARAMS = (
    {
        "id": "check_self_intersections",
        "label": "Check Self-Intersections",
        "type": "boolean",
        "default": True,
        "tooltip": (
            "Detect intersecting faces. The slowest check on very dense meshes."
        ),
    },
)


BUILTIN_MESH_OPS = (
    MeshOp(
        id="repair",
        label="Repair Mesh",
        params_schema=REPAIR_PARAMS,
        fn=repair_mesh,
        category="repair",
    ),
    MeshOp(
        id="decimate",
        label="Optimize Mesh",
        params_schema=DECIMATE_PARAMS,
        fn=decimate_mesh,
        category="optimization",
    ),
    MeshOp(
        id="smooth",
        label="Smooth Mesh",
        params_schema=SMOOTH_PARAMS,
        fn=smooth_mesh,
        category="optimization",
    ),
    MeshOp(
        id="inspect_mesh",
        label="Inspect Mesh",
        params_schema=INSPECT_PARAMS,
        fn=inspect_mesh,
        category="analysis",
    ),
    MeshOp(
        id="suggest_fixes",
        label="Suggest Fixes",
        params_schema=INSPECT_PARAMS,
        fn=suggest_fixes,
        category="analysis",
    ),
)
