"""Read-only mesh diagnostics: inspection report and derived fix suggestions."""

import json
import struct
from pathlib import Path
from typing import Any, Mapping, Optional

from .types import (
    MeshOpContext,
    MeshOpExecutionError,
    MeshOpResult,
    MeshOpUnavailableError,
)


# Above this triangle count, suggest a decimation pass (slicers and the viewer
# both struggle well before the generators' multi-million-face outputs).
HIGH_FACE_COUNT = 500_000
DECIMATE_TARGET_FACES = 200_000
# Mirrors the repair op's max_hole_size bounds.
MAX_HOLE_SIZE_BOUNDS = (10, 10000)


def _trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise MeshOpUnavailableError(
            "inspect_mesh: trimesh is not available on this system"
        ) from exc
    return trimesh


def _scene_meshes(loaded, trimesh_module) -> list[tuple]:
    """Return (geometry, world transform or None) for every non-empty triangle mesh."""
    if not isinstance(loaded, trimesh_module.Scene):
        candidates = [(loaded, None)]
    else:
        candidates = []
        for node in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node]
            candidates.append((loaded.geometry.get(geometry_name), transform))
    return [
        (geometry, transform)
        for geometry, transform in candidates
        if isinstance(geometry, trimesh_module.Trimesh) and len(geometry.faces)
    ]


def _has_uvs(geometry, trimesh_module) -> bool:
    visual = geometry.visual
    if not isinstance(visual, trimesh_module.visual.TextureVisuals):
        return False
    return visual.uv is not None and len(visual.uv) > 0


def _gltf_has_normals(document: Mapping[str, Any]) -> bool:
    primitives = [
        primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        # Only triangle primitives (mode 4, the default) are inspected.
        if primitive.get("mode", 4) == 4
    ]
    return bool(primitives) and all(
        "NORMAL" in primitive.get("attributes", {}) for primitive in primitives
    )


def _file_has_normals(input_path: Path) -> Optional[bool]:
    """Whether the file stores vertex normals; None when the format is unknown.

    trimesh computes normals on demand, so its meshes always have them; the
    file itself is the only reliable source.
    """
    suffix = input_path.suffix.lower()
    try:
        if suffix == ".glb":
            with open(input_path, "rb") as handle:
                header = handle.read(20)
                if len(header) < 20 or header[:4] != b"glTF":
                    return None
                chunk_length = struct.unpack_from("<I", header, 12)[0]
                if header[16:20] != b"JSON":
                    return None
                return _gltf_has_normals(json.loads(handle.read(chunk_length)))
        if suffix == ".gltf":
            return _gltf_has_normals(
                json.loads(input_path.read_text(encoding="utf-8"))
            )
        if suffix == ".obj":
            with open(input_path, "r", encoding="utf-8", errors="replace") as handle:
                return any(line.startswith("vn ") for line in handle)
        if suffix == ".ply":
            with open(input_path, "rb") as handle:
                header = handle.read(4096).split(b"end_header")[0]
            return b" nx" in header
        if suffix == ".stl":
            # STL only stores per-face normals.
            return False
    except (OSError, ValueError):
        return None
    return None


def _merged_geometry(parts, trimesh_module):
    """Concatenate the parts into one position-merged mesh for topology checks.

    Visuals are dropped on purpose: UV seams split vertices, and without merging
    by position every textured mesh would look open along its seams.
    """
    import numpy as np

    vertices = []
    faces = []
    offset = 0
    for geometry, transform in parts:
        points = np.asarray(geometry.vertices, dtype=float)
        if transform is not None:
            points = trimesh_module.transformations.transform_points(points, transform)
        vertices.append(points)
        faces.append(np.asarray(geometry.faces, dtype=np.int64) + offset)
        offset += len(points)

    return trimesh_module.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=True,
    )


def _edge_topology(merged, trimesh_module) -> dict[str, Any]:
    import numpy as np

    edges, counts = np.unique(merged.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    loops = (
        trimesh_module.graph.connected_components(boundary, min_len=1)
        if len(boundary)
        else []
    )
    return {
        "boundary_edge_count": int(len(boundary)),
        "non_manifold_edge_count": int(np.count_nonzero(counts > 2)),
        "hole_count": int(len(loops)),
        # A simple boundary loop has as many edges as vertices.
        "largest_hole_edges": int(max((len(loop) for loop in loops), default=0)),
    }


def _self_intersecting_faces(merged, context: MeshOpContext) -> Optional[int]:
    try:
        import pymeshlab
    except ImportError:
        context.log("Self-intersection check skipped: pymeshlab is unavailable")
        return None

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(vertex_matrix=merged.vertices, face_matrix=merged.faces)
    )
    try:
        mesh_set.compute_selection_by_self_intersections_per_face()
    except Exception as exc:
        context.log(f"Self-intersection check skipped: {exc}")
        return None
    return int(mesh_set.current_mesh().selected_face_number())


def _printable(report: Mapping[str, Any]) -> Optional[bool]:
    """True/False when the verdict is certain; None when it hinges on an unchecked test."""
    if not (
        report["is_watertight"]
        and report["is_winding_consistent"]
        and report["is_manifold"]
    ):
        return False
    if report["self_intersecting_face_count"] is None:
        return None
    return report["self_intersecting_face_count"] == 0


def _issues(report: Mapping[str, Any]) -> list[str]:
    issues = []
    if report["hole_count"]:
        issues.append(
            f"{report['hole_count']} hole(s) "
            f"({report['boundary_edge_count']} open boundary edges)"
        )
    if report["non_manifold_edge_count"]:
        issues.append(
            f"{report['non_manifold_edge_count']} non-manifold edge(s)"
        )
    if not report["is_winding_consistent"]:
        issues.append("inconsistent face winding (normals flipped on some faces)")
    if report["self_intersecting_face_count"]:
        issues.append(
            f"{report['self_intersecting_face_count']} self-intersecting face(s)"
        )
    if report["self_intersecting_face_count"] is None:
        issues.append("self-intersections were not checked")
    if report["degenerate_face_count"]:
        issues.append(f"{report['degenerate_face_count']} degenerate face(s)")
    if report["component_count"] > 1:
        issues.append(f"{report['component_count']} disconnected parts")
    return issues


def build_report(
    input_path: Path,
    context: MeshOpContext,
    check_self_intersections: bool = True,
) -> dict[str, Any]:
    """Diagnose a mesh file without modifying it; every value is JSON-safe."""
    trimesh = _trimesh()

    context.progress(10, "Loading mesh…")
    loaded = trimesh.load(str(input_path), process=False)
    parts = _scene_meshes(loaded, trimesh)
    if not parts:
        raise MeshOpExecutionError(
            f"inspect_mesh: {input_path.name} contains no triangle mesh"
        )

    context.progress(30, "Analysing topology…")
    merged = _merged_geometry(parts, trimesh)
    topology = _edge_topology(merged, trimesh)
    is_watertight = bool(merged.is_watertight)
    is_winding_consistent = bool(merged.is_winding_consistent)

    self_intersections = None
    if check_self_intersections:
        context.progress(60, "Checking self-intersections…")
        self_intersections = _self_intersecting_faces(merged, context)

    context.progress(90, "Measuring…")
    bounds = merged.bounds
    report: dict[str, Any] = {
        "vertex_count": int(sum(len(geometry.vertices) for geometry, _ in parts)),
        "face_count": int(sum(len(geometry.faces) for geometry, _ in parts)),
        "component_count": int(merged.body_count),
        "is_watertight": is_watertight,
        "is_winding_consistent": is_winding_consistent,
        "is_manifold": topology["non_manifold_edge_count"] == 0,
        **topology,
        "degenerate_face_count": int(
            len(merged.faces) - int(merged.nondegenerate_faces().sum())
        ),
        "self_intersecting_face_count": self_intersections,
        "bounds": {
            "min": [float(value) for value in bounds[0]],
            "max": [float(value) for value in bounds[1]],
        },
        "dimensions": [float(value) for value in merged.extents],
        "surface_area": float(merged.area),
        # The signed volume of an open or mis-wound mesh is meaningless.
        "volume": (
            float(abs(merged.volume))
            if is_watertight and is_winding_consistent
            else None
        ),
        "has_uvs": any(_has_uvs(geometry, trimesh) for geometry, _ in parts),
        "has_normals": _file_has_normals(input_path),
    }
    report["printable"] = _printable(report)
    report["issues"] = _issues(report)
    context.log(
        f"Inspected {report['face_count']} faces: "
        + ("printable" if report["printable"] else "; ".join(report["issues"]))
    )
    context.progress(100, "Done")
    return report


def fixes_for_report(report: Mapping[str, Any]) -> tuple[list[dict], list[str]]:
    """Derive registry op calls from a report, plus notes for unfixable issues."""
    fixes: list[dict] = []
    notes: list[str] = []

    repair_reasons = []
    if report["hole_count"]:
        repair_reasons.append(f"fill {report['hole_count']} hole(s)")
    if report["non_manifold_edge_count"]:
        repair_reasons.append("fix non-manifold edges")
    if report["degenerate_face_count"]:
        repair_reasons.append("remove degenerate faces")
    if repair_reasons:
        low, high = MAX_HOLE_SIZE_BOUNDS
        fixes.append(
            {
                "op": "repair",
                "params": {
                    "remove_duplicates": True,
                    "remove_degenerate": bool(report["degenerate_face_count"]),
                    "fix_non_manifold": bool(report["non_manifold_edge_count"]),
                    "fill_holes": bool(report["hole_count"]),
                    "max_hole_size": max(
                        low, min(high, report["largest_hole_edges"] or low)
                    ),
                },
                "reason": "Repair to " + ", ".join(repair_reasons) + ".",
            }
        )
        if report["largest_hole_edges"] > high:
            notes.append(
                f"The largest hole has {report['largest_hole_edges']} boundary "
                f"edges, above the repair limit of {high}; it may stay open."
            )

    if report["face_count"] > HIGH_FACE_COUNT:
        fixes.append(
            {
                "op": "decimate",
                "params": {"target_faces": DECIMATE_TARGET_FACES},
                "reason": (
                    f"{report['face_count']} faces is heavy; reduce to "
                    f"{DECIMATE_TARGET_FACES}."
                ),
            }
        )

    if not report["is_winding_consistent"]:
        notes.append("Inconsistent winding has no automatic fix yet.")
    if report["self_intersecting_face_count"]:
        notes.append("Self-intersections have no automatic fix yet.")
    if report["component_count"] > 1:
        notes.append(
            f"{report['component_count']} disconnected parts; remove stray "
            "fragments manually if they are not intended."
        )
    return fixes, notes


def inspect_mesh(
    input_path: Path,
    params: Mapping[str, Any],
    context: MeshOpContext,
) -> MeshOpResult:
    report = build_report(
        input_path,
        context,
        check_self_intersections=bool(params.get("check_self_intersections", True)),
    )
    return MeshOpResult(details=report)


def suggest_fixes(
    input_path: Path,
    params: Mapping[str, Any],
    context: MeshOpContext,
) -> MeshOpResult:
    report = build_report(
        input_path,
        context,
        check_self_intersections=bool(params.get("check_self_intersections", True)),
    )
    fixes, notes = fixes_for_report(report)
    return MeshOpResult(details={"report": report, "fixes": fixes, "notes": notes})
