from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from step71_common import OUTPUT_DIR, PROJECT_ROOT, jsonable, load_questions, normalize_answer, write_json, write_md
from step72_common import raw_objects_by_sample, separated_status
from step74a_common import CAMERA_CHANNELS, DEFAULT_MODEL_PATHS, count_by, image_readability, read_json, write_jsonl


STEP74B_PREFIX = "step74b_"
NUSC_ROOT = PROJECT_ROOT / "data" / "nuscenes"
NUSC_META = NUSC_ROOT / "v1.0-mini"
VIS_DIR = OUTPUT_DIR / "step74b_visual_evidence"


def out_path(name: str) -> Path:
    if not name.startswith(STEP74B_PREFIX):
        raise ValueError(f"Step 7.4B output must start with {STEP74B_PREFIX}: {name}")
    return OUTPUT_DIR / name


def rel(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def qwen_config() -> dict:
    model_path = next((str(Path(p)) for p in DEFAULT_MODEL_PATHS if p and Path(p).exists()), None)
    return {
        "backend_name": "local_qwen_vl_transformers",
        "model_path": model_path,
        "enable_local_mllm": bool(model_path),
        "device": os.getenv("LOCAL_QW_DEVICE", "cuda"),
        "max_new_tokens": int(os.getenv("LOCAL_QW_MAX_NEW_TOKENS", "180")),
        "min_pixels": int(os.getenv("STEP74B_MIN_PIXELS", str(256 * 28 * 28))),
        "max_pixels": int(os.getenv("STEP74B_MAX_PIXELS", str(1024 * 28 * 28))),
    }


def load_table(name: str) -> list[dict]:
    path = NUSC_META / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def metadata_indices() -> dict:
    tables = {name: load_table(name) for name in ["sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor"]}
    by_token = {name: {r["token"]: r for r in rows if "token" in r} for name, rows in tables.items()}
    sample_data_by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for sd in tables["sample_data"]:
        sensor = by_token["sensor"].get(by_token["calibrated_sensor"].get(sd.get("calibrated_sensor_token"), {}).get("sensor_token"), {})
        channel = sensor.get("channel")
        if sd.get("is_key_frame") and channel:
            sample_data_by_sample[str(sd.get("sample_token"))][channel] = sd
    return {"tables": tables, "by_token": by_token, "sample_data_by_sample": dict(sample_data_by_sample)}


def projection_resource_report() -> dict:
    idx = metadata_indices()
    sample_tokens = sorted({str(r.get("sample_token")) for r in load_questions(50)})
    has = {name: bool(idx["tables"][name]) for name in ["sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor"]}
    per_sample = {}
    for token in sample_tokens:
        chans = idx["sample_data_by_sample"].get(token, {})
        per_sample[token] = {
            "has_lidar_top": "LIDAR_TOP" in chans,
            "camera_channels": sorted([c for c in CAMERA_CHANNELS if c in chans]),
            "all_cameras": all(c in chans for c in CAMERA_CHANNELS),
        }
    return {
        "resources_present": has,
        "sample_count": len(sample_tokens),
        "sample_coverage": per_sample,
        "centerpoint_coordinate_frame": "assumed_lidar_top_frame",
        "coordinate_frame_safe": all(has.values()) and all(v["has_lidar_top"] and v["all_cameras"] for v in per_sample.values()),
        "transformation_chain": "CenterPoint/LIDAR_TOP -> ego at lidar timestamp -> global -> ego at camera timestamp -> camera -> image plane",
        "uses_gt_boxes": False,
        "uses_sample_annotations": False,
    }


def transform_matrix(translation: list[float], rotation: list[float], inverse: bool = False) -> np.ndarray:
    tm = np.eye(4)
    tm[:3, :3] = Quaternion(rotation).rotation_matrix
    tm[:3, 3] = np.array(translation, dtype=float)
    return np.linalg.inv(tm) if inverse else tm


def box_corners(center: list[float], size_wlh: list[float], yaw: float) -> np.ndarray:
    w, l, h = [float(x) for x in size_wlh[:3]]
    xs = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    ys = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    zs = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    pts = np.array([xs, ys, zs], dtype=float)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return (rot @ pts + np.array(center, dtype=float).reshape(3, 1)).T


def project_objects() -> tuple[list[dict], dict]:
    audit = projection_resource_report()
    if not audit["coordinate_frame_safe"]:
        return [], {"valid": False, "reason": "projection_resource_missing_or_incomplete", **audit}
    idx = metadata_indices()
    by = idx["by_token"]
    records: list[dict] = []
    errors = []
    raw = raw_objects_by_sample(50)
    for sample_token, objects in raw.items():
        chans = idx["sample_data_by_sample"].get(sample_token, {})
        lidar_sd = chans["LIDAR_TOP"]
        lidar_cal = by["calibrated_sensor"][lidar_sd["calibrated_sensor_token"]]
        lidar_pose = by["ego_pose"][lidar_sd["ego_pose_token"]]
        t_lidar_to_global = transform_matrix(lidar_pose["translation"], lidar_pose["rotation"]) @ transform_matrix(lidar_cal["translation"], lidar_cal["rotation"])
        for obj in objects:
            geom = obj.get("geometry") or {}
            center = geom.get("center_xyz")
            size = geom.get("size_wlh")
            yaw = geom.get("yaw")
            if not center or not size or yaw is None:
                errors.append({"sample_token": sample_token, "object_id": obj.get("object_id"), "error": "missing_box_geometry"})
                continue
            corners_lidar = box_corners(center, size, float(yaw))
            corners_lidar_h = np.c_[corners_lidar, np.ones(len(corners_lidar))]
            for camera in CAMERA_CHANNELS:
                cam_sd = chans[camera]
                cam_cal = by["calibrated_sensor"][cam_sd["calibrated_sensor_token"]]
                cam_pose = by["ego_pose"][cam_sd["ego_pose_token"]]
                t_global_to_cam = transform_matrix(cam_cal["translation"], cam_cal["rotation"], inverse=True) @ transform_matrix(cam_pose["translation"], cam_pose["rotation"], inverse=True)
                cam_pts = (t_global_to_cam @ t_lidar_to_global @ corners_lidar_h.T).T[:, :3]
                in_front = cam_pts[:, 2] > 0.1
                uv = []
                if np.any(in_front):
                    k = np.array(cam_cal["camera_intrinsic"], dtype=float)
                    proj = (k @ cam_pts[in_front].T).T
                    uv = proj[:, :2] / proj[:, 2:3]
                width, height = int(cam_sd.get("width") or 1600), int(cam_sd.get("height") or 900)
                valid = False
                bbox_un = bbox_cl = [None, None, None, None]
                area = trunc = 0.0
                if len(uv):
                    x1, y1 = np.min(uv, axis=0)
                    x2, y2 = np.max(uv, axis=0)
                    bbox_un = [float(x1), float(y1), float(x2), float(y2)]
                    cx1, cy1 = max(0.0, x1), max(0.0, y1)
                    cx2, cy2 = min(float(width - 1), x2), min(float(height - 1), y2)
                    bw, bh = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
                    unclipped = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    area = bw * bh
                    trunc = 1.0 - (area / unclipped) if unclipped > 0 else 1.0
                    bbox_cl = [float(cx1), float(cy1), float(cx2), float(cy2)]
                    valid = bool(area > 0 and bw >= 4 and bh >= 4)
                records.append({
                    "projection_record_id": f"{sample_token}_{obj['object_id']}_{camera}",
                    "sample_token": sample_token,
                    "object_id": obj["object_id"],
                    "category": obj.get("category"),
                    "camera": camera,
                    "projection_valid": valid,
                    "bbox_unclipped": bbox_un,
                    "bbox_clipped": bbox_cl,
                    "visible_corner_count": int(np.sum(in_front)),
                    "projected_area": round(float(area), 4),
                    "truncation_ratio": round(float(max(0.0, min(1.0, trunc))), 4),
                    "depth_m": round(float(np.median(cam_pts[in_front, 2])) if np.any(in_front) else float(np.median(cam_pts[:, 2])), 4),
                    "image_width": width,
                    "image_height": height,
                    "image_path": str(NUSC_ROOT / cam_sd["filename"]),
                    "geometry": {"center_xyz": center, "size_wlh": size, "yaw": yaw, "motion_status": separated_status(obj)["motion_status"]},
                    "source": "centerpoint_3d_projection",
                })
    audit = summarize_projection(records, errors)
    audit.update({"valid": True, "coordinate_frame": "CenterPoint boxes treated as LIDAR_TOP frame"})
    return records, audit


def summarize_projection(records: list[dict], errors: list[dict] | None = None) -> dict:
    valid = [r for r in records if r.get("projection_valid")]
    by_obj = defaultdict(int)
    for r in valid:
        by_obj[(r["sample_token"], r["object_id"])] += 1
    areas = [r["projected_area"] for r in valid]
    truncs = [r["truncation_ratio"] for r in valid]
    depths = [r["depth_m"] for r in valid]
    total_objects = len({(r["sample_token"], r["object_id"]) for r in records})
    return {
        "total_centerpoint_objects": total_objects,
        "object_camera_pairs": len(records),
        "valid_projection_pairs": len(valid),
        "valid_projection_pair_rate": len(valid) / len(records) if records else 0.0,
        "objects_projected_at_least_one_camera": len(by_obj),
        "objects_with_no_valid_camera_projection": total_objects - len(by_obj),
        "object_valid_projection_rate": len(by_obj) / total_objects if total_objects else 0.0,
        "valid_projections_by_camera": count_by(valid, "camera"),
        "projected_area": dist(areas),
        "truncation_ratio": dist(truncs),
        "depth_m": dist(depths),
        "multi_view_projection_count": dict(sorted(Counter(by_obj.values()).items())),
        "projection_errors": errors or [],
    }


def dist(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values) if values else None, "median": statistics.median(values) if values else None, "mean": statistics.mean(values) if values else None, "max": max(values) if values else None}


def select_cameras(records: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for r in records:
        if r.get("projection_valid"):
            grouped[(r["sample_token"], r["object_id"])].append(r)
    selections = []
    for key, rows in grouped.items():
        ranked = sorted(rows, key=lambda r: (-r["projected_area"], r["truncation_ratio"], -r["visible_corner_count"], r["depth_m"], r["camera"]))
        primary = ranked[0]
        secondary = None
        if len(ranked) > 1 and ranked[1]["projected_area"] >= max(64.0, primary["projected_area"] * 0.2):
            secondary = ranked[1]
        selections.append({"sample_token": key[0], "object_id": key[1], "category": primary.get("category"), "primary": primary, "secondary": secondary, "source": "projection_rank_fixed_policy"})
    audit = {"selected_objects": len(selections), "primary_by_camera": count_by([{"camera": s["primary"]["camera"]} for s in selections], "camera"), "with_secondary": sum(1 for s in selections if s.get("secondary"))}
    return sorted(selections, key=lambda s: (s["sample_token"], s["object_id"])), audit


def crop_box(bbox: list[float], width: int, height: int, pad_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    return (max(0, int(math.floor(x1 - px))), max(0, int(math.floor(y1 - py))), min(width, int(math.ceil(x2 + px))), min(height, int(math.ceil(y2 + py))))


def generate_visual_evidence(selections: list[dict]) -> tuple[list[dict], dict]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    packages = []
    for sel in selections:
        evidence_id = f"step74b_{sel['sample_token']}_{sel['object_id']}"
        sample_dir = VIS_DIR / sel["sample_token"] / sel["object_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        projection_meta = {}
        for role, rec in [("primary", sel["primary"]), ("secondary", sel.get("secondary"))]:
            if not rec:
                artifacts[f"{role}_overlay"] = None
                artifacts[f"{role}_object_crop"] = None
                artifacts[f"{role}_context_crop"] = None
                continue
            img_path = Path(rec["image_path"])
            if not img_path.exists():
                continue
            with Image.open(img_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)
                bbox = rec["bbox_clipped"]
                draw.rectangle(bbox, outline=(255, 32, 32), width=4)
                draw.text((int(bbox[0]), max(0, int(bbox[1]) - 16)), f"{rec['object_id']} {rec['category']} {rec['camera']}", fill=(255, 32, 32))
                overlay = sample_dir / f"{role}_{rec['camera']}_overlay.jpg"
                img.save(overlay, quality=92)
                obj_crop_box = crop_box(bbox, img.width, img.height, 0.10)
                ctx_crop_box = crop_box(bbox, img.width, img.height, 0.50)
                obj_crop = sample_dir / f"{role}_{rec['camera']}_object_crop.jpg"
                ctx_crop = sample_dir / f"{role}_{rec['camera']}_context_crop.jpg"
                img.crop(obj_crop_box).save(obj_crop, quality=92)
                img.crop(ctx_crop_box).save(ctx_crop, quality=92)
                artifacts[f"{role}_overlay"] = str(overlay)
                artifacts[f"{role}_object_crop"] = str(obj_crop)
                artifacts[f"{role}_context_crop"] = str(ctx_crop)
                projection_meta[role] = {**rec, "object_crop_box": list(obj_crop_box), "context_crop_box": list(ctx_crop_box), "object_crop_size": [obj_crop_box[2] - obj_crop_box[0], obj_crop_box[3] - obj_crop_box[1]], "context_crop_size": [ctx_crop_box[2] - ctx_crop_box[0], ctx_crop_box[3] - ctx_crop_box[1]]}
        packages.append({
            "evidence_id": evidence_id,
            "sample_token": sel["sample_token"],
            "object_id": sel["object_id"],
            "category": sel.get("category"),
            "primary_camera": sel["primary"]["camera"],
            "secondary_camera": sel.get("secondary", {}).get("camera") if sel.get("secondary") else None,
            "projection": projection_meta,
            "artifacts": artifacts,
            "geometry": sel["primary"].get("geometry"),
            "source": "centerpoint_3d_grounded_visual_evidence",
        })
    sizes = []
    for p in packages:
        info = image_readability(p.get("artifacts", {}).get("primary_object_crop"))
        if info.get("dimensions"):
            sizes.append(info["dimensions"][0] * info["dimensions"][1])
    audit = {"package_count": len(packages), "overlays": sum(1 for p in packages if p["artifacts"].get("primary_overlay")), "object_crops": sum(1 for p in packages if p["artifacts"].get("primary_object_crop")), "context_crops": sum(1 for p in packages if p["artifacts"].get("primary_context_crop")), "object_crop_area": dist(sizes)}
    return packages, audit


def evidence_gallery(packages: list[dict], limit: int = 20) -> str:
    lines = ["# Step 7.4B Visual Evidence Gallery", ""]
    for pkg in sorted(packages, key=lambda p: p["evidence_id"])[:limit]:
        lines += [f"## {pkg['evidence_id']}", "", f"- category: `{pkg.get('category')}`", f"- camera: `{pkg.get('primary_camera')}`", ""]
        for label, key in [("overlay", "primary_overlay"), ("object crop", "primary_object_crop"), ("context crop", "primary_context_crop")]:
            path = rel(pkg["artifacts"].get(key))
            if path:
                lines.append(f"![{label}]({path})")
                lines.append("")
    return "\n".join(lines)


def audit_grounding(packages: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for p in packages:
        art = p.get("artifacts") or {}
        proj = (p.get("projection") or {}).get("primary") or {}
        crop = image_readability(art.get("primary_object_crop"))
        ctx = image_readability(art.get("primary_context_crop"))
        area = float(proj.get("projected_area") or 0)
        raw_trunc = proj.get("truncation_ratio")
        trunc = float(raw_trunc) if raw_trunc is not None else 1.0
        if not crop.get("readable") or area <= 0:
            quality = "invalid"
        elif area >= 2500 and trunc <= 0.35 and min(crop.get("dimensions") or [0, 0]) >= 32:
            quality = "high"
        elif area >= 900 and trunc <= 0.65 and min(crop.get("dimensions") or [0, 0]) >= 16:
            quality = "medium"
        else:
            quality = "low"
        rows.append({"evidence_id": p["evidence_id"], "sample_token": p["sample_token"], "object_id": p["object_id"], "category": p.get("category"), "quality": quality, "projected_area": area, "truncation_ratio": trunc, "crop_readability": crop, "context_readability": ctx})
    return rows, {"quality_distribution": count_by(rows, "quality"), "package_count": len(rows)}


def parse_json(raw: str | None) -> tuple[dict, str | None]:
    if not raw:
        return {}, "empty_output"
    text = raw.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    try:
        val = json.loads(text)
        return (val if isinstance(val, dict) else {}), None if isinstance(val, dict) else "not_object"
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def normalize_enum(value: Any, allowed: set[str], default: str = "unknown") -> str:
    v = str(value).strip().lower()
    return v if v in allowed else default


def validate_predicate_output(predicate_type: str, object_id: str, parsed: dict) -> dict:
    reasons = []
    if parsed.get("object_id") != object_id:
        reasons.append("object_id_mismatch")
    if parsed.get("not_answering_full_question") is not True:
        reasons.append("not_answering_flag_missing")
    if predicate_type == "semantic_status":
        if parsed.get("predicate_type") != "semantic_status":
            reasons.append("predicate_type_mismatch")
        if normalize_enum(parsed.get("observed_status"), {"moving", "standing", "sitting_lying_down", "parked", "stopped", "with_rider", "without_rider", "unknown"}) == "unknown" and str(parsed.get("observed_status", "unknown")).lower() != "unknown":
            reasons.append("bad_observed_status")
        if normalize_enum(parsed.get("target_status_support"), {"yes", "no", "unknown"}) not in {"yes", "no", "unknown"}:
            reasons.append("bad_support")
    elif predicate_type == "candidate_validation":
        if parsed.get("predicate_type") != "candidate_validation":
            reasons.append("predicate_type_mismatch")
        for key in ["visible", "category_match"]:
            if normalize_enum(parsed.get(key), {"yes", "no", "unknown"}) not in {"yes", "no", "unknown"}:
                reasons.append(f"bad_{key}")
    else:
        reasons.append("unknown_predicate_type")
    if normalize_enum(parsed.get("confidence"), {"low", "medium", "high"}, "none") == "none":
        reasons.append("bad_confidence")
    if normalize_enum(parsed.get("evidence_image_role"), {"object_crop", "context_crop", "overlay", "multiple"}, "none") == "none":
        reasons.append("bad_evidence_image_role")
    direct = any(k in parsed for k in ["answer", "final_answer", "comparison_result", "count"])
    if direct:
        reasons.append("direct_answer_field_present")
    return {"valid": not reasons, "reasons": reasons, "direct_answer_violation": direct}


def md_kv(title: str, data: dict) -> str:
    return f"# {title}\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in data.items())
