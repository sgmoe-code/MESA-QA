from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from step71_common import OUTPUT_DIR, write_md


@dataclass
class CanonicalSceneObject:
    object_id: str
    sample_token: str
    category: str
    geometry: dict[str, Any]
    velocity_xy: list[float] | None = None
    speed: float | None = None
    status: str = "unknown"
    attribute: str | None = None
    confidence: float | None = None
    source: str = "centerpoint"
    feature_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_schema_doc() -> None:
    write_md(
        OUTPUT_DIR / "step71_canonical_object_schema.md",
        """# Step 7.1 Canonical Scene Object Schema

Canonical objects are the source-neutral input to the shared relation builder.

- `object_id`: stable within one sample.
- `sample_token`: NuScenes sample token.
- `category`: normalized from CenterPoint `label`; no category is fabricated.
- `geometry.center_xyz`: parsed from CenterPoint `box`.
- `geometry.size_wlh`: parsed from CenterPoint `box`.
- `geometry.yaw`: parsed from CenterPoint `box` when present.
- `geometry.raw_box`: original box values.
- `velocity_xy` / `speed`: parsed from explicit velocity or 9D box tail when available.
- `status`: explicit attribute if present, otherwise coarse `moving`/`stationary` from velocity, otherwise `unknown`.
- `attribute`: raw explicit CenterPoint attribute if present.
- `confidence`: score/detection_score if present, otherwise null.
- `feature_ref`: reference only; `feats` are not interpreted semantically.
- `metadata`: source and parsing diagnostics.

Objects are not dropped because status is unknown.
""",
    )


if __name__ == "__main__":
    write_schema_doc()
