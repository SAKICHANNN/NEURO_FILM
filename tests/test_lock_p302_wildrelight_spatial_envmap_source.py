from __future__ import annotations

from scripts.lock_p302_wildrelight_spatial_envmap_source import (
    _rank_scenes,
    _select_members,
)


def test_scene_ranking_is_deterministic() -> None:
    revision = "a" * 40
    scenes = ["gamma", "alpha", "beta"]
    assert _rank_scenes(revision, scenes) == _rank_scenes(
        revision, list(reversed(scenes))
    )


def test_select_members_requires_exact_six_time_roles() -> None:
    roles = {
        "training": ["train"],
        "development": ["dev"],
        "confirmation": ["confirm"],
        "reserve": ["reserve"],
    }
    items: list[dict[str, object]] = []
    for role_scenes in roles.values():
        for scene in role_scenes:
            paths = [f"small-aligned/{scene}/meta.json"]
            for time_index in range(6):
                paths.extend(
                    (
                        f"small-aligned/{scene}/envmap/time{time_index}_envmap.exr",
                        f"small-aligned/{scene}/photo/time{time_index}_hdr.exr",
                    )
                )
            for index, path in enumerate(paths):
                is_metadata = path.endswith("/meta.json")
                items.append(
                    {
                        "lfs": None if is_metadata else {"oid": f"{index + 1:064x}"},
                        "oid": f"{index + 1:040x}",
                        "path": path,
                        "size": index + 1,
                        "type": "file",
                    }
                )
    selected = _select_members(items, roles)
    assert len(selected) == 52
    assert {item["role"] for item in selected} == set(roles)
    metadata = [item for item in selected if item["path"].endswith("meta.json")]
    assert len(metadata) == 4
    assert all(item["sha256"] is None for item in metadata)
