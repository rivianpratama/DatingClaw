"""Versioned feature recipes. A recipe hash binds preprocessing, views, and model IDs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bumbleclaw_v6.features.preprocess import (
    LEGACY_FULL,
    PHOTO_FLIP,
    PHOTO_TIGHT,
    PHOTO_TIGHT_FLIP,
    PHOTO_V1,
    PHOTO_WIDE,
    TEXT_LIGHT,
    PhotoView,
)
from bumbleclaw_v6.provenance import stable_hash


@dataclass(frozen=True)
class FeatureRecipe:
    name: str
    blocks: tuple[str, ...]
    views: tuple[PhotoView, ...]
    siglip_model: str = "google/siglip-so400m-patch14-384"
    dinov3_path: str = r"C:\Users\Rivian\dinov3-vitb16-pretrain-lvd1689m"
    version: str = "1"

    @property
    def hash(self) -> str:
        return stable_hash({
            "name": self.name,
            "blocks": self.blocks,
            "views": [v.__dict__ for v in self.views],
            "siglip_model": self.siglip_model,
            "dinov3_path": self.dinov3_path,
            "version": self.version,
        })

    def view(self, name: str) -> PhotoView:
        for view in self.views:
            if view.name == name:
                return view
        raise KeyError(f"recipe {self.name} has no view {name}")


PHOTO_V1_RECIPE = FeatureRecipe(
    "photo_v1",
    ("arc", "sp", "d3", "fv"),
    (PHOTO_V1,),
)
PHOTO_TTA_RECIPE = FeatureRecipe(
    "photo_tta_v1",
    ("arc", "sp", "sp_flip", "d3", "fv"),
    (PHOTO_V1, PHOTO_FLIP),
)
LEGACY_FULL_RECIPE = FeatureRecipe(
    "legacy_full_v1",
    ("arc", "sf", "sp", "d3", "fv"),
    (LEGACY_FULL, PHOTO_V1),
)
TEXT_LIGHT_RECIPE = FeatureRecipe(
    "text_light_v1",
    ("arc", "stop", "sp", "d3", "fv"),
    (TEXT_LIGHT, PHOTO_V1),
)
PHOTO_TIGHT_RECIPE = FeatureRecipe(
    "photo_tight_v1",
    ("arc", "sp_tight", "d3", "fv"),
    (PHOTO_V1, PHOTO_TIGHT),
)
CROP_RECIPE = FeatureRecipe(
    "crop_v1",
    ("arc", "sp", "sp_tight", "sp_wide", "d3", "fv"),
    (PHOTO_V1, PHOTO_TIGHT, PHOTO_WIDE),
)
PHOTO_TIGHT_WIDE_RECIPE = FeatureRecipe(
    "photo_tight_wide_v1",
    ("arc", "sp_tight", "d3", "fv", "sp_wide"),
    (PHOTO_V1, PHOTO_TIGHT, PHOTO_WIDE),
)
PHOTO_TTA_TIGHT_RECIPE = FeatureRecipe(
    "photo_tta_tight_v1",
    ("arc", "sp_tight_tta", "d3", "fv"),
    (PHOTO_TIGHT, PHOTO_TIGHT_FLIP),
)

RECIPES = {
    "photo_v1": PHOTO_V1_RECIPE,
    "photo_tta_v1": PHOTO_TTA_RECIPE,
    "legacy_full_v1": LEGACY_FULL_RECIPE,
    "text_light_v1": TEXT_LIGHT_RECIPE,
    "crop_v1": CROP_RECIPE,
    "photo_tight_v1": PHOTO_TIGHT_RECIPE,
    "photo_tta_tight_v1": PHOTO_TTA_TIGHT_RECIPE,
    "photo_tight_wide_v1": PHOTO_TIGHT_WIDE_RECIPE,
}


def get_recipe(name: str) -> FeatureRecipe:
    try:
        return RECIPES[name]
    except KeyError as exc:
        raise KeyError(f"unknown recipe {name}; choose from {sorted(RECIPES)}") from exc
