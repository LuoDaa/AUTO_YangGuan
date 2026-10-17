from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PlacementConfig:
    scale_min: float = 0.85
    scale_max: float = 1.15
    cervical_band_mm: float = 1.50
    emergence_pad_mm: float = 0.70
    cervical_quantile: float = 0.04


@dataclass(frozen=True)
class ProximalConfig:
    target_gap_mm: float = 0.05
    tolerance_mm: float = 0.10
    domain_quantile: float = 0.66
    contact_fraction: float = 0.16
    min_contact_vertices: int = 20
    max_growth_mm: float = 0.65
    max_relief_mm: float = 0.50
    smooth_iterations: int = 8
    smooth_alpha: float = 0.42


@dataclass(frozen=True)
class OcclusalConfig:
    target_gap_mm: float = 0.08
    tolerance_mm: float = 0.06
    top_quantile: float = 0.58
    max_relief_mm: float = 0.60
    smooth_iterations: int = 6
    smooth_alpha: float = 0.38


@dataclass(frozen=True)
class QualityConfig:
    cervical_lock_mm: float = 0.45
    max_total_displacement_mm: float = 1.20
    review_p95_displacement_mm: float = 0.70
    hard_face_flip_limit: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    placement: PlacementConfig = PlacementConfig()
    proximal: ProximalConfig = ProximalConfig()
    occlusal: OcclusalConfig = OcclusalConfig()
    quality: QualityConfig = QualityConfig()

    def as_dict(self) -> dict:
        return asdict(self)
