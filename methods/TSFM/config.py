from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TSFMConfig:
    """Configuration for TSFM feature extraction.

    engine: "placeholder" | "timesfm"
        placeholder uses simple baselines for forecasts (no external deps).
        timesfm attempts to import the TimesFM library (optional).
    """

    engine: str = "placeholder"
    max_context: int = 1024
    max_horizon: int = 512
    use_quantiles: bool = False


