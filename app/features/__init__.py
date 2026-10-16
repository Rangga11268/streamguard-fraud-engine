"""
Real-time streaming feature store and sliding-window analytics module.
"""

from app.features.geo_velocity import (
    haversine_distance,
    compute_travel_speed,
)
from app.features.sliding_window import (
    SlidingEntityWindow,
    WindowStats,
)
from app.features.feature_store import (
    TransactionEntityEvent,
    ExtractedFeatures,
    StreamingFeatureStore,
    FEATURE_NAMES,
)

__all__ = [
    "haversine_distance",
    "compute_travel_speed",
    "SlidingEntityWindow",
    "WindowStats",
    "TransactionEntityEvent",
    "ExtractedFeatures",
    "StreamingFeatureStore",
    "FEATURE_NAMES",
]
