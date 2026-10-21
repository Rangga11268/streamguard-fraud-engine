"""
Unified real-time streaming feature store with strict zero temporal leakage.
"""

from dataclasses import dataclass
import math
import struct
from typing import Dict, List, Optional, Union, Any
import numpy as np

from app.config import (
    FEATURE_NAMES,
    IMPOSSIBLE_TRAVEL_SPEED_KMH,
    ENTITY_TTL_SEC,
    EPSILON,
    StreamGuardConfig,
    default_config,
)
from app.features.geo_velocity import compute_travel_speed
from app.features.sliding_window import SlidingEntityWindow, WindowStats

# Pre-compiled struct packer for fast 16-element float32 numpy array creation
_STRUCT_16F = struct.Struct("16f").pack


@dataclass(slots=True)
class TransactionEntityEvent:
    """Incoming transaction event payload."""
    transaction_id: str
    customer_id: str
    card_id: str
    amount: float
    timestamp: float
    merchant_category_code: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        """Validate physical boundaries and finite numeric properties."""
        if not math.isfinite(self.amount) or self.amount < 0.0:
            raise ValueError(f"Transaction amount must be a finite non-negative real number, got: {self.amount}")
        if not math.isfinite(self.timestamp) or self.timestamp <= 0.0:
            raise ValueError(f"Transaction timestamp must be a finite strictly positive real number, got: {self.timestamp}")
        if not math.isfinite(self.latitude) or not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitude must be finite and within physical bounds [-90, 90], got: {self.latitude}")
        if not math.isfinite(self.longitude) or not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitude must be finite and within physical bounds [-180, 180], got: {self.longitude}")

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionEntityEvent":
        """Instantiate from dictionary or REST API JSON payload."""
        loc = data.get("location")
        if isinstance(loc, dict):
            lat = float(loc.get("latitude", 0.0))
            lon = float(loc.get("longitude", 0.0))
        else:
            lat = float(data.get("latitude", 0.0))
            lon = float(data.get("longitude", 0.0))

        return cls(
            transaction_id=str(data.get("transaction_id", "")),
            customer_id=str(data.get("customer_id", "")),
            card_id=str(data.get("card_id", "")),
            amount=float(data.get("amount", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
            merchant_category_code=str(
                data.get("merchant_category_code", data.get("mcc", "0000"))
            ),
            latitude=lat,
            longitude=lon,
        )


@dataclass(slots=True)
class ExtractedFeatures:
    """Extracted feature vector conforming to PROJECT.md Interface Contract 1."""
    feature_names: List[str]
    feature_vector: np.ndarray  # dtype=np.float32, 1D array of 16 features
    travel_speed_kmh: float
    impossible_travel_flag: int
    velocity_5m_count: int
    velocity_1h_count: int
    velocity_24h_count: int
    amount_to_mean24h_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Summary feature dict suitable for REST API /v1/score response."""
        return {
            "velocity_5m_count": self.velocity_5m_count,
            "velocity_1h_count": self.velocity_1h_count,
            "velocity_24h_count": self.velocity_24h_count,
            "amount_to_mean24h_ratio": round(self.amount_to_mean24h_ratio, 4),
            "travel_speed_kmh": round(self.travel_speed_kmh, 2),
            "impossible_travel_flag": self.impossible_travel_flag,
        }

    def to_full_dict(self) -> Dict[str, float]:
        """Full feature dictionary mapping each of the 16 feature names to its value."""
        return {
            name: float(self.feature_vector[i])
            for i, name in enumerate(self.feature_names)
        }


class StreamingFeatureStore:
    """
    High-performance in-memory streaming feature store.

    Manages customer-level and card-level sliding windows, Haversine travel jumps,
    spending velocity ratios, and TTL eviction. Enforces strict zero temporal leakage:
    features are extracted BEFORE committing the transaction into the entity window state.
    """

    def __init__(self, config: Optional[StreamGuardConfig] = None):
        self.config: StreamGuardConfig = config or default_config
        self._customer_windows: Dict[str, SlidingEntityWindow] = {}
        self._card_windows: Dict[str, SlidingEntityWindow] = {}
        self._last_event_timestamp: float = 0.0
        self._total_processed: int = 0

    def _get_or_create_customer(self, customer_id: str) -> SlidingEntityWindow:
        window = self._customer_windows.get(customer_id)
        if window is None:
            window = SlidingEntityWindow(customer_id)
            self._customer_windows[customer_id] = window
        return window

    def _get_or_create_card(self, card_id: str) -> SlidingEntityWindow:
        window = self._card_windows.get(card_id)
        if window is None:
            window = SlidingEntityWindow(card_id)
            self._card_windows[card_id] = window
        return window

    def extract_features(
        self,
        event: Union[TransactionEntityEvent, dict],
        commit: bool = True,
    ) -> ExtractedFeatures:
        """
        Extract the 16-dimensional feature vector for an incoming transaction event.

        CRITICAL INVARIANT: Feature extraction occurs strictly BEFORE the incoming
        transaction is committed into the entity state. If commit=True, the event
        is committed after features are computed.
        """
        # Ensure event is a TransactionEntityEvent instance
        if isinstance(event, dict):
            tx_event = TransactionEntityEvent.from_dict(event)
        else:
            tx_event = event

        ts = float(tx_event.timestamp)
        amt = float(tx_event.amount)
        cust_id = tx_event.customer_id
        card_id = tx_event.card_id
        mcc = tx_event.merchant_category_code
        lat = float(tx_event.latitude)
        lon = float(tx_event.longitude)

        # Streaming input boundary contracts
        if not math.isfinite(amt) or amt < 0.0:
            raise ValueError(f"Transaction amount must be a finite non-negative real number, got: {amt}")
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError(f"Transaction timestamp must be a finite strictly positive real number, got: {ts}")
        if not math.isfinite(lat) or not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be finite and within physical bounds [-90, 90], got: {lat}")
        if not math.isfinite(lon) or not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be finite and within physical bounds [-180, 180], got: {lon}")

        # Retrieve entity windows
        cust_window = self._get_or_create_customer(cust_id)
        card_window = self._get_or_create_card(card_id)

        # 1. Compute Haversine Travel Speed BEFORE committing new coordinates
        if cust_window.last_latitude is not None and cust_window.last_timestamp is not None:
            if ts >= cust_window.last_timestamp:
                travel_speed_kmh, impossible_flag, _ = compute_travel_speed(
                    lat1=cust_window.last_latitude,
                    lon1=cust_window.last_longitude,
                    t1=cust_window.last_timestamp,
                    lat2=lat,
                    lon2=lon,
                    t2=ts,
                    speed_threshold_kmh=self.config.impossible_travel_speed_kmh,
                    min_time_delta_hours=self.config.min_time_delta_hours,
                )
            else:
                # Out-of-order past transaction (ts < cust_window.last_timestamp)
                prior_lat, prior_lon, prior_ts = cust_window.get_last_location_before(ts)
                if prior_lat is not None and prior_ts is not None:
                    travel_speed_kmh, impossible_flag, _ = compute_travel_speed(
                        lat1=prior_lat,
                        lon1=prior_lon,
                        t1=prior_ts,
                        lat2=lat,
                        lon2=lon,
                        t2=ts,
                        speed_threshold_kmh=self.config.impossible_travel_speed_kmh,
                        min_time_delta_hours=self.config.min_time_delta_hours,
                    )
                else:
                    # Cold-start: no historical event prior to ts
                    travel_speed_kmh = 0.0
                    impossible_flag = 0
        else:
            # Cold-start: first transaction seen for this customer
            travel_speed_kmh = 0.0
            impossible_flag = 0

        # 2. Extract customer-level window aggregates BEFORE committing
        cust_stats = cust_window.get_stats(
            current_timestamp=ts,
            current_amount=amt,
            epsilon=self.config.epsilon,
        )

        # 3. Extract card-level window aggregates BEFORE committing
        card_stats = card_window.get_stats(
            current_timestamp=ts,
            current_amount=amt,
            epsilon=self.config.epsilon,
        )

        # 4. Construct contiguous float32 feature array (16 features) via struct pack
        buf = _STRUCT_16F(
            amt,                                        # 0: amount
            float(cust_stats.count_5m),                 # 1: velocity_5m_count
            float(cust_stats.count_1h),                 # 2: velocity_1h_count
            float(cust_stats.count_24h),                # 3: velocity_24h_count
            cust_stats.sum_5m,                          # 4: spending_5m_sum
            cust_stats.sum_1h,                          # 5: spending_1h_sum
            cust_stats.sum_24h,                         # 6: spending_24h_sum
            cust_stats.amount_to_mean24h_ratio,         # 7: amount_to_mean24h_ratio
            cust_stats.velocity_5m_to_24h_ratio,        # 8: velocity_5m_to_24h_ratio
            float(card_stats.count_5m),                 # 9: card_velocity_5m_count
            float(card_stats.count_1h),                 # 10: card_velocity_1h_count
            float(card_stats.count_24h),                # 11: card_velocity_24h_count
            card_stats.sum_24h,                         # 12: card_spending_24h_sum
            float(card_stats.distinct_mcc_count),       # 13: card_distinct_mcc_count
            travel_speed_kmh,                           # 14: travel_speed_kmh
            float(impossible_flag),                     # 15: impossible_travel_flag
        )
        feature_vector = np.frombuffer(buf, dtype=np.float32).copy()

        extracted = ExtractedFeatures(
            feature_names=list(FEATURE_NAMES),
            feature_vector=feature_vector,
            travel_speed_kmh=travel_speed_kmh,
            impossible_travel_flag=impossible_flag,
            velocity_5m_count=cust_stats.count_5m,
            velocity_1h_count=cust_stats.count_1h,
            velocity_24h_count=cust_stats.count_24h,
            amount_to_mean24h_ratio=cust_stats.amount_to_mean24h_ratio,
        )

        # 5. Commit state only AFTER extraction has completed
        if commit:
            cust_window.commit(
                timestamp=ts,
                amount=amt,
                mcc=mcc,
                latitude=lat,
                longitude=lon,
            )
            card_window.commit(
                timestamp=ts,
                amount=amt,
                mcc=mcc,
                latitude=lat,
                longitude=lon,
            )
            if ts > self._last_event_timestamp:
                self._last_event_timestamp = ts
            self._total_processed += 1

        return extracted

    def commit(self, event: Union[TransactionEntityEvent, dict]) -> None:
        """
        Commit a transaction into customer and card window states.
        Maintains out-of-order monotonic timestamp guards.
        """
        if isinstance(event, dict):
            tx_event = TransactionEntityEvent.from_dict(event)
        else:
            tx_event = event

        ts = float(tx_event.timestamp)
        amt = float(tx_event.amount)
        mcc = str(tx_event.merchant_category_code)
        lat = float(tx_event.latitude)
        lon = float(tx_event.longitude)

        cust_window = self._get_or_create_customer(tx_event.customer_id)
        card_window = self._get_or_create_card(tx_event.card_id)

        cust_window.commit(
            timestamp=ts,
            amount=amt,
            mcc=mcc,
            latitude=lat,
            longitude=lon,
        )
        card_window.commit(
            timestamp=ts,
            amount=amt,
            mcc=mcc,
            latitude=lat,
            longitude=lon,
        )

        if ts > self._last_event_timestamp:
            self._last_event_timestamp = ts
        self._total_processed += 1

    def evict_inactive(
        self,
        current_timestamp: float,
        ttl_seconds: Optional[float] = None,
    ) -> int:
        """
        Evict inactive customer and card entity windows older than TTL.

        Returns total number of entity windows evicted.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.config.entity_ttl_sec
        evicted_count = 0

        # Evict customer windows
        cust_to_evict = [
            cid for cid, w in self._customer_windows.items()
            if w.is_inactive(current_timestamp, ttl)
        ]
        for cid in cust_to_evict:
            del self._customer_windows[cid]
            evicted_count += 1

        # Evict card windows
        card_to_evict = [
            card_id for card_id, w in self._card_windows.items()
            if w.is_inactive(current_timestamp, ttl)
        ]
        for card_id in card_to_evict:
            del self._card_windows[card_id]
            evicted_count += 1

        return evicted_count

    def get_customer_window(self, customer_id: str) -> Optional[SlidingEntityWindow]:
        """Retrieve customer window if it exists."""
        return self._customer_windows.get(customer_id)

    def get_card_window(self, card_id: str) -> Optional[SlidingEntityWindow]:
        """Retrieve card window if it exists."""
        return self._card_windows.get(card_id)

    @property
    def active_entity_count(self) -> int:
        """Total active customer and card entity windows."""
        return len(self._customer_windows) + len(self._card_windows)

    @property
    def total_transactions_processed(self) -> int:
        """Total transactions committed to the store."""
        return self._total_processed

    def reset(self) -> None:
        """Clear all entity states and reset the feature store."""
        self._customer_windows.clear()
        self._card_windows.clear()
        self._last_event_timestamp = 0.0
        self._total_processed = 0
