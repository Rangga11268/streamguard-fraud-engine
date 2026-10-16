"""
Multi-scale sliding-window entity deques and spending velocity analytics.
"""

from collections import deque
from typing import Optional, Dict, Tuple
from app.config import WINDOW_5M, WINDOW_1H, WINDOW_24H, EPSILON, ENTITY_TTL_SEC
from app.features.geo_velocity import compute_travel_speed

# Multiplicative inverse of 1440.0 for fast rate calculation (1 / 1440.0)
_INV_1440: float = 1.0 / 1440.0


class WindowStats:
    """Aggregated sliding-window statistics for an entity with slotted fast attribute access."""
    __slots__ = (
        "count_5m",
        "sum_5m",
        "mean_5m",
        "count_1h",
        "sum_1h",
        "mean_1h",
        "count_24h",
        "sum_24h",
        "mean_24h",
        "distinct_mcc_count",
        "amount_to_mean24h_ratio",
        "velocity_5m_to_24h_ratio",
    )

    def __init__(
        self,
        count_5m: int,
        sum_5m: float,
        mean_5m: float,
        count_1h: int,
        sum_1h: float,
        mean_1h: float,
        count_24h: int,
        sum_24h: float,
        mean_24h: float,
        distinct_mcc_count: int,
        amount_to_mean24h_ratio: float,
        velocity_5m_to_24h_ratio: float,
    ):
        self.count_5m = count_5m
        self.sum_5m = sum_5m
        self.mean_5m = mean_5m
        self.count_1h = count_1h
        self.sum_1h = sum_1h
        self.mean_1h = mean_1h
        self.count_24h = count_24h
        self.sum_24h = sum_24h
        self.mean_24h = mean_24h
        self.distinct_mcc_count = distinct_mcc_count
        self.amount_to_mean24h_ratio = amount_to_mean24h_ratio
        self.velocity_5m_to_24h_ratio = velocity_5m_to_24h_ratio

    def __repr__(self) -> str:
        return (
            f"WindowStats(count_5m={self.count_5m}, sum_5m={self.sum_5m:.2f}, "
            f"count_1h={self.count_1h}, sum_1h={self.sum_1h:.2f}, "
            f"count_24h={self.count_24h}, sum_24h={self.sum_24h:.2f}, "
            f"distinct_mcc={self.distinct_mcc_count}, "
            f"ratio_24h={self.amount_to_mean24h_ratio:.2f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WindowStats):
            return False
        return (
            self.count_5m == other.count_5m
            and abs(self.sum_5m - other.sum_5m) < 1e-4
            and self.count_1h == other.count_1h
            and abs(self.sum_1h - other.sum_1h) < 1e-4
            and self.count_24h == other.count_24h
            and abs(self.sum_24h - other.sum_24h) < 1e-4
            and self.distinct_mcc_count == other.distinct_mcc_count
        )


class SlidingEntityWindow:
    """
    Slotted sliding-window transaction buffer for a customer or card entity.

    Maintains multi-scale chronological events over 5-minute (300s), 1-hour (3,600s),
    and 24-hour (86,400s) durations using a Python deque. Running totals and an O(1)
    MCC frequency map optimize 24-hour window aggregations, while reverse scanning
    evaluates short-term 5m and 1h metrics in microseconds.
    """

    __slots__ = (
        "entity_id",
        "_events",
        "count_24h",
        "sum_24h",
        "_mcc_counts",
        "last_timestamp",
        "last_latitude",
        "last_longitude",
    )

    def __init__(self, entity_id: str = ""):
        self.entity_id: str = entity_id
        self._events: deque = deque()
        self.count_24h: int = 0
        self.sum_24h: float = 0.0
        self._mcc_counts: Dict[str, int] = {}
        self.last_timestamp: Optional[float] = None
        self.last_latitude: Optional[float] = None
        self.last_longitude: Optional[float] = None

    @property
    def events(self) -> deque:
        """Access the underlying deque of chronological events."""
        return self._events

    @property
    def last_lat(self) -> Optional[float]:
        return self.last_latitude

    @property
    def last_lon(self) -> Optional[float]:
        return self.last_longitude

    def prune_expired(self, current_timestamp: float, window_24h: float = 86400.0) -> None:
        """Prunes historical events older than (current_timestamp - window_24h)."""
        cutoff_24h = current_timestamp - window_24h
        events = self._events
        mccs = self._mcc_counts

        while events and events[0][0] < cutoff_24h:
            exp_ev = events.popleft()
            self.count_24h -= 1
            self.sum_24h -= exp_ev[1]

            exp_mcc = exp_ev[2]
            c = mccs.get(exp_mcc, 1) - 1
            if c <= 0:
                mccs.pop(exp_mcc, None)
            else:
                mccs[exp_mcc] = c

        # Safeguard against negative counts or floating-point epsilon underflow
        if self.count_24h <= 0:
            self.count_24h = 0
            self.sum_24h = 0.0
            mccs.clear()
        elif self.sum_24h < 0.0:
            self.sum_24h = 0.0

    def get_last_location_before(
        self, timestamp: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Find the most recent event prior to `timestamp` that has coordinates.
        Returns (latitude, longitude, timestamp) or (None, None, None).
        """
        for ev in reversed(self._events):
            if ev[0] <= timestamp and len(ev) > 4 and ev[3] is not None and ev[4] is not None:
                return (ev[3], ev[4], ev[0])
        return (None, None, None)

    def get_stats(
        self,
        current_timestamp: float,
        current_amount: float,
        epsilon: float = 1e-4,
    ) -> WindowStats:
        """
        Extract window aggregates strictly prior to committing the current transaction.

        Guarantees zero temporal leakage: only events strictly prior to or equal to
        current_timestamp that are already present in historical state are evaluated.
        """
        cutoff_24h = current_timestamp - 86400.0
        cutoff_1h = current_timestamp - 3600.0
        cutoff_5m = current_timestamp - 300.0
        events = self._events
        mccs = self._mcc_counts

        has_future_events = bool(events and events[-1][0] > current_timestamp)

        if not has_future_events:
            # Monotonic chronological stream: prune expired from left
            while events and events[0][0] < cutoff_24h:
                exp_ev = events.popleft()
                self.count_24h -= 1
                self.sum_24h -= exp_ev[1]
                exp_mcc = exp_ev[2]
                c = mccs.get(exp_mcc, 1) - 1
                if c <= 0:
                    mccs.pop(exp_mcc, None)
                else:
                    mccs[exp_mcc] = c

            if self.count_24h <= 0:
                self.count_24h = 0
                self.sum_24h = 0.0
                mccs.clear()
            elif self.sum_24h < 0.0:
                self.sum_24h = 0.0

            count_5m = 0
            sum_5m = 0.0
            count_1h = 0
            sum_1h = 0.0

            for ev in reversed(events):
                t = ev[0]
                if t < cutoff_1h:
                    break
                count_1h += 1
                sum_1h += ev[1]
                if t >= cutoff_5m:
                    count_5m += 1
                    sum_5m += ev[1]

            c24 = self.count_24h
            s24 = self.sum_24h
            distinct_mcc_count = len(mccs)
        else:
            # Out-of-order event present: evaluate strictly within [cutoff_24h, current_timestamp]
            count_5m = 0
            sum_5m = 0.0
            count_1h = 0
            sum_1h = 0.0
            c24 = 0
            s24 = 0.0
            distinct_mccs = set()

            for ev in reversed(events):
                t = ev[0]
                if t > current_timestamp:
                    continue
                if t < cutoff_24h:
                    break
                a = ev[1]
                c24 += 1
                s24 += a
                distinct_mccs.add(ev[2])
                if t >= cutoff_1h:
                    count_1h += 1
                    sum_1h += a
                    if t >= cutoff_5m:
                        count_5m += 1
                        sum_5m += a

            distinct_mcc_count = len(distinct_mccs)

        # Spending velocity ratios with epsilon smoothing
        if c24 > 0:
            m24 = s24 / c24
            r_amt = current_amount / (m24 + epsilon)
            # rate_5m = sum_5m / 5.0 = sum_5m * 0.2
            # rate_24h = sum_24h / 1440.0 = sum_24h * _INV_1440
            r_vel = (sum_5m * 0.2) / ((s24 * _INV_1440) + epsilon)
            m5 = sum_5m / count_5m if count_5m > 0 else 0.0
            m1 = sum_1h / count_1h if count_1h > 0 else 0.0
        else:
            m24 = 0.0
            m5 = 0.0
            m1 = 0.0
            r_amt = 1.0 if current_amount > 0.0 else 0.0
            r_vel = 1.0 if current_amount > 0.0 else 0.0

        return WindowStats(
            count_5m=count_5m,
            sum_5m=sum_5m,
            mean_5m=m5,
            count_1h=count_1h,
            sum_1h=sum_1h,
            mean_1h=m1,
            count_24h=c24,
            sum_24h=s24,
            mean_24h=m24,
            distinct_mcc_count=distinct_mcc_count,
            amount_to_mean24h_ratio=r_amt,
            velocity_5m_to_24h_ratio=r_vel,
        )

    def commit(
        self,
        timestamp: float,
        amount: float,
        mcc: str = "0000",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> None:
        """
        Commit a transaction into the entity window state.
        Maintains chronological sorting and updates running 24h totals.
        """
        ts = float(timestamp)
        # Check active window retention horizon: discard events older than 24h
        if self.last_timestamp is not None and ts < (self.last_timestamp - 86400.0):
            return

        # Update last known location and timestamp only if event is newer than or equal to existing state
        if self.last_timestamp is None or ts >= self.last_timestamp:
            self.last_timestamp = ts
            if latitude is not None and longitude is not None:
                self.last_latitude = float(latitude)
                self.last_longitude = float(longitude)

        # Update running 24h totals
        amt = float(amount)
        mcc_str = str(mcc)
        self.count_24h += 1
        self.sum_24h += amt
        self._mcc_counts[mcc_str] = self._mcc_counts.get(mcc_str, 0) + 1

        lat_val = float(latitude) if latitude is not None else None
        lon_val = float(longitude) if longitude is not None else None
        entry = (ts, amt, mcc_str, lat_val, lon_val)

        # Append or insert in sorted order
        events = self._events
        if not events or ts >= events[-1][0]:
            events.append(entry)
        else:
            # Out-of-order event: insert at appropriate chronological index
            inserted = False
            for i in range(len(events) - 1, -1, -1):
                if events[i][0] <= ts:
                    events.insert(i + 1, entry)
                    inserted = True
                    break
            if not inserted:
                events.appendleft(entry)

    def prune_and_aggregate(
        self,
        current_time: float,
        new_amount: float,
        new_lat: float,
        new_lon: float,
        speed_threshold_kmh: float = 800.0,
    ) -> Tuple[int, int, int, float, float, float]:
        """
        Combined high-performance sliding window feature extraction and state commit.
        Strict zero temporal leakage: stats are extracted BEFORE committing incoming event.

        Returns:
            (cnt_5m, cnt_1h, cnt_24h, amt_to_mean24h_ratio, speed_kmh, impossible_jump)
        """
        # 1. Prune expired
        self.prune_expired(current_time)

        # 2. Compute prior features
        t_5m = current_time - 300.0
        t_1h = current_time - 3600.0
        cnt_5m = 0
        cnt_1h = 0

        for ev in reversed(self._events):
            t = ev[0]
            if t > current_time:
                continue
            if t < t_1h:
                break
            cnt_1h += 1
            if t >= t_5m:
                cnt_5m += 1

        cnt_24h = self.count_24h
        mean_24h = (self.sum_24h / cnt_24h) if cnt_24h > 0 else (new_amount if new_amount > 0.0 else 1.0)
        amt_to_mean24h_ratio = new_amount / (mean_24h + 1e-4) if cnt_24h > 0 else (1.0 if new_amount > 0.0 else 0.0)

        # 3. Geo velocity
        if self.last_latitude is not None and self.last_longitude is not None and self.last_timestamp is not None and self.last_timestamp > 0.0:
            if current_time >= self.last_timestamp:
                speed_kmh, imp_flag, _ = compute_travel_speed(
                    self.last_latitude, self.last_longitude, self.last_timestamp,
                    new_lat, new_lon, current_time,
                    speed_threshold_kmh=speed_threshold_kmh,
                )
            else:
                prior_lat, prior_lon, prior_ts = self.get_last_location_before(current_time)
                if prior_lat is not None and prior_ts is not None:
                    speed_kmh, imp_flag, _ = compute_travel_speed(
                        prior_lat, prior_lon, prior_ts,
                        new_lat, new_lon, current_time,
                        speed_threshold_kmh=speed_threshold_kmh,
                    )
                else:
                    speed_kmh = 0.0
                    imp_flag = 0
        else:
            speed_kmh = 0.0
            imp_flag = 0

        # 4. Commit AFTER extraction
        self.commit(
            timestamp=current_time,
            amount=new_amount,
            mcc="0000",
            latitude=new_lat,
            longitude=new_lon,
        )

        return (
            cnt_5m,
            cnt_1h,
            cnt_24h,
            amt_to_mean24h_ratio,
            speed_kmh,
            float(imp_flag),
        )

    def is_inactive(self, current_timestamp: float, ttl_seconds: float = 86400.0) -> bool:
        """Check if entity has had no transaction activity for longer than TTL."""
        if self.last_timestamp is None:
            return True
        return (current_timestamp - self.last_timestamp) > ttl_seconds

    def reset(self) -> None:
        """Reset entity window to empty state."""
        self._events.clear()
        self.count_24h = 0
        self.sum_24h = 0.0
        self._mcc_counts.clear()
        self.last_timestamp = None
        self.last_latitude = None
        self.last_longitude = None
