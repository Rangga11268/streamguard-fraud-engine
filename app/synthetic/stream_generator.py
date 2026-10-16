"""StreamGuard Synthetic Non-Stationary Transaction Stream Generator.

Models three non-stationary streaming regimes:
  1. Regime 1: Stationary baseline with extreme class imbalance (<0.1% fraud rate).
  2. Regime 2: Holiday shopping covariate surge (4x volume, elevated tickets, invariant P(Y|X)).
  3. Regime 3: Concept drift & coordinated fraud attacks (micro-testing, rapid account drains,
               impossible geographic travel jumps, fraud burst ~3.5%-5.0%).

Also provides a dual-memory asynchronous delayed feedback queue with strict temporal gating
(zero temporal leakage) separating 1-4 hour fast investigator feedback from 14-30 day chargebacks.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple


class StreamRegime(str, Enum):
    """Streaming data generation regimes."""
    REGIME_1_BASELINE = "baseline"
    REGIME_2_COVARIATE_SURGE = "covariate_surge"
    REGIME_3_CONCEPT_ATTACK = "concept_attack"


@dataclass
class TransactionEntityEvent:
    """Core transaction entity event conforming to StreamGuard interface contracts."""
    transaction_id: str
    customer_id: str
    card_id: str
    amount: float
    timestamp: float
    merchant_category_code: str
    latitude: float
    longitude: float
    merchant_id: str = ""
    channel: str = "e-commerce"
    device_id: str = ""
    ip_address: str = ""
    currency: str = "USD"
    true_label: int = 0
    regime: str = StreamRegime.REGIME_1_BASELINE.value

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigatorFeedback:
    """Fast human investigator review feedback arriving in 1-4 hours."""
    transaction_id: str
    investigator_id: str
    verdict: str  # "FRAUD" | "LEGITIMATE"
    confidence: float
    reviewed_at: float
    notes: str = ""

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChargebackFeedback:
    """Definitive bank chargeback / dispute arriving in 14-30 days."""
    transaction_id: str
    chargeback_id: str
    reason_code: str
    chargeback_amount: float
    received_at: float
    is_fraud: bool

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CustomerProfile:
    """Persistent customer profile for realistic temporal and spatial streaming."""
    customer_id: str
    card_id: str
    home_lat: float
    home_lon: float
    base_mu: float = 3.5
    base_sigma: float = 0.8
    last_tx_timestamp: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0
    is_compromised: bool = False


class DelayedFeedbackQueue:
    """Asynchronous dual-memory delayed feedback queue with strict temporal gating.

    Ensures zero temporal leakage: feedback events are stored in a priority queue
    indexed by scheduled delivery timestamp and only yielded when current_sim_time
    reaches or exceeds the delivery time.
    """

    def __init__(self) -> None:
        # Min-heap of tuples: (delivery_timestamp, counter, event_type, feedback_object)
        self._heap: List[Tuple[float, int, str, Any]] = []
        self._counter: int = 0
        self._all_enqueued: int = 0
        self._delivered_investigator: int = 0
        self._delivered_chargebacks: int = 0

    def schedule_investigator_feedback(
        self,
        transaction_id: str,
        tx_timestamp: float,
        is_fraud: bool,
        delay_seconds: Optional[float] = None,
        investigator_id: Optional[str] = None,
        accuracy: float = 0.95,
    ) -> float:
        """Schedule fast investigator review (1-4 hours delay)."""
        if delay_seconds is None:
            # 1 to 4 hours in seconds: U(3600, 14400)
            delay_seconds = random.uniform(3600.0, 14400.0)
        
        delivery_time = tx_timestamp + delay_seconds
        inv_id = investigator_id or f"inv_{random.randint(101, 199)}"

        # Simulate review verdict with 95% catch accuracy
        if is_fraud:
            verdict = "FRAUD" if random.random() < accuracy else "LEGITIMATE"
            conf = random.uniform(0.85, 0.99) if verdict == "FRAUD" else random.uniform(0.50, 0.70)
        else:
            verdict = "LEGITIMATE" if random.random() < accuracy else "FRAUD"
            conf = random.uniform(0.88, 0.99) if verdict == "LEGITIMATE" else random.uniform(0.50, 0.65)

        feedback = InvestigatorFeedback(
            transaction_id=transaction_id,
            investigator_id=inv_id,
            verdict=verdict,
            confidence=round(conf, 4),
            reviewed_at=delivery_time,
            notes=f"Simulated review at t+{delay_seconds/3600:.1f}h",
        )

        self._counter += 1
        self._all_enqueued += 1
        heapq.heappush(self._heap, (delivery_time, self._counter, "INVESTIGATOR", feedback))
        return delivery_time

    def schedule_chargeback_feedback(
        self,
        transaction_id: str,
        tx_timestamp: float,
        amount: float,
        is_fraud: bool,
        delay_seconds: Optional[float] = None,
        reason_code: Optional[str] = None,
    ) -> float:
        """Schedule slow chargeback feedback (14-30 days delay)."""
        if delay_seconds is None:
            # 14 to 30 days in seconds: U(14*86400, 30*86400)
            delay_seconds = random.uniform(14.0 * 86400.0, 30.0 * 86400.0)

        delivery_time = tx_timestamp + delay_seconds
        rc = reason_code or ("10.4_FRAUD_CARD_ABSENT" if is_fraud else "4837_DISPUTE_RECOGNIZED")

        feedback = ChargebackFeedback(
            transaction_id=transaction_id,
            chargeback_id=f"cb_{random.randint(100000, 999999)}",
            reason_code=rc,
            chargeback_amount=amount,
            received_at=delivery_time,
            is_fraud=is_fraud,
        )

        self._counter += 1
        self._all_enqueued += 1
        heapq.heappush(self._heap, (delivery_time, self._counter, "CHARGEBACK", feedback))
        return delivery_time

    def poll(
        self,
        current_sim_time: float,
    ) -> Tuple[List[InvestigatorFeedback], List[ChargebackFeedback]]:
        """Poll all feedback events whose scheduled delivery time has arrived.

        Strictly enforces no temporal leakage: future feedback remains in heap.
        """
        investigators: List[InvestigatorFeedback] = []
        chargebacks: List[ChargebackFeedback] = []

        while self._heap and self._heap[0][0] <= current_sim_time:
            delivery_time, _, event_type, feedback = heapq.heappop(self._heap)
            if event_type == "INVESTIGATOR":
                investigators.append(feedback)
                self._delivered_investigator += 1
            elif event_type == "CHARGEBACK":
                chargebacks.append(feedback)
                self._delivered_chargebacks += 1

        return investigators, chargebacks

    def pending_count(self) -> int:
        """Return number of pending feedback events awaiting maturity."""
        return len(self._heap)

    def stats(self) -> Dict[str, int]:
        return {
            "enqueued": self._all_enqueued,
            "pending": len(self._heap),
            "delivered_investigator": self._delivered_investigator,
            "delivered_chargebacks": self._delivered_chargebacks,
        }

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0
        self._all_enqueued = 0
        self._delivered_investigator = 0
        self._delivered_chargebacks = 0


class SyntheticStreamGenerator:
    """Generates multi-regime financial transaction streams with non-stationary drift.

    Implements:
      - Regime 1: Stationary baseline (<0.1% fraud, log-normal tickets, local geo).
      - Regime 2: Black Friday covariate shift (4x volume, elevated spending, stable fraud rate).
      - Regime 3: Concept drift attack (micro-testing, rapid draining, impossible travel jumps).
    """

    STANDARD_MCCS = ["5411", "5812", "5311", "5541", "5912", "5651"]
    LUXURY_MCCS = ["5732", "5944", "5999", "4511", "7011"]
    ATTACK_MICRO_MCCS = ["7999", "4816", "5815", "5968"]
    ATTACK_DRAIN_MCCS = ["6051", "6011", "5732"]

    # Major world reference locations for impossible geo-jumps
    GEO_HUBS = [
        ("New York", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Tokyo", 35.6762, 139.6503),
        ("Singapore", 1.3521, 103.8198),
        ("Sydney", -33.8688, 151.2093),
        ("San Francisco", 37.7749, -122.4194),
        ("Frankfurt", 50.1109, 8.6821),
    ]

    def __init__(
        self,
        num_customers: int = 1000,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.num_customers = num_customers
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self._tx_seq: int = 0
        self.customers: Dict[str, CustomerProfile] = {}
        self.feedback_queue = DelayedFeedbackQueue()
        self._initialize_customers()

    def _initialize_customers(self) -> None:
        """Create pool of realistic customer profiles across major metropolitan regions."""
        base_hubs = [
            (40.7128, -74.0060),  # NYC
            (34.0522, -118.2437),  # LA
            (41.8781, -87.6298),   # Chicago
            (29.7604, -95.3698),   # Houston
            (51.5074, -0.1278),    # London
        ]

        for i in range(self.num_customers):
            cust_id = f"cust_{i:04d}"
            card_id = f"card_{i:04d}_{random.randint(1000, 9999)}"
            hub_lat, hub_lon = random.choice(base_hubs)
            # Home location with small Gaussian spread around metropolitan hub
            home_lat = hub_lat + random.gauss(0.0, 0.15)
            home_lon = hub_lon + random.gauss(0.0, 0.15)

            profile = CustomerProfile(
                customer_id=cust_id,
                card_id=card_id,
                home_lat=home_lat,
                home_lon=home_lon,
                base_mu=random.uniform(3.2, 3.8),
                base_sigma=random.uniform(0.6, 0.9),
                last_tx_timestamp=0.0,
                last_lat=home_lat,
                last_lon=home_lon,
            )
            self.customers[cust_id] = profile

    def _generate_event_id(self) -> str:
        self._tx_seq += 1
        return f"tx_{self._tx_seq:08d}"

    def generate_regime_1_baseline(
        self,
        n_samples: int,
        start_time: float = 1729080000.0,
        fraud_rate: float = 0.0008,
    ) -> Iterator[TransactionEntityEvent]:
        """Regime 1: Stationary Baseline.

        - Extreme class imbalance (<0.1% fraud rate, default 0.08%).
        - Log-normal transaction amounts ($20-$150 typical).
        - Poisson inter-arrival process (lambda = 0.2 tx/sec, mean dt = 5.0s).
        - Local geographic movement around home location (<50km).
        """
        current_time = start_time
        lam = 0.2  # tx per second

        for _ in range(n_samples):
            # Inter-arrival time exponential distribution
            dt = random.expovariate(lam)
            current_time += dt

            cust = random.choice(list(self.customers.values()))
            is_fraud = random.random() < fraud_rate

            if is_fraud:
                # Modest baseline fraud (compromised card or CNP fraud)
                amount = float(round(math.exp(random.gauss(cust.base_mu + 1.2, 0.9)), 2))
                mcc = random.choice(["5732", "5944", "7999"])
                # Slight geo jump for baseline fraud
                lat = cust.home_lat + random.gauss(0.0, 0.8)
                lon = cust.home_lon + random.gauss(0.0, 0.8)
            else:
                # Typical legitimate purchase
                amount = float(round(math.exp(random.gauss(cust.base_mu, cust.base_sigma)), 2))
                amount = max(0.50, amount)
                mcc = random.choice(self.STANDARD_MCCS)
                # Local Brownian jitter (< 5km)
                lat = cust.home_lat + random.gauss(0.0, 0.02)
                lon = cust.home_lon + random.gauss(0.0, 0.02)

            cust.last_tx_timestamp = current_time
            cust.last_lat = lat
            cust.last_lon = lon

            event = TransactionEntityEvent(
                transaction_id=self._generate_event_id(),
                customer_id=cust.customer_id,
                card_id=cust.card_id,
                amount=amount,
                timestamp=round(current_time, 4),
                merchant_category_code=mcc,
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                merchant_id=f"merch_{mcc}_{random.randint(100, 999)}",
                channel="pos" if random.random() < 0.6 else "e-commerce",
                device_id=f"dev_{cust.customer_id[-4:]}",
                ip_address=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                true_label=1 if is_fraud else 0,
                regime=StreamRegime.REGIME_1_BASELINE.value,
            )
            yield event

    def generate_regime_2_covariate_surge(
        self,
        n_samples: int,
        start_time: float,
        fraud_rate: float = 0.0008,
    ) -> Iterator[TransactionEntityEvent]:
        """Regime 2: Holiday Shopping Covariate Surge (Black Friday).

        - Input distribution P(X) shifts: volume surges 4x (lambda = 0.8 tx/sec).
        - Higher ticket sizes (mean $100 - $900).
        - Concentration in high-end retail / tech / luxury MCCs.
        - P(Y|X) and fraud rate remain constant at baseline (~0.08%).
        """
        current_time = start_time
        lam = 0.8  # 4x volume surge

        for _ in range(n_samples):
            dt = random.expovariate(lam)
            current_time += dt

            cust = random.choice(list(self.customers.values()))
            is_fraud = random.random() < fraud_rate

            if is_fraud:
                amount = float(round(math.exp(random.gauss(5.8, 0.9)), 2))
                mcc = random.choice(self.LUXURY_MCCS)
                lat = cust.home_lat + random.gauss(0.0, 1.0)
                lon = cust.home_lon + random.gauss(0.0, 1.0)
            else:
                # Elevated seasonal shopping amounts
                amount = float(round(math.exp(random.gauss(5.2, 1.1)), 2))
                amount = max(10.0, amount)
                mcc = random.choice(self.LUXURY_MCCS + self.STANDARD_MCCS)
                # Shopping centers / broader metro region
                lat = cust.home_lat + random.gauss(0.0, 0.08)
                lon = cust.home_lon + random.gauss(0.0, 0.08)

            cust.last_tx_timestamp = current_time
            cust.last_lat = lat
            cust.last_lon = lon

            event = TransactionEntityEvent(
                transaction_id=self._generate_event_id(),
                customer_id=cust.customer_id,
                card_id=cust.card_id,
                amount=amount,
                timestamp=round(current_time, 4),
                merchant_category_code=mcc,
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                merchant_id=f"merch_bf_{mcc}_{random.randint(100, 999)}",
                channel="e-commerce" if random.random() < 0.8 else "pos",
                device_id=f"dev_{cust.customer_id[-4:]}",
                ip_address=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                true_label=1 if is_fraud else 0,
                regime=StreamRegime.REGIME_2_COVARIATE_SURGE.value,
            )
            yield event

    def generate_regime_3_concept_drift(
        self,
        n_samples: int,
        start_time: float,
        target_attack_rate: float = 0.045,
    ) -> Iterator[TransactionEntityEvent]:
        """Regime 3: Concept Drift & Coordinated Fraud Wave.

        Features:
          - Fraud rate surges to 3.5% - 5.0%.
          - Card testing: 3-5 rapid micro-transactions ($0.50-$1.99).
          - Immediate account draining: large transaction ($850-$2500) within 2 minutes.
          - Impossible geographic jumps (>1500 km/h, intercontinental shifts).
        """
        current_time = start_time
        lam = 0.35
        remaining = n_samples

        # Pre-select compromised victim accounts for coordinated attacks
        compromised_pool = random.sample(
            list(self.customers.values()),
            k=max(5, int(n_samples * target_attack_rate / 4)),
        )
        for c in compromised_pool:
            c.is_compromised = True

        attack_queue: List[TransactionEntityEvent] = []

        while remaining > 0 or attack_queue:
            if attack_queue:
                event = attack_queue.pop(0)
                yield event
                continue

            dt = random.expovariate(lam)
            current_time += dt
            remaining -= 1

            # Decide whether to launch a coordinated attack wave
            if random.random() < (target_attack_rate / 3.0) and compromised_pool:
                victim = random.choice(compromised_pool)
                wave = self._generate_coordinated_attack_wave(victim, current_time)
                # Advance current_time to the final attack event timestamp to guarantee
                # strictly non-decreasing timestamps for subsequent background transactions
                current_time = wave[-1].timestamp
                # First event of wave yields now, rest put in queue
                first_event = wave[0]
                attack_queue.extend(wave[1:])
                yield first_event
                continue

            # Otherwise generate background transaction
            cust = random.choice(list(self.customers.values()))
            # Baseline fraud probability even in regime 3
            is_fraud = random.random() < 0.005

            amount = float(round(math.exp(random.gauss(cust.base_mu, cust.base_sigma)), 2))
            mcc = random.choice(self.STANDARD_MCCS)
            lat = cust.home_lat + random.gauss(0.0, 0.03)
            lon = cust.home_lon + random.gauss(0.0, 0.03)

            cust.last_tx_timestamp = current_time
            cust.last_lat = lat
            cust.last_lon = lon

            event = TransactionEntityEvent(
                transaction_id=self._generate_event_id(),
                customer_id=cust.customer_id,
                card_id=cust.card_id,
                amount=amount,
                timestamp=round(current_time, 4),
                merchant_category_code=mcc,
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                merchant_id=f"merch_{mcc}_{random.randint(100, 999)}",
                channel="e-commerce",
                device_id=f"dev_{cust.customer_id[-4:]}",
                ip_address=f"172.16.{random.randint(1, 254)}.{random.randint(1, 254)}",
                true_label=1 if is_fraud else 0,
                regime=StreamRegime.REGIME_3_CONCEPT_ATTACK.value,
            )
            yield event

    def _generate_coordinated_attack_wave(
        self,
        victim: CustomerProfile,
        base_time: float,
    ) -> List[TransactionEntityEvent]:
        """Generate coordinated attack: 2-3 micro-tests, rapid drain, impossible travel jump."""
        wave: List[TransactionEntityEvent] = []
        t = base_time

        # Step 1: Micro-testing card authorization charges
        num_micro = random.randint(2, 4)
        for _ in range(num_micro):
            t += random.uniform(5.0, 25.0)  # within seconds
            micro_amount = round(random.uniform(0.50, 1.99), 2)
            micro_event = TransactionEntityEvent(
                transaction_id=self._generate_event_id(),
                customer_id=victim.customer_id,
                card_id=victim.card_id,
                amount=micro_amount,
                timestamp=round(t, 4),
                merchant_category_code=random.choice(self.ATTACK_MICRO_MCCS),
                latitude=round(victim.home_lat + random.gauss(0, 0.01), 5),
                longitude=round(victim.home_lon + random.gauss(0, 0.01), 5),
                merchant_id="merch_micro_probe",
                channel="e-commerce",
                device_id="bot_selenium_headless",
                ip_address=f"185.220.{random.randint(100, 250)}.{random.randint(1, 250)}",
                true_label=1,
                regime=StreamRegime.REGIME_3_CONCEPT_ATTACK.value,
            )
            wave.append(micro_event)

        # Step 2: Rapid Account Drain within 30-90 seconds
        t += random.uniform(20.0, 75.0)
        drain_amount = round(random.uniform(850.0, 2490.0), 2)
        drain_event = TransactionEntityEvent(
            transaction_id=self._generate_event_id(),
            customer_id=victim.customer_id,
            card_id=victim.card_id,
            amount=drain_amount,
            timestamp=round(t, 4),
            merchant_category_code=random.choice(self.ATTACK_DRAIN_MCCS),
            latitude=round(victim.home_lat + random.gauss(0, 0.02), 5),
            longitude=round(victim.home_lon + random.gauss(0, 0.02), 5),
            merchant_id="merch_crypto_exchange_drain",
            channel="e-commerce",
            device_id="dev_proxy_exit_node",
            ip_address="45.154.255.89",
            true_label=1,
            regime=StreamRegime.REGIME_3_CONCEPT_ATTACK.value,
        )
        wave.append(drain_event)

        # Step 3: Impossible Geographic Jump (e.g. London or Tokyo in 15 minutes)
        t += random.uniform(300.0, 900.0)  # 5-15 minutes later
        distant_hub = random.choice([hub for hub in self.GEO_HUBS if hub[0] != "New York"])
        jump_amount = round(random.uniform(400.0, 1800.0), 2)
        jump_event = TransactionEntityEvent(
            transaction_id=self._generate_event_id(),
            customer_id=victim.customer_id,
            card_id=victim.card_id,
            amount=jump_amount,
            timestamp=round(t, 4),
            merchant_category_code=random.choice(["5732", "5944", "6011"]),
            latitude=round(distant_hub[1] + random.gauss(0, 0.01), 5),
            longitude=round(distant_hub[2] + random.gauss(0, 0.01), 5),
            merchant_id=f"merch_intl_{distant_hub[0].lower().replace(' ', '_')}",
            channel="pos",
            device_id="pos_terminal_foreign",
            ip_address="212.58.244.70",
            true_label=1,
            regime=StreamRegime.REGIME_3_CONCEPT_ATTACK.value,
        )
        wave.append(jump_event)

        # Update customer profile tracking
        victim.last_tx_timestamp = t
        victim.last_lat = jump_event.latitude
        victim.last_lon = jump_event.longitude
        return wave

    def generate_multi_phase_stream(
        self,
        n_baseline: int = 2000,
        n_covariate: int = 1500,
        n_attack: int = 1500,
        start_time: float = 1729080000.0,
        auto_enqueue_feedback: bool = True,
    ) -> Iterator[TransactionEntityEvent]:
        """Generate a complete 3-phase synthetic stream end-to-end.

        Phase 1: Baseline (t=0..2000)
        Phase 2: Covariate Shift Surge (t=2001..3500)
        Phase 3: Concept Drift Attack (t=3501..5000)
        """
        current_time = start_time

        # Phase 1: Baseline
        for event in self.generate_regime_1_baseline(n_baseline, start_time=current_time):
            current_time = event.timestamp
            if auto_enqueue_feedback:
                self._enqueue_event_feedback(event)
            yield event

        # Phase 2: Covariate Shift
        for event in self.generate_regime_2_covariate_surge(n_covariate, start_time=current_time):
            current_time = event.timestamp
            if auto_enqueue_feedback:
                self._enqueue_event_feedback(event)
            yield event

        # Phase 3: Concept Drift
        for event in self.generate_regime_3_concept_drift(n_attack, start_time=current_time):
            current_time = event.timestamp
            if auto_enqueue_feedback:
                self._enqueue_event_feedback(event)
            yield event

    def _enqueue_event_feedback(self, event: TransactionEntityEvent) -> None:
        """Enqueue simulated asynchronous feedback into dual-memory queue."""
        is_fraud = bool(event.true_label == 1)

        # Investigator review: all fraud transactions + 2% sample of legitimate
        if is_fraud or random.random() < 0.02:
            self.feedback_queue.schedule_investigator_feedback(
                transaction_id=event.transaction_id,
                tx_timestamp=event.timestamp,
                is_fraud=is_fraud,
            )

        # Chargebacks: all fraud transactions receive chargeback dispute in 14-30 days
        if is_fraud:
            self.feedback_queue.schedule_chargeback_feedback(
                transaction_id=event.transaction_id,
                tx_timestamp=event.timestamp,
                amount=event.amount,
                is_fraud=True,
            )
        elif random.random() < 0.0001:  # Rare legitimate billing dispute
            self.feedback_queue.schedule_chargeback_feedback(
                transaction_id=event.transaction_id,
                tx_timestamp=event.timestamp,
                amount=event.amount,
                is_fraud=False,
            )
