"""Online detector: the real-time counterpart to the offline training
pipeline. Consumes the same `netstream.events.v1` OTLP logs `otel-gateway`
reads to fill ClickHouse — on its own consumer group, so neither path
affects the other (`docs/architecture/system.md`, "Целевая архитектура") —
feeds them into `features.WindowAccumulator`, the exact accumulator the
batch/streaming parity tests already check against `features.build_features`,
and scores each completed window with a baseline fitted offline.

One `WindowAccumulator` per `host_id`, not a single shared one: the
accumulator keys windows by `src_ip` alone, and two different sensors can
legitimately see overlapping private IP ranges (different sites, different
VLANs) — sharing one accumulator across sensors would silently merge their
windows.

Deliberately not ONNX yet (M14): loads whatever `joblib.dump` produced for
any of the existing baselines (threshold, Isolation Forest, Random Forest,
gradient boosting) — they already share a `.predict(frame) -> pl.Series` /
`.decision_score(frame) -> pl.Series` interface, so the detector does not
care which one it got. ONNX export and a model registry pull are a real
follow-up, not a blocker for a first working loop.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import confluent_kafka
import joblib
import polars as pl

from netstream_ml.features import WindowAccumulator, WindowSpec
from netstream_ml.otlp_decode import iter_records

logger = logging.getLogger(__name__)


class Baseline(Protocol):
    """What every baseline in `baselines.py`/`anomaly.py`/`supervised.py`/
    `gradient_boosting.py` already implements — the detector depends on
    this, not on any one of their concrete types."""

    def predict(self, frame: pl.DataFrame) -> pl.Series: ...

    def decision_score(self, frame: pl.DataFrame) -> pl.Series: ...


def load_model(path: Path) -> Baseline:
    model: Baseline = joblib.load(path)
    return model


def save_model(model: Baseline, path: Path) -> None:
    joblib.dump(model, path)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    host_id: str
    src_ip: str
    window_start: int
    window_seconds: int
    score: float
    is_attack: bool
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "host_id": self.host_id,
            "src_ip": self.src_ip,
            "window_start": self.window_start,
            "window_seconds": self.window_seconds,
            "score": self.score,
            "is_attack": self.is_attack,
            "latency_ms": self.latency_ms,
        }


class OnlineDetector:
    def __init__(self, model: Baseline, window: WindowSpec) -> None:
        self._model = model
        self._window = window
        self._accumulators: dict[str, WindowAccumulator] = {}

    def _accumulator_for(self, host_id: str) -> WindowAccumulator:
        accumulator = self._accumulators.get(host_id)
        if accumulator is None:
            accumulator = WindowAccumulator(self._window)
            self._accumulators[host_id] = accumulator
        return accumulator

    def _score_windows(
        self, host_id: str, windows: list[dict[str, object]], now: dt.datetime
    ) -> list[DetectionResult]:
        if not windows:
            return []
        frame = pl.DataFrame(windows)
        predictions = self._model.predict(frame)
        scores = self._model.decision_score(frame)

        results = []
        for row, is_attack, score in zip(windows, predictions, scores, strict=True):
            window_start = cast(int, row["window_start"])
            window_end = dt.datetime.fromtimestamp(window_start + self._window.seconds, tz=dt.UTC)
            latency_ms = (now - window_end).total_seconds() * 1000
            results.append(
                DetectionResult(
                    host_id=host_id,
                    src_ip=str(row["src_ip"]),
                    window_start=window_start,
                    window_seconds=self._window.seconds,
                    score=float(score),
                    is_attack=bool(is_attack),
                    latency_ms=latency_ms,
                )
            )
        return results

    def process_message(
        self, payload: bytes, now: dt.datetime | None = None
    ) -> list[DetectionResult]:
        """Decodes one Kafka message's OTLP batch, feeds every record into
        its sensor's accumulator, and scores whatever windows that closed
        as a result. `now` is injectable for tests; defaults to wall clock
        so `latency_ms` means something in production."""
        now = now or dt.datetime.now(tz=dt.UTC)
        by_host: dict[str, list[dict[str, object]]] = {}

        for kind, host_id, record in iter_records(payload):
            accumulator = self._accumulator_for(host_id)
            if kind == "flow":
                accumulator.add_flow_record(record)
            else:
                accumulator.add_dns_record(record)
            completed = accumulator.pop_completed()
            if completed:
                by_host.setdefault(host_id, []).extend(completed)

        results = []
        for host_id, windows in by_host.items():
            results.extend(self._score_windows(host_id, windows, now))
        return results

    def flush(self, now: dt.datetime | None = None) -> list[DetectionResult]:
        """Force-closes every still-open window across every sensor and
        scores them. A source that goes quiet never sends the later record
        that would otherwise close its last window — call this
        periodically (an idle timeout) or on shutdown, not only in
        response to new messages."""
        now = now or dt.datetime.now(tz=dt.UTC)
        results = []
        for host_id, accumulator in self._accumulators.items():
            accumulator.flush()
            completed = accumulator.pop_completed()
            if completed:
                results.extend(self._score_windows(host_id, completed, now))
        return results


def _commit_if_anything_to_commit(consumer: confluent_kafka.Consumer) -> None:
    """A bare `commit()` (no `message=`) commits whatever this consumer
    handle has stored since its last commit — which, since every message
    offset is stored via the per-message `commit(message=msg, ...)` call
    right after a successful publish, is normally nothing by the time the
    idle-flush or shutdown path gets here. confluent_kafka raises
    `_NO_OFFSET` for that case instead of treating it as a no-op; without
    this guard, an idle period with nothing new to force-close (or a
    shutdown right after the last per-message commit) crashes the whole
    detector."""
    try:
        consumer.commit(asynchronous=False)
    except confluent_kafka.KafkaException as exc:
        error = exc.args[0]
        if not (hasattr(error, "code") and error.code() == confluent_kafka.KafkaError._NO_OFFSET):
            raise


def _latency_percentiles(latencies: list[float]) -> tuple[float, float]:
    ordered = sorted(latencies)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    return p50, p95


@dataclass(frozen=True, slots=True)
class KafkaDetectorConfig:
    model: Baseline
    window_seconds: int
    bootstrap_servers: str
    events_topic: str
    detections_topic: str
    consumer_group: str
    idle_flush_seconds: float
    max_messages: int | None = None
    # How long to wait for a batch of detections to be durably acked by the
    # broker before giving up and treating the publish as failed (see
    # `_produce` below) — not how long `flush()` blocks on every call, just
    # the ceiling for one.
    produce_timeout_seconds: float = 10.0


def _detection_key(result: DetectionResult) -> bytes:
    """A deterministic identity for one scored window, independent of how
    many times it gets reprocessed and re-published (committed-then-crashed
    input offsets, a consumer rebalance, a manual replay) — a downstream
    reader that wants to dedupe `netstream.detections.v1` can key on this
    (or just on the fields it is built from: host_id/src_ip/window_start/
    window_seconds, already in `DetectionResult.to_dict()`) rather than
    assuming one message per window."""
    key = f"{result.host_id}:{result.src_ip}:{result.window_start}:{result.window_seconds}"
    return key.encode()


def _produce_detections(
    producer: confluent_kafka.Producer,
    detections_topic: str,
    results: list[DetectionResult],
    timeout_seconds: float,
    latencies: list[float],
) -> bool:
    """Publishes every result and blocks until the broker has acked all of
    them (or `timeout_seconds` runs out). Returns whether every single one
    was actually delivered — the caller must not commit the input offset
    that produced `results` unless this is `True`, or a delivery failure
    becomes a silent, uncommittable loss instead of "will be retried on
    the next poll"."""
    if not results:
        return True
    delivered = True

    def _on_delivery(err: object, _msg: object) -> None:
        nonlocal delivered
        if err is not None:
            delivered = False
            logger.error("Failed to publish detection to %s: %s", detections_topic, err)

    for result in results:
        producer.produce(
            detections_topic,
            json.dumps(result.to_dict()).encode(),
            key=_detection_key(result),
            callback=_on_delivery,
        )
        latencies.append(result.latency_ms)
        if result.is_attack:
            logger.info("detection: %s", result.to_dict())

    remaining = producer.flush(timeout_seconds)
    if remaining > 0:
        delivered = False
        logger.error(
            "%d detection(s) still unacknowledged after %.1fs; not committing the input "
            "offset for them",
            remaining,
            timeout_seconds,
        )
    return delivered


def run_kafka_detector(config: KafkaDetectorConfig) -> None:
    """The real I/O loop: `OnlineDetector` plus Kafka plumbing around it.
    Not unit-tested itself — `OnlineDetector.process_message`/`.flush` are,
    and that is where the actual decoding/windowing/scoring logic lives;
    this function is a thin wrapper, the same split the rest of the
    project draws between pure logic and run loops.

    Input offsets are committed manually, only after the detections that
    message produced have been durably published — `enable.auto.commit`
    (the default) would let the consumer group checkpoint an offset before
    `producer.produce()`'s calls actually reach the broker, so a crash
    between the two could commit an input message whose detection was
    never really published, losing it silently. `enable.idempotence` on
    the producer guards the other direction: a retried `produce()` (after
    a transient broker error) can't create a duplicate message on the
    broker side. Neither guards against *this process* reprocessing the
    same window after a crash before it managed to commit — that still
    produces a duplicate `DetectionResult`, which is why `_detection_key`
    gives it a stable identity a downstream reader can dedupe on, instead
    of pretending re-delivery can't happen.
    """
    online = OnlineDetector(config.model, WindowSpec(config.window_seconds))
    consumer = confluent_kafka.Consumer(
        {
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.consumer_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )
    producer = confluent_kafka.Producer(
        {"bootstrap.servers": config.bootstrap_servers, "enable.idempotence": True}
    )
    consumer.subscribe([config.events_topic])

    def _produce(results: list[DetectionResult]) -> bool:
        return _produce_detections(
            producer, config.detections_topic, results, config.produce_timeout_seconds, latencies
        )

    latencies: list[float] = []
    processed = 0
    last_flush = time.monotonic()
    try:
        while config.max_messages is None or processed < config.max_messages:
            msg = consumer.poll(timeout=1.0)
            if msg is not None:
                value = msg.value()
                if msg.error():
                    logger.warning("Kafka consumer error: %s", msg.error())
                elif value is not None:
                    if _produce(online.process_message(value)):
                        consumer.commit(message=msg, asynchronous=False)
                    else:
                        logger.error(
                            "Not committing offset %d for %s/%d: detections may be "
                            "reprocessed and republished (see _detection_key)",
                            msg.offset(),
                            msg.topic(),
                            msg.partition(),
                        )
                    processed += 1

            now = time.monotonic()
            if now - last_flush >= config.idle_flush_seconds:
                if _produce(online.flush()):
                    _commit_if_anything_to_commit(consumer)
                last_flush = now
    finally:
        if _produce(online.flush()):
            _commit_if_anything_to_commit(consumer)
        consumer.close()
        if latencies:
            p50, p95 = _latency_percentiles(latencies)
            logger.info("latency_ms p50=%.1f p95=%.1f over %d detections", p50, p95, len(latencies))
