"""Explicit Brain2Qwerty v1-compatible event preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from sklearn.preprocessing import RobustScaler


OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
RAW_SAMPLING_RATE = 1000.0
SAMPLING_RATE = 50.0
FILTER = (0.1, 20.0)
EVENT_WINDOW = (-0.2, 0.3)
BASELINE = (0.0, 0.2)
CLAMP = (-5.0, 5.0)
CHANNEL_COUNT = 306


@dataclass(frozen=True)
class OfficialV1Event:
    """A single event tensor and its explicit preprocessing provenance."""

    tensor: np.ndarray
    event_time_seconds: float
    metadata: dict[str, object]


class NeuroSelectOfficialV1EventPreprocessing:
    """Continuous-scope, per-event Brain2Qwerty v1-compatible preprocessing."""

    def __init__(
        self,
        signal: np.ndarray,
        channel_names: tuple[str, ...],
        *,
        recording_start_seconds: float,
    ) -> None:
        if signal.ndim != 2 or signal.shape[1] != CHANNEL_COUNT:
            raise ValueError(f"Expected (time, {CHANNEL_COUNT}) signal, got {signal.shape}")
        if not np.isfinite(signal).all():
            raise ValueError("Preprocessed signal contains non-finite values")
        if len(channel_names) != CHANNEL_COUNT:
            raise ValueError(f"Expected {CHANNEL_COUNT} channel names")
        self.signal = np.asarray(signal, dtype=np.float32)
        self.channel_names = channel_names
        self.recording_start_seconds = float(recording_start_seconds)

    @classmethod
    def from_raw(
        cls,
        raw_path: str | Path,
        *,
        recording_start_seconds: float,
    ) -> "NeuroSelectOfficialV1EventPreprocessing":
        """Prepare the complete continuous MEG recording before event extraction."""
        raw = mne.io.read_raw_fif(
            raw_path,
            preload=False,
            verbose=False,
            allow_maxshield=True,
        )
        raw = raw.copy().pick("meg")
        raw.load_data()
        channel_names = tuple(raw.ch_names)
        raw.filter(*FILTER, n_jobs=1, verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto", n_jobs=1, verbose=False)
        raw._data = RobustScaler().fit_transform(raw._data.T).T
        return cls(
            raw.get_data().T.astype(np.float32, copy=False),
            channel_names,
            recording_start_seconds=recording_start_seconds,
        )

    def extract_event(self, event_time_seconds: float) -> OfficialV1Event:
        """Extract one complete event tensor as `(time, channels)` without padding."""
        event_time_seconds = float(event_time_seconds)
        event_start = event_time_seconds + EVENT_WINDOW[0]
        event_duration = EVENT_WINDOW[1] - EVENT_WINDOW[0]
        signal_start = self.recording_start_seconds
        signal_duration = self.signal.shape[0] / SAMPLING_RATE

        # This is the effective operation of neuralset.TimedArray.overlap:
        # overlap bounds are resolved in seconds, then both the relative start
        # and duration are converted independently with Frequency.to_ind
        # (int(round(seconds * frequency))).
        def overlap_slice(
            array_start: float,
            array_duration: float,
            start: float,
            duration: float,
        ) -> tuple[float, float, slice] | None:
            overlap_start = max(start, array_start)
            overlap_stop = min(start + duration, array_start + array_duration)
            if overlap_stop < overlap_start or (
                overlap_stop == overlap_start and signal_duration and duration
            ):
                return None
            start_index = int(round((overlap_start - array_start) * SAMPLING_RATE))
            duration_index = int(round((overlap_stop - overlap_start) * SAMPLING_RATE))
            if duration_index <= 0:
                duration_index = 1
            array_samples = int(round(array_duration * SAMPLING_RATE))
            if start_index > array_samples - duration_index:
                start_index = array_samples - duration_index
            if start_index < 0:
                raise ValueError("Event window is outside the continuous preprocessed signal")
            snapped_start = array_start + start_index / SAMPLING_RATE
            snapped_duration = duration_index / SAMPLING_RATE
            return snapped_start, snapped_duration, slice(
                start_index, start_index + duration_index
            )

        extended = overlap_slice(signal_start, signal_duration, event_start, event_duration)
        if extended is None:
            raise ValueError("Event window is outside the continuous preprocessed signal")
        extended_start, _extended_duration, extended_slice = extended
        extended_data = self.signal[extended_slice]

        # The official extractor receives the event-window start as `start`;
        # baseline offsets are relative to that same start, not the keypress
        # timestamp. Thus [0, 0.2] selects the first baseline portion of the
        # requested [-0.2, 0.3] window.
        baseline_request_start = event_start + BASELINE[0]
        baseline_request_duration = BASELINE[1] - BASELINE[0]
        baseline_relative = overlap_slice(
            extended_start,
            extended_data.shape[0] / SAMPLING_RATE,
            baseline_request_start,
            baseline_request_duration,
        )
        if baseline_relative is None:
            raise ValueError("Event baseline is outside the continuous preprocessed signal")
        baseline_start, _baseline_duration, baseline_slice = baseline_relative
        baseline_start = int(round((baseline_start - extended_start) * SAMPLING_RATE))
        baseline_values = extended_data[
            baseline_start : baseline_start + baseline_slice.stop - baseline_slice.start
        ]
        baseline = baseline_values.mean(axis=0, keepdims=True)
        corrected_extended = extended_data - baseline

        window = overlap_slice(
            extended_start,
            extended_data.shape[0] / SAMPLING_RATE,
            event_start,
            event_duration,
        )
        if window is None:
            raise ValueError("Event window is outside the extended overlap")
        window_time, window_duration, window_slice = window
        window_start = extended_slice.start + window_slice.start
        window_stop = window_start + (window_slice.stop - window_slice.start)
        channel_time = corrected_extended[window_slice]

        # The official extractor never returns the overlap directly. It allocates
        # a zero TimedArray covering the full requested duration and adds the
        # overlap into it, which resolves both sides through the same
        # `_overlap_slice` (neuralset.base.TimedArray.__iadd__). For a
        # full-length overlap that placement is the identity. When the keypress
        # falls exactly halfway between two 50 Hz samples the overlap is one
        # sample short, and the official output keeps a single zero sample at
        # whichever edge the rounding leaves free.
        expected_samples = int(round(event_duration * SAMPLING_RATE))
        output_placement = overlap_slice(
            event_start, event_duration, window_time, window_duration
        )
        source_placement = overlap_slice(
            window_time, window_duration, event_start, event_duration
        )
        if output_placement is None or source_placement is None:
            raise ValueError("Official output placement does not overlap the event window")
        _, _, output_slice = output_placement
        _, _, source_slice = source_placement
        placed = np.zeros((expected_samples, CHANNEL_COUNT), dtype=channel_time.dtype)
        placed[output_slice] = channel_time[source_slice]
        zero_filled_samples = expected_samples - (output_slice.stop - output_slice.start)
        tensor = np.clip(placed, *CLAMP).astype(np.float32)
        return OfficialV1Event(
            tensor=tensor,
            event_time_seconds=float(event_time_seconds),
            metadata={
                "preprocessing_variant": "official_v1_compatible",
                "official_revision": OFFICIAL_REVISION,
                "sampling_rate": SAMPLING_RATE,
                "filter": list(FILTER),
                "scaler": "RobustScaler",
                "event_window": list(EVENT_WINDOW),
                "baseline": list(BASELINE),
                "clamp": list(CLAMP),
                "channels": CHANNEL_COUNT,
                "tensor_layout": "time_channels",
                "dtype": "float32",
                "channel_names": list(self.channel_names),
                "recording_start_seconds": self.recording_start_seconds,
                "event_sample_index": int(round((event_time_seconds - signal_start) * SAMPLING_RATE)),
                "window_start_sample_index": window_start,
                "window_end_sample_index_exclusive": window_stop,
                "official_overlap_samples": int(channel_time.shape[0]),
                "zero_filled_samples": int(zero_filled_samples),
                "zero_padded": bool(zero_filled_samples > 0),
                "preprocessing_order": [
                    "MEG channel selection",
                    "continuous filter",
                    "continuous resample",
                    "continuous RobustScaler",
                    "event-window extraction",
                    "per-event baseline",
                    "official output placement",
                    "per-event clamp",
                ],
            },
        )


def preprocess_event_official_v1(
    raw_path: str | Path,
    event_time_seconds: float,
    *,
    recording_start_seconds: float,
) -> OfficialV1Event:
    """Prepare continuous MEG and extract one explicit official-style event."""
    processor = NeuroSelectOfficialV1EventPreprocessing.from_raw(
        raw_path,
        recording_start_seconds=recording_start_seconds,
    )
    return processor.extract_event(event_time_seconds)
