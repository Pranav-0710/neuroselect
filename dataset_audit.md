# Dataset audit

Audit date: 2026-09-13. Values below are based on the current official
Brain2Qwerty repository and the public Hugging Face dataset card. Values not
published there are marked `unknown` or `not verified`; no missing statistics
are fabricated.

## Selected accessible dataset

| Field | Verified value |
|---|---|
| Dataset name | DECOMEG / SpanishBCBL |
| Source URL | https://huggingface.co/datasets/bcbl190626/SpanishBCBL |
| Official code | https://github.com/facebookresearch/brain2qwerty |
| License | CC BY-NC 4.0; recordings belong to BCBL |
| Access status | Public, non-gated Hugging Face dataset; full archive is approximately 262 GB |
| Subjects | 35 volunteers reported in the card; 19 unique MEG participants after the official v1 participant mapping |
| Sessions | Dataset card reports multiple recording directories; exact participant-session count for sentence trials is not independently recomputed here |
| Modality | MEG and EEG |
| MEG sensors | 306 (102 magnetometers + 204 planar gradiometers) |
| EEG channels | 64 |
| Sampling rate | 1 kHz |
| Recording duration | Approximately 21.5 h MEG and 17.7 h EEG of typing in total |
| Sentences/trials | Approximately 5.1K MEG sentences and 4K EEG sentences |
| Unique sentences | 128 unique declarative Spanish sentences per session, according to the dataset card |
| Labels | Behavioral logs contain stimuli, keystrokes, and timing; v1 maps Spanish keys to 29 classes including blank-like special/number categories |
| Raw storage | Approximately 262 GB |
| Estimated download | Approximately 262 GB for the full public release; modality-specific download is smaller but not quantified in the card |
| Input format | Raw `.fif` MEG or BrainVision `.eeg/.vhdr/.vmrk` EEG; the baseline manifest uses exported `(time, channels)` `.npy`/`.npz` sentence windows |
| Subject identifiers | Public IDs such as `S1`; repeated-person mappings are documented for MEG |
| Session identifiers | Encoded in recording/timeline metadata; exact normalized count is not verified in this audit |
| Official train/val/test | v1 code creates an 80/10/10 split by TF-IDF similarity clusters; v2 code defines a deterministic split by unique sentence text but v2 data is embargoed |
| Same sentence across subjects | Likely yes because each session uses the same 128-sentence stimulus set; exact cross-subject overlap count is not recomputed from downloaded events |
| Text leakage risk | High if trial-level random splitting is used. Identical text and paraphrase-similar text must be grouped before splitting |
| Author preprocessing | v1 uses 50 Hz extraction, 0.1–20 Hz filtering, 0–0.2 s baseline, robust scaling, clamp 5, and keystroke windows from -0.2 to 0.5 s; these are code-verified, not independently rerun here |
| Disk after preprocessing | Unknown; depends on cache representation and selected modality |
| RAM | Unknown for full preprocessing; raw continuous files and event caches can exceed laptop memory if eagerly loaded |
| GPU | Full official training is configured for up to 8 GPUs; the compact baseline is designed for one free-tier GPU or CPU smoke tests |
| Colab recommendation | Start with one modality, one subject/session subset, local manifest, batch size 1–4, compact model, and checkpointing. Do not mirror 262 GB into a free-tier runtime |

## Access findings

- The official README states that Brain2Qwerty v2 data is under embargo.
- The official README links Brain2Qwerty v1 to the public SpanishBCBL dataset.
- The Hugging Face dataset is not gated and its metadata is reachable, but raw
  files are large. No access restriction was bypassed.

## Leakage-safe split policy

The repository's v1 official split is preserved as a documented reference.
NeuroSelect's loader accepts explicit split assignments and includes
`split_by_unique_text` for a clean sentence-text grouping policy. A full audit
of duplicate and paraphrase overlap requires building the official events from
downloaded logs; that has not been claimed as complete because the archive is
not present locally.

## Problems and unresolved items

1. The full raw release is too large to download as part of this phase.
2. The downloaded S22 subset contains one subject, one session, and one block.
3. Tiny real-data overfitting has not succeeded, so no real CER/WER is claimed.

## Phase 2 downloaded subset

The usable public subset is subject S22, session 1, block 1:

- `MEG/FIF/22_9788/231214/block1.fif` (1,107,644,612 bytes)
- `MEG/logs/S22-session1_block1_list1.mat` (approximately 222 KB)

The official event pipeline produced 2,743 cleaned events, including 132
sentence events and 2,119 keystrokes before postprocessing. Eight complete
production trials were materialized for the baseline. The MEG recording is
1000 Hz with 306 MEG channels. The verified dataset-specific CTC vocabulary
is documented in `docs/DATASET_MAPPING.md`.
