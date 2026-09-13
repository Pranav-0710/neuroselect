# NeuroSelect dataset card

**Dataset:** DECOMEG / SpanishBCBL  
**Source:** [Hugging Face](https://huggingface.co/datasets/bcbl190626/SpanishBCBL)  
**License:** CC BY-NC 4.0, non-commercial use with attribution.

The public Brain2Qwerty-related release contains de-identified MEG and EEG
recordings of healthy Spanish-speaking adults typing briefly memorized
sentences. It reports 35 volunteers, 306 MEG sensors or 64 EEG channels, 1 kHz
sampling, 128 sentences per session, and approximately 262 GB total storage.

Brain2Qwerty v2 data is currently documented as under embargo. This card
therefore describes the usable v1-related release, not v2. The raw archive is
not bundled with NeuroSelect. Reproducible experiments should record the exact
downloaded file subset, event-building code version, preprocessing parameters,
and manifest checksum.

The NeuroSelect baseline consumes sentence-level signal windows exported to a
JSONL manifest. It does not infer metadata, change the official split, or
silently discard raw trials.

