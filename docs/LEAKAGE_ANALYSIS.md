# Leakage analysis

The prepared subset contains one subject, one session, one recording block,
and eight distinct sentence IDs. The official splitter assigns the selected
records to train/validation/test, but this is not a meaningful generalization
evaluation because only one recording unit is represented and the selected
sample is tiny.

No duplicate sentence text was observed among the eight selected records.
The failed real-data overfit means test CER/WER has intentionally not been
reported. A scientifically meaningful leakage-safe evaluation requires more
recording units and should preserve the official split by subject/session.
