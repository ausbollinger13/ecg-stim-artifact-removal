# Validation

## Metrics

Each recording was cleaned exactly as a user would clean it, with
`clean_ecg(raw, ts, fs)` on the raw ECG at its recorded rate, both with
and without the stim markers. The 1 kHz output was scored on:

| metric | definition | ideal |
|---|---|---|
| stim-locked residual | peak-to-peak of the stimulation-locked average, divided by the 99th percentile of the same quantity at random times. Computed in two bands: > 100 Hz (spikes) and 10-100 Hz (smooth after-effects) | ≤ ~1 |
| R in/out | median R amplitude of beats inside the stimulation-train window, divided by the median outside it | 1 |
| noise in/out | RMS of the 30-250 Hz baseline (QRS ± 60 ms excluded) inside the train window, divided by outside it | 1 |
| HRV | NeuroKit2 time-domain HRV, before and after | Sham-like |

## Dedicated stimulation-artifact recordings

Six subjects each recorded Sham, auricular (cymba conchae), cervical and
back stimulation for ~150 s (one train per second, 150 cycles):

- Subjects 1-5 at 5 kHz; Subject 6 at 25 kHz.
- Protocols: 25-100 Hz pulses in 0.25-0.5 s trains.
- ECG gains differed 3× across subjects.

Subject 6 was added after the method was frozen, as an out-of-sample test
at a higher sampling rate. The method was not changed for it.

Results:

- **Sham.** All 12 runs (6 subjects, with and without markers) returned
  unchanged. No artifact was found.
- **Spike residual (> 100 Hz).** At the null (≤ 1.15) on 16/18 active
  recordings with markers and 15/18 without. The exceptions, all reduced
  from 10-17× raw:
  - Subject 4 Back: 1.37-1.45
  - Subject 2 Back: 1.40-1.61
  - Subject 3 Cervical, marker-free only: 1.29
- **Smooth residual (10-100 Hz).** At the null everywhere (≤ 1.28).
- **R in/out.** 0.98-1.03. QRS amplitude is unchanged inside the trains.
- **Noise in/out after cleaning.** 0.65-1.07. One recording at 1.18 had
  the same excess before cleaning, i.e. its own background.
- **HRV.** Unchanged where the raw signal was clean and restored where it
  was corrupted. For example, in a 25 kHz back recording the peak
  detector found 260 "beats" before cleaning and 132 after (RMSSD 64 ms,
  in line with that subject's Sham).
- **Sampling rate.** At 25 kHz, cleaning at 5 kHz was as good as native
  and ~6× faster. Cleaning at 1 kHz left spikes on the auricular and
  cervical recordings.

Four of these subjects had every pulse logged on a hardware trigger
channel. These per-pulse checks found two calibration defects during
development, both fixed:

- The first pulse led its software marker by ~12.5 ms.
- A > 50% duty-cycle train hid in the histogram baseline.

After the fixes, the per-pulse residual on the 25 kHz subject was
0.89-0.94× the noise floor.

## A 28-subject study

336 recordings: Sham, auricular and finger stimulation during a P300 task:

- **Sham.** 0/168 recordings altered.
- **Detection.** Artifact found on 78/84 auricular and 48/84 finger
  recordings (finger coupling is weaker). Undetected recordings are
  returned unchanged.
- **HRV.** The cleaned HRV correlated with the study's independent
  reference pipeline at r = 0.995-0.997.
- **Marker-free vs markers.** With the marker channel withheld, cleaned
  HRV was identical (r = 1.0000) wherever both paths detected the
  artifact.

## Example data

`examples/validate_examples.py` reproduces the metrics on the 60 s
excerpts in `examples/data`:

- **Subject A** is subject 5 above (5 kHz).
- **Subject B** is subject 6 (25 kHz).

Results (spike / smooth residual after cleaning, × null):

| subject | condition | markers | spike raw → clean | smooth raw → clean | beats | RMSSD raw → clean (ms) |
|---|---|---|---|---|---|---|
| A | sham | either | not altered | not altered | 64 | 33.4 → 33.4 |
| A | cymba | no | 1.40 → 0.74 | 0.81 → 0.76 | 66 | 45.4 → 45.4 |
| A | cervical | no | 2.84 → 0.70 | 0.88 → 0.85 | 71 | 29.1 → 29.2 |
| A | back | no | 5.44 → 0.30 | 1.70 → 0.69 | 67 | 34.4 → 34.4 |
| A | back | yes | 5.88 → 0.23 | 1.74 → 0.95 | 67 | 34.4 → 34.4 |
| B | sham | either | not altered | not altered | 54 | 59.9 → 59.9 |
| B | cymba | no | 1.11 → 0.72 | 0.88 → 0.74 | 54 | 73.0 → 73.0 |
| B | cervical | no | 2.99 → 0.82 | 0.62 → 0.60 | 53 | 73.6 → 73.8 |
| B | back | no | 4.77 → 0.28 | 1.57 → 0.49 | 51 | 67.3 → 67.4 |
| B | back | yes | 4.65 → 0.27 | 1.96 → 0.46 | 51 | 67.3 → 67.4 |

The raw ratios here are lower than in the full recordings. The ratio
grows with the square root of the number of cycles averaged: 60 here,
150 in the full recordings. In these excerpts the artifact did not corrupt NeuroKit2's
beat detection, so HRV barely moves. The before/after figure in the README
shows the waveforms.
