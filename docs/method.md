# Method

Electrical stimulation couples into an ECG lead as trains of narrow,
high-amplitude spikes, often followed by a smooth recovery of the
amplifier. Each train recurs at the same latency in every stimulation
cycle. Heartbeats are not locked to the stimulation. The method uses both
facts and assumes nothing about the device or protocol: the timing,
threshold and sampling rate are all taken from the recording itself.

`clean_ecg` runs these steps:

## 1. Polarity, without filtering yet

Polarity (whether R waves point up) is decided with NeuroKit2's
`ecg_invert`. That heuristic is tuned for ~1 kHz ECG, and heavy artifact
can flip its answer, so it is run on a copy that was median-filtered over
5 ms (which deletes 1-3 sample spikes but keeps the ~100 ms QRS) and
brought to ~1 kHz.

Cleaning then runs on the **unfiltered** signal. Filtering a signal that
still contains the spikes smears them into stimulation-locked content
that no later stage can remove: the 0.5 Hz high-pass turns each train
into a slow baseline shift, and a zero-phase 60 Hz notch rings for
~150 ms before and after every spike. Curvature (below) ignores baseline
wander, and at ≥ 1 kHz mains hum has negligible curvature, so nothing is
lost. Below 1 kHz, where mains itself looks spike-like, the filters run
first.

## 2. The discriminator: local curvature

For each sample, curvature is its distance from the straight line
through its two neighbours:

```
c[i] = x[i] - (x[i-1] + x[i+1]) / 2
```

Even the sharpest corners of a QRS change smoothly from one sample to the
next, while a stimulation spike does not. Simpler criteria fail:

- **Amplitude or derivative thresholds.** A weakly coupled spike can have
  a smaller derivative than the QRS upstroke.
- **Excursion width.** A spike that lands on the QRS upstroke reads as
  "wide" and is left in.

Curvature separates the two wherever in the beat the spike lands.

**Sampling rate matters.** A smooth waveform's curvature scales as
`1/fs²`, but a one-sample spike's does not. At 5 kHz a QRS's curvature is
~25× smaller than at 1 kHz, while the spikes are unchanged. That is why
cleaning runs at the recording's native rate and decimation happens
afterwards.

**Threshold** (`resolve_curv_thresh`). The base value is 10,000 at 1 kHz.
It is raised by `(1000/fs)²` below 1 kHz and never lowered above it,
because noise curvature does not shrink with rate. It is then clamped to
2-8× the recording's own QRS curvature, measured as the 25th percentile
across beats at the R tips:

- The **lower bound (2×)** protects R peaks on high-gain recordings. On
  one subject with a 3× higher gain, the fixed threshold sat below the
  QRS curvature, so R tips were shaved and Sham showed a false "artifact"
  at the heart rate.
- The **upper bound (8×)** makes recordings in other units (e.g.
  millivolts) work.

If heartbeats can't be found (< 10 s of signal, or NeuroKit2
unavailable), the base value is used.

## 3. Where to look: the artifact window

Cleaning is restricted to a window, in milliseconds relative to each
stimulation onset, that is found from the data.

**With markers** (`calibrate_artifact_window`):

1. Collect the latency of every above-threshold curvature event, relative
   to its marker. The search covers 95% of the median stimulation period
   and starts 5% of the period before each marker, because software
   markers can lag the hardware pulse. On one subject the first pulse led
   its marker by 12.5 ms.
2. Histogram the latencies in 20 ms bins. The artifact forms bins far
   above the baseline: more than 4× the 25th-percentile bin (and at least
   5 events above it). The baseline is the 25th percentile, not the
   median, because a train filling more than half the period would
   otherwise make the median bin itself artifact-level.
3. Merge elevated bins across gaps of up to 80 ms. Fast pulse trains alias
   against the bin width into a comb of elevated bins. Pad the merged
   cluster by 15 ms on each side.
4. If no bin is elevated, there is no stimulation-locked artifact and the
   signal is returned unchanged.

**Without markers** (`detect_pseudo_stim_times`):

1. Flag curvature events over the whole recording.
2. **Period.** Autocorrelate a 20 ms-binned indicator of the events. The
   true stimulation period is elevated at its 2nd and 3rd multiples too,
   whereas the much faster within-train pulse interval is not. The
   smallest candidate whose harmonics are all elevated is accepted. For
   sparse events (weak coupling), a Rayleigh test folds every event onto
   each candidate period and pools evidence across all cycles. The
   significance level is Bonferroni-corrected, and only periods seen at
   least 30 times are considered.
3. **Phase.** The artifact sits on the side of the cycle opposite its
   longest quiet stretch. This is used instead of a density-weighted mean,
   which a long, unevenly flagged train pulled ~200 ms off centre on one
   subject. Pseudo-markers are phased so that the artifact sits in the
   middle of the calibration search range.
4. **Refine the period.** A period that is off by 1 ms drifts by 150 ms
   over 150 cycles. Each cycle's event-cluster centre is regressed on its
   cycle number, with outlier rejection, and the slope becomes the period.
5. The pseudo-markers then go through exactly the marker-based path
   above. The calibration acts as an independent check: if the period or
   phase were wrong, no clean latency cluster would appear and the signal
   would be left alone.

## 4. Stage 1: despike

Within the window of every cycle, samples with |curvature| above the
threshold are flagged and dilated by 1 ms on each side (a duration, so
spike tails are covered at any sampling rate). The flagged samples are
then replaced by linear interpolation from their neighbours.

**Optional `phase_locked=True`.** On weakly coupled sites, some cycles'
spikes never clear the threshold. The median curvature at each window
offset, taken across all cycles, is still elevated at stimulation-locked
offsets, while QRS curvature averages out because it is not locked. Those
offsets are despiked in every cycle. The option is off by default; it
removed residual stimulation-locked spikes to the random-time null on the
weakest validation recordings without changing any other.

## 5. Stage 2: subtract the stimulation-locked template

Despiking cannot catch smooth after-effects: amplifier recovery (seen
once as a ~30 Hz sawtooth after each pulse) and spike tails.
`subtract_stim_template` handles them:

1. High-pass the despiked signal at 10 Hz.
2. Take the per-offset **median across all cycles**, over one period.
   Heartbeats fall at random offsets, so they don't enter the median.
3. Zero every offset whose median is not significantly non-zero (more
   than 5 robust standard errors). Subtracting an insignificant template
   would only add noise.
4. Re-align each cycle by up to ±5 ms on its >100 Hz content, to absorb
   marker jitter, and recompute the median (2 iterations).
5. Confine the template to the artifact window ±100 ms and subtract it.

At least 20 cycles are needed, otherwise this stage is skipped.

Step 3 sets the trade-off: after-effects smaller than the beat-to-beat
variability at that offset are left in place. On the validation data the
remaining smooth (10-100 Hz) stimulation-locked content is at the
random-time null. On a synthetic signal with a large exponential tail,
about 1% of the original artifact remained.

## 6. Filter and decimate

The 60/50 Hz notch and 0.5 Hz high-pass are applied to the cleaned signal
(and to the uncleaned comparison copy, `ecg_preprocessed`). Both are then
decimated to `target_fs` with a zero-phase FIR anti-aliasing filter.

## Limitations

- **Periodic stimulation is assumed.** Continuous or aperiodic
  stimulation waveforms are out of scope.
- **Background noise that the stimulation did not cause is left alone.**
  Muscle noise, for example, is not locked to the pulses, so no artifact
  method can subtract it.
- **Weak artifact on a high-gain ECG.** Artifact below 2× the QRS
  curvature is not despiked per cycle. HRV was unaffected in every case
  measured; use `phase_locked=True` if the residual matters.
- **Marker-free detection needs enough contamination overall.** Very
  weak coupling in a short recording may go undetected. In every such case
  measured, HRV differed from the marker-based result by < 0.01 ms.
- **Edges of a synthetic stress test.** A very low sampling rate
  combined with a dense pulse train (256 Hz, 50 Hz pulses) leaves too few
  clean samples between pulses. A long period seen over few cycles
  (4.1 s, ~60 cycles) was estimated to within ~1%. Both cases were
  substantially cleaned but not perfectly, and neither ended up worse
  than the raw signal.
- **HRV is plain NeuroKit2.** If you have a validated HRV pipeline with
  its own beat correction, feed it the cleaned signal.
