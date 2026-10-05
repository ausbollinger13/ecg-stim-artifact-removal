# Example data

Eight 60 s single-lead ECG excerpts from two subjects, recorded during
transcutaneous electrical stimulation. Each subject was recorded in four
conditions, one train per second:

| file | sampling rate | stimulation site |
|---|---|---|
| `subjectA_sham.mat` | 5 kHz | none (Sham) |
| `subjectA_cymba.mat` | 5 kHz | auricular (cymba conchae) |
| `subjectA_cervical.mat` | 5 kHz | cervical |
| `subjectA_back.mat` | 5 kHz | back |
| `subjectB_sham.mat` | 25 kHz | none (Sham) |
| `subjectB_cymba.mat` | 25 kHz | auricular (cymba conchae) |
| `subjectB_cervical.mat` | 25 kHz | cervical |
| `subjectB_back.mat` | 25 kHz | back |

Stimulation protocols:

- **Subject A:** 100 Hz pulses in ~0.5 s trains.
- **Subject B:** 80 Hz pulses, 40 per 0.5 s train.

Each `.mat` file holds two variables:

- `ecg`: float32 samples in the amplifier's raw units (not calibrated to
  µV), with the median removed.
- `fs`: sampling rate in Hz.

Time starts at 0. `*_stim_times.csv` lists the stimulation-train onset
times in seconds, on the same clock, one per line with no header. These
come from software markers and can lag the first hardware pulse by
~10 ms. Use them with `--stim-times` or `clean_ecg(..., stim_times=...)`,
or ignore them and let the marker-free path find the artifact.

## De-identification

The excerpts contain the ECG channel only. Removed:

- all other channels;
- recording dates and times (timestamps were reset to 0);
- device and file metadata;
- participant identifiers (subjects are coded A and B).

Each excerpt is a 60 s window starting 40 s into the recording.

## License

The example data are distributed with the code under the repository's MIT
license, for testing and demonstrating the method.
