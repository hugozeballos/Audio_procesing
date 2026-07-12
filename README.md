# Audio Processing Pipeline

An end-to-end pipeline for building and benchmarking a speech enhancement dataset: audio is ingested from a Google Drive folder into a versioned Hugging Face Hub dataset (with a deduplicated manifest and dev/test splits), preprocessed (silence trimming + peak normalization), run through multiple pluggable speech-enhancement backends, and scored with objective audio quality metrics.

## Pipeline / How It Works

### 1. Ingestion — Drive → Hugging Face Hub
`dataset-audio-raw/ingest_drive_to_hf.py` authenticates to Google Drive with a service account (`pydrive2` + `google-auth`), recursively walks a target Drive folder (including shortcuts) and collects audio files by extension (`.wav .flac .mp3 .m4a .aac .ogg .opus .wma`). For each file it:
- Downloads the bytes and computes a **SHA-256** hash, which is used as an idempotency key — files already present in the manifest are skipped, so re-running the script never re-uploads duplicates.
- Extracts duration, sample rate, channel count and bitrate via **ffprobe** (external binary, must be on `PATH`).
- Uploads new files to a Hugging Face **dataset** repo (`huggingface_hub.HfApi.create_commit`) under a fixed layout: `data/raw/YYYY/MM/DD/<sha256>__<original_filename>`.
- Appends the new entries to `manifests/manifest.csv` (`path, sha256, duration_s, sample_rate, channels, bitrate, size_bytes, source, drive_file_id, created_at`) and commits the updated CSV.
- Maintains `splits/dev.list`, a plain-text list of dataset-relative paths, prioritizing the shortest clips first (for fast iteration) and growing it incrementally up to a configured target size.
- Supports a `FORCE_REINDEX` mode that rebuilds the whole manifest from scratch (re-hashing/re-probing every Drive file) and a `FORCE_COMMIT` flag to force a no-op commit as a "last refresh" heartbeat.

`preprocesing/gen_splits.py` is a companion, standalone script that (re)derives `splits/dev.list` and `splits/test.list` directly from `manifests/manifest.csv`: it sorts by duration ascending and takes the N shortest clips for `dev`, then draws a seeded random sample from the remainder (non-overlapping with `dev`) for `test`.

### 2. Preprocessing — VAD trimming + peak normalization
Implemented in `utils/audio_prep.py` and driven by `preprocesing/prep_dev.py`:
- **`trim_tails_vad`** — a lightweight, dependency-free voice activity detector. It computes RMS energy in dBFS over 30 ms frames and trims leading/trailing silence only when a continuous silent run exceeds a minimum duration (default 500 ms), keeping a small margin (default 150 ms) around the trimmed edges. It never touches interior silence. Note: this is a hand-written energy-threshold VAD, **not** a library such as `webrtcvad` or `silero-vad` (neither appears in the codebase or `requirements.txt`).
- **`peak_normalize_minus1_dbfs`** — scales the waveform so its peak sits at −1 dBFS (~0.891 linear amplitude). It only attenuates (never boosts) a signal that's already quieter than the target.

`preprocesing/prep_dev.py` reads every file listed in `splits/dev.list` (locally if present, otherwise streamed from the HF Hub dataset repo via `hf_hub_download`), applies VAD trimming then peak normalization, writes the result as 16-bit PCM WAV under `dataset-audio-raw/prep/`, and logs input/output duration and peak/gain values to `prep/prep_log.csv`.

### 3. Enhancement — pluggable backends
`enh/run_enh_dev.py` is the orchestrator. Backend selection is a simple **registry + factory pattern**: a `BACKENDS` dict maps string names to backend classes, and `get_backend(name)` instantiates the requested one. The backend is chosen via the `--backend` CLI flag (or `ENH_BACKEND` env var), and an intensity `--preset` (`light|medium|aggressive`, or `ENH_PRESET`) is passed through to each backend's `enhance(x, sr, preset, opts)` method — a common interface defined in `enh/backends/base.py`. For each dev file it re-applies VAD trim + peak normalization, runs the selected backend, and logs per-file timing (including real-time factor) and levels to `enh/<backend>/<preset>/enh_log.csv`.

Backends actually wired into the `BACKENDS` registry (`enh/run_enh_dev.py`):

| Backend key | File | Model / mechanism |
|---|---|---|
| `metricgan` | `enh/backends/metricgan.py` | **MetricGAN+** via SpeechBrain (`speechbrain.pretrained.SpectralMaskEnhancement`, pretrained checkpoint `speechbrain/metricgan-plus-voicebank`), 16 kHz, with per-preset wet/dry mix and post-gain |
| `voicefixer` | `enh/backends/voicefixer_real.py` | **VoiceFixer** (`voicefixer` package, `VoiceFixer().restore(...)`), 22.05 kHz internally; preset maps to `mode=1` (denoise only) or `mode=0` (full restoration) |
| `deepfilternet` | `enh/backends/deepfilternet.py` | **DeepFilterNet**, invoked as an external CLI subprocess (`python -m df.enhance in.wav out.wav`, from the `deepfilternet` package), 48 kHz, with wet/dry mix and peak normalization |
| `sepformer` | `enh/backends/sepformer.py` | SepFormer speech enhancement via SpeechBrain (`speechbrain/sepformer-wham16k-enhancement`), 16 kHz |
| `resemble_enh` | `enh/backends/resemble_enh.py` | **Resemble-Enhance** (`resemble_enhance` package, `denoise`/`enhance` inference calls), 44.1 kHz, with per-preset solver/NFE/tau (diffusion-style config) |
| `clearervoice` | `enh/backends/clearervoice.py` | Generic external-CLI wrapper; requires an `ENH_CLEARERVOICE_CMD` command template to be set — no bundled model, so it's only usable if a ClearerVoice CLI is separately installed |

Two additional files (`voicefixer_stub.py`, a simple amplitude gate, and `resemble_stub.py`, a one-pole low-pass smoother) exist as lightweight placeholders but are **not** registered in `run_enh_dev.py`'s `BACKENDS` dict, so they are not part of the active pipeline — dev/testing scaffolding only.

So, confirming the description in the project brief: **MetricGAN+, VoiceFixer, and DeepFilterNet are all genuinely implemented and callable** through the pluggable backend interface, alongside two extra real backends (SepFormer, Resemble-Enhance) not originally mentioned.

### 4. Metrics & reporting
- `enh/compute_metrics.py` scans `dataset-audio-raw/enh/<backend>/<preset>/` outputs and, for each dev file, computes: **STOI** (`pystoi`, intelligibility vs. the preprocessed reference), **SRMR** (`srmrpy`, optional), **integrated loudness in LUFS** (`pyloudnorm`), and a clipping rate — writing everything to `enh/enh_metrics.csv`.
- `enh/report.py` aggregates that CSV by backend/preset (mean/percentile STOI, mean clip rate, mean RTF), flags outliers (any clipping or STOI < 0.75), and writes `report_summary.csv`, `report_outliers.csv`, and `report_detail.csv`.

## Tech Stack

- **Language:** Python 3
- **Ingestion:** `pydrive2`, `google-auth` / `google-auth-oauthlib` (Drive service-account access), `huggingface_hub` (dataset repo commits/downloads), `ffprobe` (external binary, for audio metadata)
- **Audio I/O & DSP:** `soundfile`, `scipy` (polyphase resampling), `numpy`
- **Enhancement models:** `speechbrain` + `torch` (MetricGAN+, SepFormer), `voicefixer`, `deepfilternet`, `resemble-enhance`
- **Metrics:** `pystoi`, `pyloudnorm`, `srmrpy` (optional)
- **Utilities:** `pandas`, `tqdm`, `python-dotenv`
- `librosa` and `numba` are pinned in `requirements.txt` but are not directly imported anywhere in the current codebase.

## How to Run

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`ffprobe` (part of FFmpeg) must also be available on `PATH` for the ingestion step.

### Credentials
Create a `.env` file at the repo root (loaded via `python-dotenv`, and already excluded by `.gitignore`) with:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json   # Drive service account, folder shared with its email
DRIVE_FOLDER_ID=<google-drive-folder-id>
HF_TOKEN=<huggingface-write-token>
HF_REPO_ID=<namespace/dataset-name>
AUDIO_ROOT=data/raw          # optional, defaults to data/raw
DEV_LIST_SIZE=80             # optional, target size for splits/dev.list
ENH_BACKEND=metricgan        # optional, default backend for run_enh_dev.py
ENH_PRESET=medium            # optional: light | medium | aggressive
ENH_DEVICE=cpu               # optional: cpu | cuda
```

### Run the pipeline
```powershell
# 1) Ingest new audio from Drive into the HF Hub dataset (manifest + dev.list)
python dataset-audio-raw/ingest_drive_to_hf.py

# 2) (Re)generate dev/test splits from the manifest
python preprocesing/gen_splits.py --dataset-dir dataset-audio-raw --dev-size 80 --test-size 0

# 3) Preprocess the dev split (VAD trim + peak normalize -> WAV)
python preprocesing/prep_dev.py

# 4) Run a speech-enhancement backend over the preprocessed dev split
python enh/run_enh_dev.py --backend metricgan --preset medium
python enh/run_enh_dev.py --backend voicefixer --preset medium
python enh/run_enh_dev.py --backend deepfilternet --preset medium

# 5) Compute objective quality metrics (STOI / SRMR / LUFS / clipping)
python enh/compute_metrics.py --dataset-dir dataset-audio-raw

# 6) Aggregate results and flag outliers
python enh/report.py
```

## Notes / Caveats
- Ingestion, preprocessing, and enhancement are all decoupled by files on disk (`manifests/manifest.csv`, `splits/*.list`, `prep/`, `enh/<backend>/<preset>/`), so each stage can be re-run independently.
- Raw and processed audio are intentionally excluded from git via `.gitignore`; only `manifests/manifest.csv`, `splits/*.list`, and this `README.md` are tracked at the dataset-folder level, with the audio itself living on the Hugging Face Hub dataset repo (private).
