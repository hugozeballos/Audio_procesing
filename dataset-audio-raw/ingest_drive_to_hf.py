#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drive -> HF Hub audio importer with manifest and dev split.
- Auth: Google Service Account (folder shared to SA email) + HF token.
- Idempotent by sha256. No duplicates uploaded.
- Writes/updates:
  - data/raw/YYYY/MM/DD/<sha256>__orig.ext
  - manifests/manifest.csv
  - splits/dev.list  (kept small and stable by target size)
Requires ffprobe in PATH.
"""

import os
import io
import csv
import sys
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd, create_repo
from huggingface_hub import hf_hub_download


# ---------- Config ----------
load_dotenv(find_dotenv())
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
AUDIO_ROOT = os.getenv("AUDIO_ROOT", "data/raw")

# Flags
FORCE_REINDEX = os.getenv("FORCE_REINDEX") == "1"
FORCE_COMMIT  = os.getenv("FORCE_COMMIT")  == "1"
DEV_LIST_SIZE = int(os.getenv("DEV_LIST_SIZE", "0"))

# Extensiones permitidas
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}

# Rutas en repo
MANIFESTS_DIR = "manifests"
SPLITS_DIR = "splits"
MANIFEST_CSV = f"{MANIFESTS_DIR}/manifest.csv"
DEV_LIST = f"{SPLITS_DIR}/dev.list"
README = "README.md"

# ---------- Helpers ----------
def fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def check_env():
    if not GOOGLE_APPLICATION_CREDENTIALS or not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
        fail("GOOGLE_APPLICATION_CREDENTIALS missing or file not found.")
    if not DRIVE_FOLDER_ID:
        fail("DRIVE_FOLDER_ID missing.")
    if not HF_TOKEN:
        fail("HF_TOKEN missing.")
    if not HF_REPO_ID:
        fail("HF_REPO_ID missing.")

def gauth_drive() -> GoogleDrive:
    """Authenticate to Google Drive using Service Account."""
    gauth = GoogleAuth(settings={
        "client_config_backend": "service",
        "service_config": {
            "client_json_file_path": GOOGLE_APPLICATION_CREDENTIALS,
        }
    })
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

def list_drive_files(drive: GoogleDrive, folder_id: str) -> List[Dict]:
    """Lista recursiva de archivos de audio descargables desde Drive."""
    out: List[Dict] = []

    def walk(fid: str):
        q = f"'{fid}' in parents and trashed=false"
        for f in drive.ListFile({'q': q}).GetList():
            mt = f['mimeType']
            if mt == 'application/vnd.google-apps.folder':
                walk(f['id'])
            elif mt == 'application/vnd.google-apps.shortcut':
                tid = f.get('shortcutDetails', {}).get('targetId')
                if tid: walk(tid)
            else:
                title = f['title']
                size = int(f.get('fileSize', 0) or 0)
                ext = Path(title).suffix.lower()
                if size > 0 and ext in ALLOWED_EXT:
                    out.append(dict(id=f['id'], title=title, mimeType=mt, fileSize=size))
    walk(folder_id)
    return out

def is_audio_by_ext(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT

def compute_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def ffprobe_audio_bytes(data: bytes) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Extract duration, sample_rate, channels, bitrate using ffprobe from bytes.
    Returns (duration_s, sr, ch, bitrate).
    """
    # Write to temp file to let ffprobe read container metadata safely.
    tmp = Path(".tmp_probe")
    tmp.mkdir(exist_ok=True)
    tmpf = tmp / "probe_input"
    tmpf.write_bytes(data)
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,bit_rate",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            str(tmpf)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(out)
        duration = None
        bitrate = None
        sr = None
        ch = None

        if "format" in info:
            duration = float(info["format"].get("duration")) if info["format"].get("duration") else None
            br = info["format"].get("bit_rate")
            bitrate = int(br) if br else None

        # pick first audio stream
        streams = info.get("streams", [])
        for s in streams:
            if "sample_rate" in s or "channels" in s:
                sr = int(s.get("sample_rate")) if s.get("sample_rate") else sr
                ch = int(s.get("channels")) if s.get("channels") else ch
                break

        return duration, sr, ch, bitrate
    except subprocess.CalledProcessError as e:
        print("ffprobe failed:", e.output.decode(errors="ignore"), file=sys.stderr)
        return None, None, None, None
    finally:
        try:
            tmpf.unlink(missing_ok=True)
            # keep .tmp_probe dir for reuse
        except Exception:
            pass

def ts_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def target_relpath(sha256: str, orig_name: str) -> str:
    today = datetime.now(timezone.utc)
    y = today.year
    m = f"{today.month:02d}"
    d = f"{today.day:02d}"
    ext = Path(orig_name).suffix.lower()
    safe_orig = Path(orig_name).name.replace("/", "_")
    return f"{AUDIO_ROOT}/{y}/{m}/{d}/{sha256}__{safe_orig}"

def ensure_repo(api: HfApi, repo_id: str, token: str):
    # Create if missing
    try:
        api.repo_info(repo_id=repo_id, token=token, repo_type="dataset")
    except Exception:
        create_repo(repo_id=repo_id, token=token, repo_type="dataset", private=True)

    
def read_manifest_df(api: HfApi, repo_id: str, token: str) -> pd.DataFrame:
    try:
        local = hf_hub_download(repo_id=repo_id, filename=MANIFEST_CSV, token=token, repo_type="dataset")
        return pd.read_csv(local)
    except Exception:
        return pd.DataFrame(columns=[
            "path","sha256","duration_s","sample_rate","channels","bitrate",
            "size_bytes","source","drive_file_id","created_at"
        ])

def read_list(api: HfApi, repo_id: str, token: str, path: str) -> List[str]:
    try:
        local = hf_hub_download(repo_id=repo_id, filename=path, token=token, repo_type="dataset")
        with open(local, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except Exception:
        return []

def write_commit(api: HfApi, repo_id: str, token: str, ops: List[CommitOperationAdd], commit_msg: str):
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=commit_msg,
        token=token
    )

def ensure_readme(api: HfApi, repo_id: str, token: str):
    try:
        files = api.list_repo_files(repo_id=repo_id, token=token, repo_type="dataset")
        if README in files:
            return
    except Exception:
        pass
    content = (
        "# dataset-audio-raw (private)\n\n"
        "- Source: Google Drive\n"
        "- Layout: data/raw/YYYY/MM/DD/<sha256>__orig.ext\n"
        "- Duplicates: avoided by sha256\n"
        "- Splits: lists under /splits\n\n"
        "## Changelog\n- v0.1: initial import\n"
    ).encode("utf-8")
    ops = [CommitOperationAdd(path_in_repo=README, path_or_fileobj=io.BytesIO(content))]
    write_commit(api, repo_id, token, ops, "Add minimal README")

# ---------- Main ----------
def main():
    check_env()
    HfFolder.save_token(HF_TOKEN)
    api = HfApi()
    ensure_repo(api, HF_REPO_ID, HF_TOKEN)
    ensure_readme(api, HF_REPO_ID, HF_TOKEN)

    drive = gauth_drive()
    files = list_drive_files(drive, DRIVE_FOLDER_ID)

    # Filter audio-like by extension
    audio_files = [f for f in files if is_audio_by_ext(f["title"])]
    if not audio_files:
        print("No audio files found under the provided Drive folder.")
        return

    manifest_df = read_manifest_df(api, HF_REPO_ID, HF_TOKEN)
    known_sha = set(manifest_df["sha256"].astype(str)) if not manifest_df.empty else set()
    dev_list = read_list(api, HF_REPO_ID, HF_TOKEN, DEV_LIST)

    known_path_by_sha = {}
    if not manifest_df.empty:
        for _, r in manifest_df.iterrows():
            known_path_by_sha[str(r["sha256"])] = r["path"]

    ops: List[CommitOperationAdd] = []
    newly_added_paths: List[str] = []
    manifest_rows: List[List] = []

    for f in tqdm(audio_files, desc="Processing Drive files"):
        file_id = f["id"]
        name = f["title"]
        # Download bytes
        gfile = drive.CreateFile({"id": file_id})
        data = io.BytesIO()
        gfile.GetContentFile("tmp_download")  # to disk then read; Drive streaming can be flaky for big files
        blob = Path("tmp_download").read_bytes()
        Path("tmp_download").unlink(missing_ok=True)

        sha256 = compute_sha256_bytes(blob)
        if sha256 in known_sha:
            # Already present: skip upload, but could append to dev list later
            continue

        duration_s, sr, ch, bitrate = ffprobe_audio_bytes(blob)
        size_bytes = len(blob)
        relpath = target_relpath(sha256, name)

        # Stage file and metadata ops
        ops.append(CommitOperationAdd(path_in_repo=relpath, path_or_fileobj=io.BytesIO(blob)))
        manifest_rows.append([
            relpath, sha256, duration_s, sr, ch, bitrate,
            size_bytes, "drive", file_id, ts_iso()
        ])
        newly_added_paths.append(relpath)

        # --- REINDEX: reconstruir manifest completo si se pide o si está vacío ---
    reindex_rows: List[List] = []
    if FORCE_REINDEX or manifest_df.empty:
        print("Reindex: rebuilding manifest from Drive candidates...")
        for f in tqdm(audio_files, desc="Reindex ffprobe+sha"):
            file_id = f["id"]
            name = f["title"]
            # descarga a tmp y calcula meta
            gfile = drive.CreateFile({"id": file_id})
            gfile.GetContentFile("tmp_download_reindex")
            blob = Path("tmp_download_reindex").read_bytes()
            Path("tmp_download_reindex").unlink(missing_ok=True)

            sha256 = compute_sha256_bytes(blob)
            duration_s, sr, ch, bitrate = ffprobe_audio_bytes(blob)
            size_bytes = len(blob)
            # reutiliza path existente si ya estaba, si no genera nuevo
            relpath = known_path_by_sha.get(sha256, target_relpath(sha256, name))

            reindex_rows.append([
                relpath, sha256, duration_s, sr, ch, bitrate,
                size_bytes, "drive", file_id, ts_iso()
            ])

        # DataFrame final del manifest: sólo reindex, o merge si además hubo nuevos
        if manifest_rows:
            out_df = pd.concat([
                pd.DataFrame(reindex_rows, columns=[
                    "path","sha256","duration_s","sample_rate","channels","bitrate",
                    "size_bytes","source","drive_file_id","created_at"
                ]),
                pd.DataFrame(manifest_rows, columns=[
                    "path","sha256","duration_s","sample_rate","channels","bitrate",
                    "size_bytes","source","drive_file_id","created_at"
                ])
            ], ignore_index=True)
            # quita duplicados por sha (prioriza la primera aparición = reindex)
            out_df = out_df.sort_values("duration_s").drop_duplicates(subset=["sha256"], keep="first")
        else:
            out_df = pd.DataFrame(reindex_rows, columns=[
                "path","sha256","duration_s","sample_rate","channels","bitrate",
                "size_bytes","source","drive_file_id","created_at"
            ])

    # escribir manifest.csv a ops SIEMPRE en reindex
    if 'out_df' in locals():
        buf = io.StringIO(); out_df.to_csv(buf, index=False)
        ops.append(CommitOperationAdd(
            path_in_repo=MANIFEST_CSV,
            path_or_fileobj=io.BytesIO(buf.getvalue().encode("utf-8"))
        ))

    df_for_split = out_df if 'out_df' in locals() else manifest_df

    if DEV_LIST_SIZE > 0:
        if not df_for_split.empty:
            tmp = df_for_split.copy()
            tmp["duration_s"] = pd.to_numeric(tmp["duration_s"], errors="coerce").fillna(0.0)
            dev_paths = tmp.sort_values("duration_s").head(DEV_LIST_SIZE)["path"].tolist()
            dev_text = "\n".join(dev_paths) + ("\n" if dev_paths else "")
        else:
            dev_text = ""  # no hay datos, split vacío

        ops.append(CommitOperationAdd(
            path_in_repo=DEV_LIST,
            path_or_fileobj=io.BytesIO(dev_text.encode("utf-8"))
        ))
    # Prepare manifest.csv update
    if manifest_rows and 'out_df' not in locals():
        # Merge with existing
        out_df = pd.concat([
            manifest_df,
            pd.DataFrame(manifest_rows, columns=[
                "path","sha256","duration_s","sample_rate","channels","bitrate",
                "size_bytes","source","drive_file_id","created_at"
            ])
        ], ignore_index=True)

        # Ensure folders placeholders exist by committing empty files (HF auto-creates paths, but keep manifest dir)
        # Write CSV to bytes
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        ops.append(CommitOperationAdd(path_in_repo=MANIFEST_CSV, path_or_fileobj=io.BytesIO(buf.getvalue().encode("utf-8"))))

    # Update dev.list up to DEV_LIST_SIZE, keep it stable once filled
    if dev_list is None:
        dev_list = []
    need = max(0, DEV_LIST_SIZE - len(dev_list))
    add_for_dev = []
    if need > 0:
        # Prefer short clips first for fast iteration
        candidates = []
        if manifest_rows:
            for r in manifest_rows:
                rp, _, dur, *_ = r
                d = dur if dur is not None else 60.0
                candidates.append((d, rp))
            candidates.sort(key=lambda x: x[0])  # shortest first
            for _, rp in candidates:
                if rp not in dev_list and len(add_for_dev) < need:
                    add_for_dev.append(rp)

    if add_for_dev:
        dev_text = "\n".join(dev_list + add_for_dev) + "\n"
        ops.append(CommitOperationAdd(path_in_repo=DEV_LIST, path_or_fileobj=io.BytesIO(dev_text.encode("utf-8"))))

    # Create folder placeholders if empty repo: add empty .gitkeep files
    if not ops:
        if FORCE_COMMIT:
            stamp = ts_iso().encode("utf-8")
            ops.append(CommitOperationAdd(path_in_repo=".last_refresh", path_or_fileobj=io.BytesIO(stamp)))
        else:
            print("Nothing to upload. Manifest and dev.list unchanged.")
            return


    # Commit all operations atomically
    msg_parts = []
    if newly_added_paths:
        msg_parts.append(f"Add {len(newly_added_paths)} audio files")
    if 'out_df' in locals() or manifest_rows:
        msg_parts.append("Update manifest.csv")
    if add_for_dev:
        msg_parts.append(f"Update dev.list (+{len(add_for_dev)})")
    commit_msg = "; ".join(msg_parts) if msg_parts else "No-op update"
    write_commit(api, HF_REPO_ID, HF_TOKEN, ops, commit_msg)

    print("Done.")
    if newly_added_paths:
        print(f"New files: {len(newly_added_paths)}")
    if add_for_dev:
        print(f"dev.list added: {len(add_for_dev)}")

if __name__ == "__main__":
    main()
