"""Voice model files from the private bucket, cached on the worker by content hash.

A voice model (.pth, ~55 MB) and its index (.index, ~300 MB) are fetched once per worker (or once per network volume, when the
endpoint has one mounted at /runpod-volume) and kept as <sha256><suffix>. With the hash in the request, a cached file is used
without touching the bucket, and a download whose hash differs is thrown away and refused. Without it, the object's ETag and size
name an alias file that remembers the hash of what was downloaded. Only the newest VOICE_CACHE_FILES files are kept."""
import hashlib
import os
import uuid
from pathlib import Path


def cache_dir():
    wanted = os.environ.get("VOICE_CACHE_DIR")
    if wanted:
        d = Path(wanted)
    elif os.path.isdir("/runpod-volume") and os.access("/runpod-volume", os.W_OK):
        d = Path("/runpod-volume/voice-cache")
    else:
        d = Path("/tmp/voice-cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prune(d, keep):
    files = sorted((p for p in d.iterdir() if p.is_file() and p.suffix in (".pth", ".index")), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass
    for alias in d.glob("alias-*.txt"):
        try:
            name = alias.read_text().strip()
            if not (d / name).exists():
                alias.unlink()
        except OSError:
            pass


def fetch(client, bucket, key, suffix, expected_sha=None):
    """-> (local path, {"sha256", "cached", "bytes"}). Raises ValueError (plain words) on a hash mismatch or a missing file."""
    d = cache_dir()
    if expected_sha:
        hit = d / f"{expected_sha}{suffix}"
        if hit.exists():
            os.utime(hit)
            return str(hit), {"sha256": expected_sha, "cached": True, "bytes": hit.stat().st_size}
        alias = None
    else:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            if type(error).__name__ in ("NoSuchKey", "ClientError"):
                raise ValueError("The voice model is not in storage. Upload it again.") from None
            raise
        tag = hashlib.sha256(f"{key}|{head.get('ETag', '')}|{head.get('ContentLength', '')}".encode()).hexdigest()[:32]
        alias = d / f"alias-{tag}.txt"
        try:
            name = alias.read_text().strip()
            if (d / name).exists():
                os.utime(d / name)
                return str(d / name), {"sha256": name[:64], "cached": True, "bytes": (d / name).stat().st_size}
        except OSError:
            pass
    part = d / f"tmp-{uuid.uuid4().hex}.part"
    try:
        try:
            client.download_file(bucket, key, str(part))
        except Exception as error:
            if type(error).__name__ in ("NoSuchKey", "ClientError"):
                raise ValueError("The voice model is not in storage. Upload it again.") from None
            raise
        got = sha256_file(part)
        if expected_sha and got != expected_sha:
            raise ValueError("The voice model in storage does not match its record, so it was not used. Upload it again.")
        final = d / f"{got}{suffix}"
        os.replace(part, final)
        if alias is not None:
            alias.write_text(final.name)
        _prune(d, int(os.environ.get("VOICE_CACHE_FILES", "8")))
        return str(final), {"sha256": got, "cached": False, "bytes": final.stat().st_size}
    finally:
        if part.exists():
            part.unlink()
