#!/usr/bin/env python3
import sys
import json
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import tqdm as hf_tqdm

def resolve_real_filename(repo_id, requested_filename):
    """If requested_filename matches a real file in the repo exactly, use it.
    Otherwise, treat it as a quant tag (e.g. 'Q4_K_M.gguf') and find the real
    file whose name contains that tag."""
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        sys.stderr.write(f"[WARN] could not list repo files: {e}\n")
        return requested_filename  # fall back to whatever was passed in

    gguf_files = [f for f in files if f.lower().endswith(".gguf")]

    # Exact match - the caller already gave us a real filename.
    if requested_filename in gguf_files:
        return requested_filename

    # Treat requested_filename (minus .gguf) as a quant tag, find a real
    # file whose name contains it case-insensitively.
    tag = requested_filename[:-5] if requested_filename.lower().endswith(".gguf") else requested_filename
    matches = [f for f in gguf_files if tag.upper() in f.upper()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        sys.stderr.write(f"[WARN] multiple files match tag '{tag}': {matches} - using first match\n")
        return matches[0]
    else:
        sys.stderr.write(f"[WARN] no file matching tag '{tag}' found in {repo_id}, files available: {gguf_files}\n")
        return requested_filename  # let it fail downstream with a clear 404
    
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

class JsonProgressTqdm(hf_tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emit()

    def update(self, n=1):
        super().update(n)
        self._emit()

    def display(self, msg=None, pos=None):
        # No-op: suppress tqdm's own bar rendering entirely. We only want
        # our own JSON lines on stdout, not tqdm's text output.
        pass

    def _emit(self):
        emit({"completed": int(self.n or 0), "total": int(self.total or 0)})

if __name__ == "__main__":
    repo_id = sys.argv[1]
    requested_filename = sys.argv[2]
    filename = resolve_real_filename(repo_id, requested_filename)
    if filename != requested_filename:
        sys.stderr.write(f"[INFO] resolved '{requested_filename}' -> '{filename}'\n")

    real_sha256 = None
    real_size = None
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        meta = get_hf_file_metadata(url)
        if meta.etag:
            real_sha256 = meta.etag.strip('"')
        real_size = meta.size
    except Exception as e:
        sys.stderr.write(f"[WARN] metadata lookup failed: {e}\n")

    emit({
        "status": "pulling manifest",
        "digest": f"sha256:{real_sha256}" if real_sha256 else None,
        #"total": real_size,
    })

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        tqdm_class=JsonProgressTqdm,
    )
    emit({"done": True, "path": path, "digest": f"sha256:{real_sha256}" if real_sha256 else None})
