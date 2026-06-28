#!/usr/bin/env python3
import sys
import json
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import tqdm as hf_tqdm

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
    filename = sys.argv[2]

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
