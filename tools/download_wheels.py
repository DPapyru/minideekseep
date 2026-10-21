"""Resume large locked wheels in verified HTTP ranges on unreliable connections."""

import argparse
import hashlib
import os
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def download(name, destination, workers=4):
    lock = tomllib.loads(Path("uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == name)
    wheel = next(w for w in package["wheels"] if "x86_64" in w["url"])
    filename = Path(urlparse(wheel["url"]).path).name
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    expected = wheel["hash"].removeprefix("sha256:")
    if target.exists() and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == expected:
        print(f"Already verified {name}", flush=True)
        return
    parts = destination / (filename + ".parts")
    parts.mkdir(exist_ok=True)
    size = wheel["size"]
    chunk = 2 * 1024**2
    count = (size + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(10):
            try:
                req = Request(wheel["url"], headers={"Range": f"bytes={start}-{end}"})
                with urlopen(req, timeout=60) as response:
                    if response.status != 206:
                        raise OSError("Server did not honor byte range")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise OSError(f"Unexpected range: {content_range}")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError("Incomplete download range")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                if index % 20 == 0:
                    print(f"{name}: part {index + 1}/{count}", flush=True)
                return
            except (OSError, HTTPException):
                if attempt == 9:
                    raise
                time.sleep(min(5, attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range(count)))
    temporary = target.with_suffix(".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for index in range(count):
            data = (parts / str(index)).read_bytes()
            digest.update(data)
            handle.write(data)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch for {name}; retained parts for diagnosis")
    temporary.replace(target)
    for index in range(count):
        (parts / str(index)).unlink()
    os.rmdir(parts)
    print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--destination", default="data/wheels")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download(args.package, args.destination, args.workers)
