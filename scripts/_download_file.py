"""Download a remote file via SFTP with binary-safe chunked transfer."""
from __future__ import annotations

import os
import sys
import time

import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]


def main() -> int:
    remote = sys.argv[1]
    local = sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()
    size = sftp.stat(remote).st_size
    print(f"Downloading {remote} ({size / 1e9:.2f} GB) -> {local}")

    chunk = 4 * 1024 * 1024
    done = os.path.getsize(local) if os.path.isfile(local) else 0
    mode = "ab" if done > 0 else "wb"
    if done >= size:
        print("Already complete.")
        sftp.close()
        client.close()
        return 0
    if done > 0:
        print(f"Resuming from {done / 1e9:.2f} GB ({100.0 * done / size:.1f}%)")

    t0 = time.time()
    base_done = done
    with sftp.open(remote, "rb") as rf, open(local, mode) as lf:
        if done > 0:
            rf.seek(done)
        while done < size:
            buf = rf.read(min(chunk, size - done))
            if not buf:
                break
            lf.write(buf)
            done += len(buf)
            if done == size or done % (64 * 1024 * 1024) < chunk:
                elapsed = max(time.time() - t0, 0.1)
                rate = (done - base_done) / elapsed / 1e6
                pct = 100.0 * done / size
                print(
                    f"  {pct:5.1f}%  {done/1e9:.2f}/{size/1e9:.2f} GB  {rate:.1f} MB/s",
                    flush=True,
                )

    sftp.close()
    client.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
