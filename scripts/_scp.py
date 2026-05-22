"""scp helper using paramiko sftp.

Credentials are read from environment variables; export them in your shell or
in a local .env file (which is gitignored). Required vars:
    SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD
"""
import os
import sys
import stat as _stat
import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

direction = sys.argv[1]  # "put" or "get"
local = sys.argv[2]
remote = sys.argv[3]
recursive = (len(sys.argv) > 4 and sys.argv[4] == "-r")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD)
sftp = client.open_sftp()


def _ensure_remote_dir(path: str) -> None:
    parts = path.replace("\\", "/").strip("/").split("/")
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def put_recursive(local_dir: str, remote_dir: str) -> None:
    _ensure_remote_dir(remote_dir)
    for entry in os.listdir(local_dir):
        lp = os.path.join(local_dir, entry)
        rp = remote_dir.rstrip("/") + "/" + entry
        if os.path.isdir(lp):
            put_recursive(lp, rp)
        else:
            sftp.put(lp, rp)


def get_recursive(remote_dir: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        rp = remote_dir.rstrip("/") + "/" + entry.filename
        lp = os.path.join(local_dir, entry.filename)
        if _stat.S_ISDIR(entry.st_mode):
            get_recursive(rp, lp)
        else:
            sftp.get(rp, lp)


if direction == "put":
    if recursive and os.path.isdir(local):
        put_recursive(local, remote)
    else:
        sftp.put(local, remote)
    print(f"PUT {local} -> {remote}")
elif direction == "get":
    if recursive:
        get_recursive(remote, local)
    else:
        sftp.get(remote, local)
    print(f"GET {remote} -> {local}")

sftp.close()
client.close()
