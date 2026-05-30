"""Upload a single yaml file to the remote configs/ dir via SFTP."""
import os, sys
import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

local = sys.argv[1]
remote = sys.argv[2]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD)
sftp = client.open_sftp()
try:
    sftp.remove(remote)
except IOError:
    pass
sftp.put(local, remote)
size = sftp.stat(remote).st_size
print(f"OK uploaded {local} -> {remote} ({size} bytes)")
sftp.close()
client.close()
