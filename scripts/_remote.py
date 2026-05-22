"""Run a remote command via paramiko, streaming stdout/stderr.

SSH credentials read from env vars: SSH_HOST/SSH_PORT/SSH_USER/SSH_PASSWORD.
"""
import os
import paramiko
import sys

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60

# wrap in bash login shell so PATH and aliases are sane
wrapped = f"bash -lc {repr(cmd)}"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15)

stdin, stdout, stderr = client.exec_command(wrapped, timeout=timeout, get_pty=True)
for line in iter(stdout.readline, ""):
    if not line:
        break
    sys.stdout.write(line)
    sys.stdout.flush()
exit_code = stdout.channel.recv_exit_status()
err = stderr.read().decode(errors="replace")
if err.strip():
    sys.stderr.write(err)
client.close()
sys.exit(exit_code)
