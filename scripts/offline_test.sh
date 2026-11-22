#!/bin/bash
# Offline proof: run the pipeline inside an EMPTY network namespace (only a loopback interface exists),
# so no network access is possible, without disconnecting your SSH session.
# Run as:  sudo unshare --net bash scripts/offline_test.sh 2>&1 | tee run_offline_1.log
# NOTE: this is the script exactly as it was run on the Pi; edit the user name and path below for your setup.
echo "--- interfaces visible in this isolated environment:"
ip -br addr
echo "--- trying to reach the internet (must fail):"
python3 - << 'PY'
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    print("RESULT: NETWORK REACHABLE (offline test invalid)")
except OSError as e:
    print("RESULT: no network, as intended:", e)
PY
echo "--- running the pipeline:"
su armsocproject -s /bin/bash -c 'cd /home/armsocproject/threat_detector && source venv/bin/activate && python pi_pipeline.py'
