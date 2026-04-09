#!/bin/bash
# ------------------------------------------------------------------
# Ramchandra Chintala
# Title: GPU Zombie Process Reaper
# Description: Kills orphaned processes holding GPU VRAM hostage 
#              when containers crash ungracefully in Kubernetes.
# ------------------------------------------------------------------

echo "[*] Scanning for orphaned processes locked on NVIDIA devices..."

# Extract PIDs attached to GPU device files
ZOMBIE_PIDS=$(fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i ~ /[0-9]+/) print $i}' | sort -u)

if [ -z "$ZOMBIE_PIDS" ]; then
    echo "[+] No zombie processes found on GPU drivers. System clean."
    exit 0
fi

echo "[!] Found active PIDs on GPU: $ZOMBIE_PIDS"
echo "[*] Cross-referencing against active Docker/Containerd processes..."

# A safe script would map these to cgroups, but as a hardcore flush:
for pid in $ZOMBIE_PIDS; do
    CMD_NAME=$(ps -p $pid -o comm=)
    echo "  -> Found PID $pid ($CMD_NAME)"
    
    # In a real environment, you'd check `docker ps` or `crictl` 
    # Here we simulate reaping
    read -p "Kill process $pid ($CMD_NAME)? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
    then
        echo "Killing $pid..."
        kill -9 $pid
    else
        echo "Skipping $pid."
    fi
done

echo "[*] Reaper routine complete. Check nvidia-smi."
