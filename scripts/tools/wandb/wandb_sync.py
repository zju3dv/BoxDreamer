import os
import sys
import time
import subprocess
from datetime import datetime

# Sync offline wandb logs periodically
period = 60  # 1 min

def sync_wandb(wandb_root: str, proxy = None):
    os.chdir(wandb_root)
    env = os.environ.copy()
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        
    try:
        result = subprocess.run("wandb sync --sync-all", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr

def main():
    wandb_root = "***"
    proxy = None  # Replace with your proxy server if needed
    
    if not os.path.exists(wandb_root):
        print(f"WANDB_ROOT does not exist: {wandb_root}. Exiting...")
        sys.exit(1)
    
    while True:
        stdout, stderr = sync_wandb(wandb_root, proxy)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"wandb sync at {now}.")
        if stdout:
            print("Standard Output:\n", stdout)
        if stderr:
            print("Standard Error:\n", stderr)
        time.sleep(period)

if __name__ == "__main__":
    main()