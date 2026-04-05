import sys, time, psutil, subprocess

def wait_and_run():
    # Find the gemma_eval steer.py process
    target_pid = None
    print("Looking for running steer.py process...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline')
            if cmd and 'python' in proc.info.get('name', '').lower() and 'steer.py' in cmd and 'hunt' in cmd:
                target_pid = proc.pid
                print(f"Found steer.py (PID {target_pid}). Waiting for it to finish...")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    if target_pid:
        while psutil.pid_exists(target_pid):
            time.sleep(10)
        print("steer.py finished. Starting batch...")
        
    print("Running batch command for other models...")
    subprocess.run("call conda activate gemma_eval && set PYTHONUNBUFFERED=1 && python scripts/steer.py batch", shell=True)

if __name__ == "__main__":
    wait_and_run()
