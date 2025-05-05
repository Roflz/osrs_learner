import os
import subprocess

def run_split(src_dir: str, split_pct: int) -> int:
    """
    Calls split_real.py with the given src_dir and split_pct.
    Returns the exit code.
    """
    script = os.path.abspath(os.path.join(os.getcwd(), 'training', 'split_real.py'))
    cmd = ['python', script, '--src_dir', src_dir, '--split_pct', str(split_pct)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd()
    )
    # Stream output to console (and GUI)
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    return proc.returncode
