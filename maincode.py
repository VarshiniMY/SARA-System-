import subprocess

# List of scripts to run
scripts = [ "camera_video.py","mainspeed.py"]

# Run scripts in parallel
processes = []
for script in scripts:
    p = subprocess.Popen(["python", script])
    processes.append(p)

# Wait for all processes to finish
for p in processes:
    p.wait()
