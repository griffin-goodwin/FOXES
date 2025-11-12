# Quick Start - FOXES Docker Container

## Build and Run

```bash
# Build the container
docker build -t foxes:latest .

# Start the container (runs in background)
docker run -d \
  --name FOXES \
  --gpus all \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v $(pwd):/workspace/FOXES \
  -v /data/FOXES_Data:/data/FOXES_Data \
  --shm-size=8gb \
  --restart unless-stopped \
  -it foxes:latest

# Check container status
docker ps | grep FOXES

# Access the container shell
docker exec -it FOXES bash

# Inside the container, activate conda environment
conda activate foxes
```

## Verify GPU Access

```bash
# Check NVIDIA drivers in container
docker exec -it FOXES nvidia-smi

# Verify PyTorch can see GPU
docker exec -it FOXES conda run -n foxes python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run Python Scripts

```bash
# Run a script in the conda environment
docker exec -it FOXES conda run -n foxes python your_script.py

# Or enter container and run interactively
docker exec -it FOXES bash
conda activate foxes
python your_script.py
```

## Using tmux (Terminal Multiplexer)

tmux is pre-installed in the container for managing multiple terminal sessions:

```bash
# Start a new tmux session inside the container
docker exec -it FOXES tmux new -s foxes

# Attach to an existing tmux session
docker exec -it FOXES tmux attach -t foxes

# List all tmux sessions
docker exec -it FOXES tmux ls

# Detach from tmux: Press Ctrl+B, then D
# Create new window: Press Ctrl+B, then C
# Switch windows: Press Ctrl+B, then N (next) or P (previous)
```

## Stop and Start

```bash
# Stop container
docker stop FOXES

# Start container
docker start FOXES

# Restart container
docker restart FOXES
```

## Useful Commands

```bash
# View logs
docker logs FOXES

# Follow logs in real-time
docker logs -f FOXES

# Remove container (keeps image)
docker stop FOXES && docker rm FOXES

# Remove container and image
docker stop FOXES && docker rm FOXES && docker rmi foxes:latest

# Rebuild after changes
docker build --no-cache -t foxes:latest .
```

