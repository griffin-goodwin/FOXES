# Docker Setup for FOXES

This guide explains how to build and run the FOXES Docker container with NVIDIA CUDA support and conda environment.

## Prerequisites

1. **Docker** installed on your system
2. **NVIDIA Docker Runtime** (nvidia-docker2) for GPU support
3. **NVIDIA drivers** installed on the host system

### Verify NVIDIA Docker Support

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this command works, you're ready to proceed.

## Building the Container

```bash
docker build -t foxes:latest .
```

## Running the Container

```bash
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
```

This will:
- Start the container in detached mode
- Mount the current directory to `/workspace/FOXES`
- Mount `/data/FOXES_Data` to `/data/FOXES_Data` in the container
- Enable GPU access
- Keep the container running persistently

## Accessing the Container

### Enter the container shell

```bash
docker exec -it FOXES bash
```

Once inside the container, activate the conda environment:

```bash
conda activate foxes
```

### Run commands in the conda environment

```bash
docker exec -it FOXES conda run -n foxes python your_script.py
```

## Container Management

### Stop the container

```bash
docker stop FOXES
```

### Start the container

```bash
docker start FOXES
```

### View container logs

```bash
docker logs FOXES
```

### Remove the container

```bash
docker stop FOXES && docker rm FOXES
```

### Remove the container and image

```bash
docker stop FOXES && docker rm FOXES && docker rmi foxes:latest
```

## Using the Container

The conda environment `foxes` is pre-configured with all dependencies from `requirements.txt`. To use it:

1. Enter the container: `docker exec -it FOXES bash`
2. Activate conda: `conda activate foxes`
3. Run your Python scripts or notebooks

## Volume Mounts

- `/workspace/FOXES`: Your FOXES project directory (mounted from host)
- `/data/FOXES_Data`: Data directory (mounted from `/data/FOXES_Data` on host)

## GPU Verification

To verify GPU access inside the container:

```bash
docker exec -it FOXES bash -c "conda run -n foxes python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'"
```

## Troubleshooting

### Container won't start
- Check if NVIDIA Docker runtime is installed: `docker info | grep -i runtime`
- Verify NVIDIA drivers: `nvidia-smi`

### CUDA not available in PyTorch
- Ensure the container has GPU access: `docker exec -it FOXES nvidia-smi`
- Check PyTorch installation: `docker exec -it FOXES conda run -n foxes python -c "import torch; print(torch.__version__)"`

### Permission issues
- Ensure your user has Docker permissions
- Check volume mount permissions

### Out of memory
- Increase shared memory: use `--shm-size=16gb` or higher in the docker run command
- Adjust batch sizes in your training scripts

## Notes

- The container is configured to restart automatically unless stopped manually
- All changes to files in `/workspace/FOXES` are reflected on the host
- The conda environment persists across container restarts
- GPU access requires NVIDIA drivers and nvidia-docker runtime on the host

