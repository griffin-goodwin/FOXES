# Use NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based, no TOS acceptance required)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Make conda available in PATH and set conda to non-interactive mode
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ALWAYS_YES=true

# Verify conda installation
RUN $CONDA_DIR/bin/conda --version && \
    ls -la $CONDA_DIR/etc/profile.d/conda.sh

# Set bash as default shell for better conda support
SHELL ["/bin/bash", "-c"]

# Configure conda (Miniforge already uses conda-forge, no TOS acceptance needed)
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda config --set auto_activate_base false && \
    conda config --set auto_update_conda false

# Set working directory
WORKDIR /workspace/FOXES

# Copy requirements.txt first for better Docker layer caching
COPY requirements.txt /workspace/FOXES/requirements.txt

# Create conda environment (Miniforge uses conda-forge by default, no TOS required)
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda create -n foxes python=3.11 -y

# Install requirements with retry logic for large packages
# Use a retry loop to handle network timeouts on large downloads (torch ~900MB, cublas ~600MB)
# Clean pip cache and conda cache immediately after installation to save space
RUN set +e && \
    for i in 1 2 3 4 5; do \
        echo "Installation attempt $i of 5..." && \
        $CONDA_DIR/bin/conda run -n foxes pip install --no-cache-dir \
            --default-timeout=10000 \
            -r requirements.txt && \
        echo "Installation successful! Cleaning up..." && \
        $CONDA_DIR/bin/conda run -n foxes pip cache purge && \
        $CONDA_DIR/bin/conda clean -afy && \
        rm -rf /tmp/* /var/tmp/* && \
        exit 0; \
        echo "Attempt $i failed, waiting 10 seconds before retry..." && sleep 10; \
    done && \
    set -e && \
    echo "ERROR: Installation failed after 5 attempts" && exit 1

# Set up conda environment activation in bashrc and profile
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate foxes" >> ~/.bashrc && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile && \
    echo "conda activate foxes" >> ~/.profile

# Note: The FOXES project directory is mounted as a volume at runtime
# No need to copy it into the image - it will be mounted via -v $(pwd):/workspace/FOXES

# Final cleanup to reduce image size
RUN rm -rf /tmp/* /var/tmp/* && \
    find $CONDA_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find $CONDA_DIR -type f -name "*.pyc" -delete 2>/dev/null || true

# Set the default command to keep container running
# The docker-compose.yml will override this with conda run
CMD ["tail", "-f", "/dev/null"]

