#!/bin/bash

# Set the container name
CONTAINER_NAME="sdn-loadbalancer"
IMAGE_NAME="sdn-loadbalancer-image"

# Detect OS type
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS specific setup
  if ! command -v xquartz &> /dev/null; then
    echo "XQuartz is not installed. Please install it from https://www.xquartz.org/"
    echo "You can also install it with: brew install --cask xquartz"
    exit 1
  fi

  # Check if XQuartz is running
  if ! ps -ef | grep -v grep | grep -q XQuartz; then
    echo "Starting XQuartz..."
    open -a XQuartz
    # Wait for XQuartz to start
    sleep 5
  fi

  # Set up X11 security
  echo "Setting up X11 security..."
  xhost + localhost
  
  # Get the IP of the host machine
  HOST_IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
  if [ -z "$HOST_IP" ]; then
    # Try alternative network interface
    HOST_IP=$(ifconfig en1 | grep inet | awk '$1=="inet" {print $2}')
  fi

  if [ -z "$HOST_IP" ]; then
    echo "Could not determine host IP address. Using host.docker.internal instead."
    export DISPLAY="host.docker.internal:0"
  else
    echo "Host IP: $HOST_IP"
    export DISPLAY="$HOST_IP:0"
    # Also allow connections from this IP
    xhost + $HOST_IP
  fi
else
  # Linux specific setup (Ubuntu, Debian, etc.)
  echo "Linux detected. Setting up X11 forwarding..."
  
  # Check if X server is installed
  if ! command -v xhost &> /dev/null; then
    echo "X server not found. Please install X11 packages."
    echo "For Ubuntu/Debian: sudo apt-get install -y x11-xserver-utils"
    echo "For Fedora/RHEL: sudo dnf install -y xorg-x11-server-utils"
    exit 1
  fi
  
  # Allow local connections to X server
  xhost +local: > /dev/null 2>&1
  
  # Set DISPLAY variable if not already set
  if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "DISPLAY variable set to $DISPLAY"
  else
    echo "Using existing DISPLAY: $DISPLAY"
  fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "Docker is not installed. Please install Docker first."
  echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
  exit 1
fi

# Check if the image exists
if ! docker images $IMAGE_NAME | grep -q $IMAGE_NAME; then
  echo "Docker image $IMAGE_NAME not found. Building it..."
  docker build -t $IMAGE_NAME .
else
  echo "Using existing Docker image $IMAGE_NAME"
fi

# Check if container already exists
if docker ps -a | grep -q $CONTAINER_NAME; then
  # Check if container is running
  if docker ps | grep -q $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME is already running. Attaching to it..."
  else
    echo "Container $CONTAINER_NAME exists but is not running. Starting it..."
    docker start $CONTAINER_NAME
  fi
else
  # Run the container
  echo "Creating and starting SDN Load Balancer container with X11 forwarding..."
  docker run -d \
    --name $CONTAINER_NAME \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=${XAUTHORITY:-~/.Xauthority} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${XAUTHORITY:-~/.Xauthority}:/root/.Xauthority:rw \
    -v /lib/modules:/lib/modules \
    -v $(pwd):/root/SDNLoadBalancing \
    -p 6633:6633 \
    -p 6653:6653 \
    -p 6654:6654 \
    -p 6655:6655 \
    -p 6656:6656 \
    -p 6657:6657 \
    -p 6640:6640 \
    -p 9000:9000 \
    --restart unless-stopped \
    -it $IMAGE_NAME
fi

# Attach to the container
echo "Attaching to the container..."
docker attach $CONTAINER_NAME

echo "Container '$CONTAINER_NAME' stopped." 
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME