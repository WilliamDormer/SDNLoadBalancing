FROM ubuntu:18.04

USER root
WORKDIR /root

# Set timezone and avoid interactive prompts during installation
ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY scripts/ENTRYPOINT.sh /

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iproute2 \
    iputils-ping \
    mininet \
    net-tools \
    openvswitch-switch \
    openvswitch-testcontroller \
    tcpdump \
    vim \
    x11-xserver-utils \
    xterm \
    wireshark-qt \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    python3-tk \
    ca-certificates \
    git \
    sudo \
    less \
    nano

# Clean up apt cache
RUN rm -rf /var/lib/apt/lists/* \
    && chmod +x /ENTRYPOINT.sh

# Clone Mininet repository
RUN git clone https://github.com/mininet/mininet.git
COPY utils/miniedit.py /root/mininet/examples/miniedit.py

# Update pip and install ryu using Python 3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
# Install a compatible version of eventlet before installing Ryu
RUN python3 -m pip install eventlet==0.30.2
RUN python3 -m pip install ryu
RUN pip install numpy flask

# Set up OVS controller symlink
RUN ln -sf /usr/bin/ovs-testcontroller /usr/bin/controller

# Expose ports for OpenFlow and OVS
# 6633/6653: OpenFlow controller ports
# 6640: OVS manager port
EXPOSE 6633 6653 6640 9000

# To run with an external controller:
# 1. Run the container with: docker run -p 6633:6633 -p 6653:6653 -p 6640:6640 -e CONTROLLER_IP=<external_ip> -e CONTROLLER_PORT=6653 <image_name>
# 2. Or use host networking: docker run --network=host -e CONTROLLER_IP=<external_ip> -e CONTROLLER_PORT=6653 <image_name>

ENTRYPOINT ["/ENTRYPOINT.sh"]