FROM registry.access.redhat.com/ubi8/ubi-minimal as base

ARG UID=2000
ARG GID=2000

# we install pip and other python packages in /usr/local
ENV PATH="/usr/local/bin:${PATH}"

RUN microdnf update -y \
    && microdnf install -y --disableplugin=subscription-manager \
        gcc gcc-c++ make.x86_64 openssl-devel bzip2-devel libffi-devel glibc-langpack-en \
        java-1.8.0-openjdk-devel \
        shadow-utils \
        findutils \
        util-linux \
        sqlite-devel \
        python38 python38-setuptools python38-devel \
    && microdnf clean all \
    && pip3 install --upgrade pip
    
RUN microdnf update -y \
    && microdnf install -y --disableplugin=subscription-manager \
        git
        
RUN microdnf update -y \
    && microdnf install -y --disableplugin=subscription-manager \
        wget curl
        
RUN pip install --upgrade ipython

RUN pip install --upgrade nbformat==5.1.3