#!/usr/bin/env bash
# Usage:
#     ./install_deb_packages [--without_cmake]
# Pass --without_cmake to prevent cmake from being installed with apt-get

set -e

if [[ "$1" != "" ]] && [[ "$1" != "--without_cmake" ]]; then
  echo "Unknown argument '$1'"
  exit 1
fi

# Install bootstrap dependencies from ubuntu deb repository.
apt-get update
apt-get install -y --no-install-recommends software-properties-common
apt-get clean
rm -rf /var/lib/apt/lists/*

# Add the OpenJDK repository.
add-apt-repository -y ppa:openjdk-r/ppa

# Install dependencies from ubuntu deb repository.
apt-get update

apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    clang-format-3.9 \
    curl \
    ffmpeg \
    git \
    libcurl4-openssl-dev \
    libtool \
    libssl-dev \
    mlocate \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    pkg-config \
    python3-dev \
    python3-setuptools \
    python3-pip \
    rsync \
    sudo \
    subversion \
    swig \
    unzip \
    wget \
    zip \
    zlib1g-dev

# Populate the database.
updatedb

if [[ "$1" != "--without_cmake" ]]; then
  apt-get install -y --no-install-recommends \
    cmake
fi

# Install ca-certificates, and update the certificate store.
apt-get install -y ca-certificates-java
update-ca-certificates -f

apt-get clean
rm -rf /var/lib/apt/lists/*

# Install the TensorFlow PIP Package
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.1.0

# Install Protobuf.
PROTOBUF_VERSION="3.11.4"
PROTOBUF_URL="https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip"
PROTOBUF_ZIP=$(basename "${PROTOBUF_URL}")
UNZIP_DEST="google-protobuf"
wget "${PROTOBUF_URL}"
unzip "${PROTOBUF_ZIP}" -d "${UNZIP_DEST}"
cp "${UNZIP_DEST}/bin/protoc" /usr/local/bin/
rm -f "${PROTOBUF_ZIP}"
rm -rf "${UNZIP_DEST}"

# Install SBT.
apt-get update
apt-get install -y --no-install-recommends gpg-agent
apt-get install -y --no-install-recommends apt-transport-https
echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
apt-get update
apt-get install -y --no-install-recommends sbt
