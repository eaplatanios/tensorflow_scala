FROM ubuntu:16.04

LABEL maintainer="Jan Prach <jendap@google.com>"

# Copy and run the install scripts.
COPY .circleci/images/install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_proto3.sh
RUN /install/install_buildifier.sh
RUN /install/install_auditwheel.sh
RUN /install/install_golang.sh
RUN /install/install_mpi.sh
RUN /install/install_sbt.sh

# Set up MPI
ENV TF_NEED_MPI 1

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

COPY . /tensorflow_scala
