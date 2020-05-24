SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONTEXT_PATH="$(realpath ${SCRIPT_DIR}/../../)"

DOCKER_IMAGE="eaplatanios/tensorflow_scala:linux-cpu-x86_64-0.5.0"
DOCKER_FILE=".circleci/images/Dockerfile.linux-cpu-x86_64"

docker build \
  -t "${DOCKER_IMAGE}" \
  -f "${DOCKER_CONTEXT_PATH}/${DOCKER_FILE}" \
  "${DOCKER_CONTEXT_PATH}"
