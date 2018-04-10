/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace perftools {
namespace gputools {
class Stream;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

namespace gpu = ::perftools::gputools;

class GPUDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  GPUDeviceContext(int stream_id, gpu::Stream* stream,
                   gpu::Stream* host_to_device_stream,
                   gpu::Stream* device_to_host_stream,
                   gpu::Stream* device_to_device_stream)
      : stream_id_(stream_id),
        stream_(stream),
        host_to_device_stream_(host_to_device_stream),
        device_to_host_stream_(device_to_host_stream),
        device_to_device_stream_(device_to_device_stream) {}

  ~GPUDeviceContext() override {}

  gpu::Stream* stream() const override { return stream_; }
  gpu::Stream* host_to_device_stream() const { return host_to_device_stream_; }
  gpu::Stream* device_to_host_stream() const { return device_to_host_stream_; }
  gpu::Stream* device_to_device_stream() const {
    return device_to_device_stream_;
  }
  int stream_id() const { return stream_id_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

  void MaintainLifetimeOnStream(
      const Tensor* t, perftools::gputools::Stream* stream) const override {}

 private:
  int stream_id_;
  // The default primary stream to use for this context.
  // All the memory belongs to this stream.
  gpu::Stream* stream_;
  // The stream to use for copy data from host into GPU.
  gpu::Stream* host_to_device_stream_;
  // The stream to use for copy data from GPU to host.
  gpu::Stream* device_to_host_stream_;
  // The stream to use for copy data between GPU.
  gpu::Stream* device_to_device_stream_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
