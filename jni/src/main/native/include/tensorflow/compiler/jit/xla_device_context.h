/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_

#include <memory>

#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device. The allocator
// ignores the alignment and size of the request and always returns a new,
// empty, XlaTensor.
class XlaDeviceAllocator : public Allocator {
 public:
  XlaDeviceAllocator();
  ~XlaDeviceAllocator() override;

  string Name() override;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  void GetStats(AllocatorStats* stats) override;
};

// Helper class for managing data transfers between host and XLA devices.
class XlaTransferManager {
 public:
  explicit XlaTransferManager(perftools::gputools::Stream* stream,
                              xla::LocalClient* client,
                              bool transfer_as_literal);

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done) const;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done);
  perftools::gputools::Stream* stream() const { return stream_; }

 private:
  Status TransferLiteralToDevice(const Tensor& host_tensor,
                                 Tensor* device_tensor) const;
  Status TransferLiteralFromDevice(Tensor* host_tensor,
                                   const Tensor& device_tensor) const;

  // Stream obtained from a Device, used to transfer tensors between
  // CPU and device.
  perftools::gputools::Stream* stream_;
  // For the underlying memory allocator and XLA's TransferManager.
  xla::LocalClient* client_;
  // Transfer manager, for marshalling data to and from the device.
  xla::TransferManager* transfer_manager_;
  // True if we must use XLA's TransferManager for correct device transfers.
  bool transfer_as_literal_;
};

// DeviceContext for operators assigned to XlaDevice devices. The
// implementation must inherit from DeviceContext but otherwise just
// wraps the methods in XlaTransferManager.
class XlaDeviceContext : public DeviceContext {
 public:
  explicit XlaDeviceContext(perftools::gputools::Stream* stream,
                            xla::LocalClient* client, bool transfer_as_literal);

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;
  perftools::gputools::Stream* stream() const override {
    return manager_.stream();
  }

 private:
  XlaTransferManager manager_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_
