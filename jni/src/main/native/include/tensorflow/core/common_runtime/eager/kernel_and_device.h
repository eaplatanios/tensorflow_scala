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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_

// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// KernelAndDevice encapsulates an instantiated kernel and the device it is on.
//
// Also see:
// https://www.tensorflow.org/code/tensorflow/core/common_runtime/kernel_benchmark_testlib.h
// and
// https://www.tensorflow.org/code/tensorflow/core/kernels/ops_testutil.h
class KernelAndDevice {
 public:
  // Populates 'out' with a kernel appropriate for 'ndef'.
  //
  // The provided FunctionLibraryRuntime MUST outlive all calls to
  // Run() on the returned KernelAndDevice.
  //
  // TODO(ashankar): Figure out thread-safety concerns around
  // FunctionLibraryRuntime (in particular, how the underlying
  // FunctionLibraryDefinition might be mutated by another thread as new
  // functions are registered with it).  Conservatively, thread-safe usage of
  // the FunctionLibraryRuntime is pushed on to the caller (see locking in
  // c_api.cc).
  static Status Init(const NodeDef& ndef, FunctionLibraryRuntime* flib,
                     KernelAndDevice* out);
  // TODO(ashankar): Remove this
  static Status InitOp(Device* device, const NodeDef& ndef,
                       KernelAndDevice* out);

  KernelAndDevice(tensorflow::Rendezvous* rendez)
      : device_(nullptr), flib_(nullptr), rendez_(rendez) {}

  // TODO(ashankar): Handle list-valued inputs.
  Status Run(std::vector<Tensor>* inputs, std::vector<Tensor>* outputs,
             NodeExecStats* stats);

  const OpKernel* kernel() const { return kernel_.get(); }

  Device* device() const { return device_; }

  DataTypeVector* mutable_output_dtypes() { return &output_dtypes_; }
  const DataTypeVector& output_dtypes() { return output_dtypes_; }

 private:
  std::unique_ptr<OpKernel> kernel_;
  Device* device_;
  FunctionLibraryRuntime* flib_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;
  Rendezvous* rendez_;
  DataTypeVector output_dtypes_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
