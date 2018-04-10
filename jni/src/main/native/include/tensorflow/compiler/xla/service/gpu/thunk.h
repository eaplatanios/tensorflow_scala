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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the
// metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction.
//
// Thunk provides the Initialize and ExecuteOnStream interface for GpuExecutable
// to initialize and execute the invocation respectively. Its subclasses are
// supposed to override these interfaces to launch a generated kernel or call an
// external library function (such as operations in cuBLAS).
//
// This is thread-compatible.
class Thunk {
 public:
  enum class Kind {
    kConditional,
    kConvolution,
    kCopy,
    kCudnnBatchNormBackward,
    kCudnnBatchNormForwardInference,
    kCudnnBatchNormForwardTraining,
    kFft,
    kGemm,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kSequential,
    kTuple,
    kWhile,
  };

  // The hlo_instruction argument is meant to be the instruction this thunk was
  // generated from, but Thunk never uses this argument other than to save it
  // to Thunk::hlo_instruction, so it can be null.
  explicit Thunk(Kind kind, const HloInstruction* hlo_instruction)
      : kind_(kind), hlo_instruction_(hlo_instruction) {}
  virtual ~Thunk() {}
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  Kind kind() const { return kind_; }
  const HloInstruction* hlo_instruction() const { return hlo_instruction_; }

  // Prepares for executing the thunk. This method is called only once over
  // Thunk's lifetime. For example, KernelThunk::Initialize loads the PTX of a
  // kernel, which is the same in every execution.
  virtual tensorflow::Status Initialize(const GpuExecutable& executable) {
    return tensorflow::Status::OK();
  }

  // Users of Thunk should call ShouldHaltAllActivityBeforeRunning(stream)
  // before calling ExecuteOnStream(stream).  If it returns true, it's the
  // user's responsibility to wait for all activity on the GPU to finish before
  // calling ExecuteOnStream.
  //
  // This value is not required to be constant for a given Thunk.  For example,
  // a Thunk that performs autotuning may return true for its first run and
  // false thereafter.
  virtual bool ShouldHaltAllActivityBeforeRunning(
      perftools::gputools::Stream* /*stream*/) {
    return false;
  }

  // Indicates whether thunks scheduled after this one should wait for this one
  // to complete before running. For example, a convolution thunk creates a
  // scratch allocator, then kicks off a convolution in cudnn via the stream
  // executor. When the stream executor call returns, the scratch allocator goes
  // out of scope, and the scratch memory is deallocated. In this case, the
  // convolution thunk needs to return true so that future thunks wait for the
  // convolution thunk to avoid reusing the deallocated memory until the
  // convolution thunk is done with it.
  virtual bool ShouldBlockFutureThunks() { return false; }

  // Execute the kernel for the thunk on the given stream. This method must be
  // called after Initialize and can be called multiple times over Thunk's
  // lifetime. Stream argument must be non-null.
  virtual tensorflow::Status ExecuteOnStream(
      const BufferAllocations& buffer_allocations,
      perftools::gputools::Stream* stream) = 0;

 private:
  Kind kind_;
  const HloInstruction* hlo_instruction_;
};

// A sequence of thunks.
using ThunkSequence = std::vector<std::unique_ptr<Thunk>>;

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
