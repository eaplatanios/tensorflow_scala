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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/pool.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace Eigen {
struct ThreadPoolDevice;
}

namespace xla {

// Options to configure the backend when it is created.
class BackendOptions {
 public:
  // Set the platform backing the backend, or nullptr for the default platform.
  BackendOptions& set_platform(perftools::gputools::Platform* platform);
  perftools::gputools::Platform* platform() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  // The default value of -1 will result in initializing the thread pool with
  // the number of threads equal to the number of cores in the system.
  BackendOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

 private:
  perftools::gputools::Platform* platform_ = nullptr;
  int intra_op_parallelism_threads_ = -1;
};

// Class which encapsulates an XLA backend. It includes everything necessary
// to compile and execute computations on a particular platform.
//
// It also offers a pooling API for creation/use of initialized streams:
//
//    StreamPtr stream = backend->BorrowStream().ConsumeValueOrDie();
class Backend {
 public:
  using StreamPtr = Pool<perftools::gputools::Stream>::SmartPtr;

  // Creates a new backend.
  static StatusOr<std::unique_ptr<Backend>> CreateBackend(
      const BackendOptions& options);

  // Creates a backend for the default platform. The default platform is defined
  // in PlatformUtil.
  static StatusOr<std::unique_ptr<Backend>> CreateDefaultBackend();

  ~Backend();

  // Accessors for the various objects.
  perftools::gputools::Platform* platform() const { return platform_; }
  Compiler* compiler() const { return compiler_; }
  DeviceMemoryAllocator* memory_allocator() const {
    return memory_allocator_.get();
  }
  TransferManager* transfer_manager() const { return transfer_manager_; }
  ComputationPlacer* computation_placer() const { return computation_placer_; }

  // Returns the number of devices of the platform type which are visible. Not
  // all of these devices may be usable by XLA.
  int device_count() const { return stream_executors_.size(); }

  // Returns the device ordinal number of the default device.
  int default_device_ordinal() const;

  // Returns stream executors of all supported devices for this backend. The
  // executors are ordered by the device ordinal.
  const std::vector<perftools::gputools::StreamExecutor*>& stream_executors()
      const {
    return stream_executors_;
  }

  // Returns the stream executor for the given device ordinal.
  StatusOr<perftools::gputools::StreamExecutor*> stream_executor(
      int device_ordinal) const;

  // Returns the stream executor for the default device ordinal. This stream
  // executor can only be used when the number of computations is 1 (replication
  // can be > 1).
  perftools::gputools::StreamExecutor* default_stream_executor() const {
    CHECK(!stream_executors_.empty());
    return stream_executors_[0];
  }

  // Borrows a stream for use by the caller, either by grabbing it from an
  // internal pool, or by constructing/initializating it, and returns the result
  // to the caller.
  StatusOr<StreamPtr> BorrowStream(int device_ordinal);
  StatusOr<StreamPtr> BorrowStream(
      perftools::gputools::StreamExecutor* executor);

  // Returns a function to borrow a stream, as `BorrowStream` above does.
  // Purely for convenience, the caller could rather make this anonymous
  // function itself.
  std::function<StatusOr<StreamPtr>(int)> StreamBorrower() {
    return [this](int device_ordinal) { return BorrowStream(device_ordinal); };
  }

  // Returns whether the given device ordinal of the backend is supported.
  bool device_ordinal_supported(int device_ordinal) const {
    return (device_ordinal >= 0 && device_ordinal < device_count() &&
            stream_executors_[device_ordinal] != nullptr);
  }

  // Return a string identifier for the given device, eg: "GPU:3".
  string device_name(int device_ordinal) const {
    return tensorflow::strings::StrCat(platform_->Name(), ":", device_ordinal);
  }

  // Returns true if the devices with the given ordinals are equivalent from
  // XLA's perspective. That is, an executable compiled for one device would
  // be equivalent to an executable compiled for the other.
  StatusOr<bool> devices_equivalent(int device_ordinal_a, int device_ordinal_b);

  // For the host platform, returns the threadpool to use when scheduling
  // parallel operators. For other platforms, returns NULL.
  tensorflow::thread::ThreadPool* inter_op_thread_pool() const;

  // For the host platform, returns the configured eigen threadpool device to be
  // used for scheduling work. For other platforms, returns NULL.
  const Eigen::ThreadPoolDevice* eigen_intra_op_thread_pool_device() const;
  tensorflow::thread::ThreadPool* eigen_intra_op_thread_pool() const;

  // Resets the devices associated with this backend.
  Status ResetDevices();

 private:
  struct EigenThreadPoolWrapper;
  Backend(perftools::gputools::Platform* platform, Compiler* compiler,
          tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
              stream_executors,
          TransferManager* transfer_manager,
          ComputationPlacer* computation_placer,
          int intra_op_parallelism_threads);
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  perftools::gputools::Platform* platform_;
  Compiler* compiler_;
  TransferManager* transfer_manager_;
  ComputationPlacer* computation_placer_;

  // Vector of stream executors. stream_executors_[0] is the default executor.
  std::vector<perftools::gputools::StreamExecutor*> stream_executors_;

  tensorflow::mutex mu_;

  // Mapping from stream executor to stream pools, used by `BorrowStream` above.
  std::map<perftools::gputools::StreamExecutor*,
           Pool<perftools::gputools::Stream>>
      stream_pools_ GUARDED_BY(mu_);

  // The default memory allocator to use.
  std::unique_ptr<StreamExecutorMemoryAllocator> memory_allocator_;

  // For the CPU backend, a threadpool for scheduling parallel operators.
  std::unique_ptr<tensorflow::thread::ThreadPool> inter_op_thread_pool_;

  // For the CPU backend, an Eigen threadpool device for use by Eigen code.
  std::unique_ptr<EigenThreadPoolWrapper> intra_op_thread_pool_wrapper_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_
