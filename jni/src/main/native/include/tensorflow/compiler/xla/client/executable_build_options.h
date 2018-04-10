/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/optional.h"

namespace xla {

// Class containing options for building an LocalExecutable with
// LocalClient::Compile.
class ExecutableBuildOptions {
 public:
  // If set, this is the device to build the computation for. Valid
  // device_ordinal values are: 0 to # of devices - 1. These values are
  // identical to the device ordinal values used by StreamExecutor. The built
  // executable will be executable on any device equivalent to the specified
  // device as determined by Backend::devices_equivalent(). A value of -1
  // indicates this option has not been set.
  ExecutableBuildOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this specifies the layout of the result of the computation. If not
  // set, the service will chose the layout of the result. A Shape is used to
  // store the layout to accommodate tuple result shapes. A value of nullptr
  // indicates the option has not been set.
  ExecutableBuildOptions& set_result_layout(const Shape& shape_with_layout);
  const Shape* result_layout() const;

  // If set, this specifies an allocator that can be used to allocate temporary
  // space on the device during compilation.  For example, the compiler might
  // want to run various algorithms on the device and pick the fastest one -- it
  // might allocate buffers for use by these algorithms using this allocator.
  //
  // This does not need to be the same as the DeviceMemoryAllocator passed when
  // running the executable.
  ExecutableBuildOptions& set_device_allocator(
      DeviceMemoryAllocator* allocator);
  DeviceMemoryAllocator* device_allocator() const;

  // If set, specifies a regexp of HLO graphs to dump (as in DebugOptions).
  ExecutableBuildOptions& set_generate_hlo_graph(string regex);
  const tensorflow::gtl::optional<string>& generate_hlo_graph() const;

  // If set, specifies a dirpath to dump the end-of-optimization-pipeline HLO
  // protobuf to (as in DebugOptions).
  ExecutableBuildOptions& set_dump_optimized_hlo_proto_to(
      tensorflow::StringPiece dirpath);
  const tensorflow::gtl::optional<string>& dump_optimized_hlo_proto_to() const;

  // If set, specifies a dirpath to dump the per-pass-in-pipeline HLO protobufs
  // to (as in DebugOptions).
  ExecutableBuildOptions& set_dump_per_pass_hlo_proto_to(
      tensorflow::StringPiece dirpath);
  const tensorflow::gtl::optional<string>& dump_per_pass_hlo_proto_to() const;

  // If true, specifies that we should record an HLO profile during execution
  // and log it after execution (as in DebugOptions). If nullopt the default is
  // used.
  ExecutableBuildOptions& set_hlo_profile(bool enabled);
  tensorflow::gtl::optional<bool> hlo_profile() const;

  // Returns a string representation of the build options, suitable for
  // debugging.
  string ToString() const;

 private:
  tensorflow::gtl::optional<bool> hlo_profile_;
  int device_ordinal_ = -1;
  Shape result_layout_;
  bool result_layout_set_ = false;
  tensorflow::gtl::optional<string> generate_hlo_graph_;
  tensorflow::gtl::optional<string> dump_optimized_hlo_proto_to_;
  tensorflow::gtl::optional<string> dump_per_pass_hlo_proto_to_;
  DeviceMemoryAllocator* device_allocator_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
