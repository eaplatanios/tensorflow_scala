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
#ifndef TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_OPS_COMMON_FLAGS_H_
#define TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_OPS_COMMON_FLAGS_H_

namespace tensorflow {
namespace legacy_flags {

// Flags common to the _Xla* ops and their kernels.
struct XlaOpsCommonFlags {
  // If true, _XlaCompile always refuses to compile the cluster, which means the
  // XLA clusters always run in the TF executor.  Defaults to false.
  bool tf_xla_always_defer_compilation;
};

// Parses the flags in XlaOpsCommonFlags from the TF_XLA_FLAGS environment
// variable and returns a reference to the parsed copy.  Parses TF_XLA_FLAGS
// only the first time this routine is called.
const XlaOpsCommonFlags& GetXlaOpsCommonFlags();

}  // namespace legacy_flags
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_OPS_COMMON_FLAGS_H_
