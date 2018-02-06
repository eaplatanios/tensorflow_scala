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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include "tensorflow/c/c_eager_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/c_eager_api_runtime.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  TFE_ContextDevicePlacementPolicy policy{
      TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32};
};

struct TFE_Context {
  explicit TFE_Context(const TFE_ContextOptions& opts, TF_Session* s)
      : policy(opts.policy),
        session(s),
        rendezvous(new tensorflow::IntraProcessRendezvous(s->device_mgr)),
        pflr(new tensorflow::ProcessFunctionLibraryRuntime(
            session->device_mgr, opts.session_options.options.env,
            TF_GRAPH_DEF_VERSION, &func_lib_def, {})) {}

  const TFE_ContextDevicePlacementPolicy policy;

  // Note: we cannot use C++11 thread_local here as there is no concept of a
  // thread-local-object-local variable in C++11.
  tensorflow::mutex policy_map_mu;
  std::unordered_map<std::thread::id, TFE_ContextDevicePlacementPolicy>
      thread_local_policies GUARDED_BY(policy_map_mu);

  // TFE_Context is an extension of TF_Session. And TF_Session needs a TF_Graph.
  TF_Session* const session;
  tensorflow::Rendezvous* const rendezvous;

  tensorflow::mutex functions_mu;
  tensorflow::FunctionLibraryDefinition func_lib_def GUARDED_BY(functions_mu){
      tensorflow::OpRegistry::Global(), {}};

  // One FunctionLibraryRuntime per device.
  // func_libs[i] is the FunctionLibraryRuntime corresponding to
  // session->devices[i].
  const std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr;

  tensorflow::mutex cache_mu;
  std::unordered_map<tensorflow::Fprint128, tensorflow::KernelAndDevice*,
                     tensorflow::Fprint128Hasher>
      kernel_cache GUARDED_BY(cache_mu);

  tensorflow::FunctionLibraryRuntime* func_lib(tensorflow::Device* d) const {
    return pflr->GetFLR(d->name());
  }

  const std::vector<tensorflow::Device*>& devices() { return session->devices; }

  // Whether we should compute RunMetadata.
  std::atomic<bool> should_store_metadata{false};
  tensorflow::mutex metadata_mu;
  tensorflow::RunMetadata run_metadata GUARDED_BY(metadata_mu);
};

struct TFE_TensorHandle {
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d)
      : t(t), d(d) {}

  tensorflow::Tensor t;
  // TODO(ashankar): d == nullptr iff local CPU
  // This was expedient, but perhaps worth revisiting ('d' should always be a
  // valid pointer?)
  // This can be done if TFE_NewOp() and the TFE_TensorHandle constructors are
  // provided with the appropriate TFE_Context.
  //
  // TODO(ashankar): Reference count TFE_Context to ensure that 'd' of a
  // TFE_TensorHandle does not outlive the TFE_Context from which it came?
  tensorflow::Device* d;
};

struct TFE_Op {
  TFE_Op(TFE_Context* ctx, const char* op, const tensorflow::AttrTypeMap* t)
      : ctx(ctx), name(op), attrs(op), attr_types(t), device(nullptr) {}

  bool const is_function() const { return attr_types == nullptr; }

  TFE_Context* ctx;  // Must outlive the TFE_Op.
  const tensorflow::string name;
  tensorflow::AttrBuilder attrs;
  const tensorflow::AttrTypeMap* attr_types;
  std::vector<tensorflow::Tensor> inputs;
  std::vector<tensorflow::Device*> input_devices;
  tensorflow::Device* device;
};

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
