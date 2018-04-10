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

#ifndef TENSORFLOW_KERNELS_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource stored by variables in the resource manager
// (new, resource-style version).
class Var : public ResourceBase {
 public:
  explicit Var(DataType dtype) : tensor_(dtype) {}
  // Not copyable or movable.
  Var(const Var&) = delete;
  Var& operator=(const Var&) = delete;

  // TODO(ebrevdo): Use LockSet instead of exposing mu.
  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

  // Only used in the resource variable path. In resource variables,
  // tensor.IsInitialized() can be true (i.e. have memory allocated to it) while
  // there is not a good value there due to a race condition, and it's possible
  // to stumble upon this during variable.initialized_value(). So it's best to
  // just store directly whether the variable is initialized.
  bool is_initialized = false;  // GUARDED_BY(mu_) but annotalysis doesn't like
                                // it.

 private:
  mutex mu_;
  Tensor tensor_;

  ~Var() override {}
};

class VariableOp : public OpKernel {
 public:
  explicit VariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;

  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  TF_DISALLOW_COPY_AND_ASSIGN(VariableOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_VARIABLE_OPS_H_
