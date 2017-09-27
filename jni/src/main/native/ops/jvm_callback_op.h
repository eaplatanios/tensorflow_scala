/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef TENSORFLOW_JVM_CALLBACK_OP_H_
#define TENSORFLOW_JVM_CALLBACK_OP_H_

#include <jni.h>
#include <string>
#include <sstream>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

struct TF_Status {
  tensorflow::Status status;
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

// A call to the registered JVM function.
struct JVMCall {
  JNIEnv* env;
  jclass registry;
  jmethodID call_method_id;

  // Passed to the JVM to call the function registered with this ID.
  int id;

  // Inputs and outputs of this function invocation.
  std::vector<tensorflow::Tensor> inputs;
  std::vector<tensorflow::Tensor> outputs;
};

#endif  // TENSORFLOW_JVM_CALLBACK_OP_H_
