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

#include "jvm_callback_op.h"

#include "exception.h"
#include "utilities.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/c_eager_api.h"
#include "tensorflow/c/c_eager_api_internal.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/kernel_def.pb_text.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace {
// Copy of the C Eager API struct due to the circular dependency issue.
TFE_TensorHandle* TFE_Local_NewTensorHandle(const tensorflow::Tensor& t) {
  return new TFE_TensorHandle(t, nullptr);
}

// Copy of the C Eager API struct due to the circular dependency issue.
const tensorflow::Tensor* TFE_Local_TensorHandleUnderlyingTensorInHostMemory(
    TFE_TensorHandle* h, TF_Status* status) {
  if (h->d != nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "TFE_TensorHandle is placed in device (not host) memory. Cannot return "
            "a tensorflow::Tensor");
    return nullptr;
  }
  return &h->t;
}
}

namespace tensorflow {
REGISTER_OP("JVMCallback")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("id: int")
    .Attr("jvm_pointer: string")
    .Attr("registry_pointer: string")
    .Attr("registry_call_pointer: string")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >=0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Invokes a JVM callback function, `f` to compute `f(input)->output`.

This operation is considered stateful. For a stateless version, see
`JVMCallback`.

id: A unique ID representing a registered JVM callback function
  in this address space.
jvm_pointer: A pointer to an existing JVM instance represented as a
  string. This is the JVM that will be used when invoking this JVM
  callback.
registry_pointer: Pointer to the JVM callbacks registry class.
registry_call_pointer: Pointer to the JVM callbacks registry class
  'call' method.
input: List of tensors that will provide input to the op.
output: Output tensors from the op.
Tin: Data types of the inputs to the op.
Tout: Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
)doc");

REGISTER_OP("JVMCallbackStateless")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("id: int")
    .Attr("jvm_pointer: string")
    .Attr("registry_pointer: string")
    .Attr("registry_call_pointer: string")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A stateless version of `JVMCallback`.
)doc");

namespace {
  // Given the 'call', prepares the inputs as a JNI long array that is appropriate for calling the registry.
  jlongArray MakeInputs(JVMCall* call) {
    unsigned long n = call->inputs.size();
    jlongArray inputs = call->env->NewLongArray(static_cast<jsize>(n));
    jlong* inputs_array = call->env->GetLongArrayElements(inputs, nullptr);
    for (int64 i = 0; i < n; ++i) {
      const Tensor& t = call->inputs[i];
      TFE_TensorHandle* tensor = TFE_Local_NewTensorHandle(t);
      inputs_array[i] = reinterpret_cast<jlong>(tensor);
    }
    call->env->ReleaseLongArrayElements(inputs, inputs_array, 0);
    return inputs;
  }

  // Process the return values by converting them back to TensorFlow tensors and adding them to the call outputs.
  void ProcessOutputs(JVMCall* call, jlongArray call_outputs, TF_Status* status) {
    call->outputs.clear();
    jsize n = call->env->GetArrayLength(call_outputs);
    jlong* outputs_array = call->env->GetLongArrayElements(call_outputs, nullptr);
    for (int i = 0; i < n; ++i) {
      static_assert(sizeof(jlong) >= sizeof(TFE_TensorHandle*), "Cannot package C object pointers as a Java long");
      if (outputs_array[i] == 0) {
        status->status = errors::InvalidArgument("One of the op output tensors has been disposed already.");
        return;
      }
      auto* h = reinterpret_cast<TFE_TensorHandle*>(outputs_array[i]);
      if (h == nullptr) {
        status->status = errors::InvalidArgument("Could not obtain tensor handle to one of the outputs.");
        return;
      }
      const Tensor* t = TFE_Local_TensorHandleUnderlyingTensorInHostMemory(h, status);
      if (!status->status.ok()) return;
      call->outputs.push_back(*t);
    }
    call->env->ReleaseLongArrayElements(call_outputs, outputs_array, 0);
  }

  // Calls the registered JVM function through the registry.
  Status CallJVMFunction(JVMCall* call) {
    // Prepare the call arguments.
    jlongArray call_inputs = MakeInputs(call);

    // Invoke the registry 'call' method.
    if (call->registry != nullptr && call->call_method_id != nullptr) {
      auto outputs = (jlongArray) call->env->CallStaticObjectMethod(
          call->registry, call->call_method_id, call->id, call_inputs);
      jthrowable exc(call->env->ExceptionOccurred());
      if (exc) {
        // Get the exception string representation to use as the error message.
        jclass throwableCls(call->env->FindClass("java/lang/Throwable"));
        jmethodID toString = call->env->GetMethodID(throwableCls, "toString", "()Ljava/lang/String;");
        jstring exc_string = (jstring) call->env->CallObjectMethod(exc, toString);
        const char* c_exc_string = call->env->GetStringUTFChars(exc_string, 0);
        tensorflow::StringPiece tf_exc_string(c_exc_string);
        call->env->ReleaseStringUTFChars(exc_string, c_exc_string);
        // Get the exception class name and convert it to a TensorFlow error code.
        jclass excObjCls(call->env->GetObjectClass(exc));
        jclass classCls(call->env->FindClass("java/lang/Class"));
        jmethodID getName(call->env->GetMethodID(classCls, "getName", "()Ljava/lang/String;"));
        jstring clsName(static_cast<jstring>(call->env->CallObjectMethod(excObjCls, getName)));
        const char* clsNameCString = call->env->GetStringUTFChars(clsName, 0);
        std::string clsNameCppString(clsNameCString);
        int error_code = tf_error_code(clsNameCppString);
        call->env->ReleaseStringUTFChars(clsName, clsNameCString);
        return tensorflow::Status((tensorflow::error::Code) error_code, tf_exc_string);
      }

      if (outputs == nullptr) {
        return errors::Unknown("Failed to run JVM callback function.");
      }

      // Process the return values and convert them back to TensorFlow tensors.
      auto* status = new TF_Status;
      ProcessOutputs(call, outputs, status);
      return status->status;
    } else {
      return errors::Unknown("Failed to run JVM callback function. Could not find registry class or its 'call' method.");
    }
  }
}  // namespace

class JVMCallbackOp : public OpKernel {
public:
  explicit JVMCallbackOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id", &id_));
    std::string jvm_pointer;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("jvm_pointer", &jvm_pointer));
    jvm_ = pointerFromString<JavaVM*>(jvm_pointer);
    JNIEnv* env;
    int jvmEnvStatus = jvm_->GetEnv((void **) &env, JNI_VERSION_1_6);
    if (jvmEnvStatus != JNI_OK) {
      jint status = jvm_->AttachCurrentThread((void**) &env, nullptr);
      assert(status == JNI_OK);
    }
    std::string registry_pointer;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("registry_pointer", &registry_pointer));
    registry_ = pointerFromString<jclass>(registry_pointer);
    std::string registry_call_pointer;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("registry_call_pointer", &registry_call_pointer));
    call_method_id_ = pointerFromString<jmethodID>(registry_call_pointer);
    if (jvmEnvStatus != JNI_OK) {
      jint status = jvm_->DetachCurrentThread();
      assert(status == JNI_OK);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    JNIEnv* env;
    int jvmEnvStatus = jvm_->GetEnv((void**) &env, JNI_VERSION_1_6);
    if (jvmEnvStatus != JNI_OK) {
      jint status = jvm_->AttachCurrentThread((void**) &env, nullptr);
      assert(status == JNI_OK);
    }

    JVMCall call;
    call.env = env;
    call.registry = registry_;
    call.call_method_id = call_method_id_;
    call.id = id_;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call.inputs.push_back(ctx->input(i));
    }

    Status s = CallJVMFunction(&call);
    if (jvmEnvStatus != JNI_OK) {
      jint status = jvm_->DetachCurrentThread();
      assert(status == JNI_OK);
    }

    OP_REQUIRES_OK(ctx, s);

    OP_REQUIRES(ctx, static_cast<int32>(call.outputs.size()) == ctx->num_outputs(),
                errors::InvalidArgument(id_, " returns ", call.outputs.size(),
                                        " values, but expects to see ",
                                        ctx->num_outputs(), " values."));
    for (size_t i = 0; i < call.outputs.size(); ++i) {
      const auto& t = call.outputs[i];
      OP_REQUIRES(
          ctx, t.dtype() == output_type(i),
          errors::InvalidArgument(i, "-th value returned by ", id_, " is ",
                                  DataTypeString(t.dtype()), ", but expects ",
                                  DataTypeString(output_type(i))));
      ctx->set_output(i, t);
    }
  }

private:
  int id_;
  JavaVM* jvm_;
  jclass registry_;
  jmethodID call_method_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(JVMCallbackOp);
};

namespace kernel_factory {
struct KernelRegistration {
  KernelRegistration(const KernelDef& d, StringPiece c,
                     kernel_factory::OpKernelRegistrar::Factory f)
      : def(d), kernel_class_name(c.ToString()), factory(f) {}
  const KernelDef def;
  const string kernel_class_name;
  const kernel_factory::OpKernelRegistrar::Factory factory;
};

auto jvmCallbackOpInitializer = []{
  auto* reg = reinterpret_cast<std::unordered_multimap<string, KernelRegistration>*>(GlobalKernelRegistry());
  if (reg->find(strings::StrCat("JVMCallback:", DeviceTypeString(DEVICE_CPU), ":")) == reg->end()) {
    REGISTER_KERNEL_BUILDER(Name("JVMCallback").Device(DEVICE_CPU), JVMCallbackOp);
    REGISTER_KERNEL_BUILDER(Name("JVMCallbackStateless").Device(DEVICE_CPU), JVMCallbackOp);
  }
  return 0;
}();
}
}  // namespace tensorflow
