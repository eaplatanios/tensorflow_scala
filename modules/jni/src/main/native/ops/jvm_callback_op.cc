/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

#include "jvm_callback_op.h"

#include "exception.h"
#include "utilities.h"

#include "tensorflow/c/ops.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {
  // Given the 'call', prepares the inputs as a JNI long array that is appropriate for calling the registry.
  jlongArray MakeInputs(JVMCall* call) {
    unsigned long n = call->inputs.size();
    jlongArray inputs = call->env->NewLongArray(static_cast<jsize>(n));
    jlong* inputs_array = call->env->GetLongArrayElements(inputs, nullptr);
    for (int64 i = 0; i < n; ++i) {
      TF_Tensor* t = call->inputs[i];
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
      TFE_TensorHandle* tensor = TFE_NewTensorHandle(t, status.get());
      CHECK_STATUS(call->env, status.get(), nullptr);
      inputs_array[i] = reinterpret_cast<jlong>(tensor);
    }
    call->env->ReleaseLongArrayElements(inputs, inputs_array, 0);
    return inputs;
  }

  // Process the return values by converting them back to TensorFlow tensors and adding them to the call outputs.
  void ProcessOutputs(JVMCall* call, jlongArray call_outputs) {
    call->outputs.clear();
    jsize n = call->env->GetArrayLength(call_outputs);
    jlong* outputs_array = call->env->GetLongArrayElements(call_outputs, nullptr);
    for (int i = 0; i < n; ++i) {
      static_assert(sizeof(jlong) >= sizeof(TFE_TensorHandle*), "Cannot package C object pointers as a Java long");
      if (outputs_array[i] == 0) {
        TF_Status* status = TF_NewStatus();
        TF_SetStatus(status, TF_INVALID_ARGUMENT, "One of the op output tensors has been disposed already.");
        TF_OpKernelContext_Failure(call->ctx, status);
        return;
      }
      auto* h = reinterpret_cast<TFE_TensorHandle*>(outputs_array[i]);
      if (h == nullptr) {
        TF_Status* status = TF_NewStatus();
        TF_SetStatus(status, TF_INVALID_ARGUMENT, "Could not obtain tensor handle to one of the outputs.");
        TF_OpKernelContext_Failure(call->ctx, status);
        return;
      }
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
      TF_Tensor* t = TFE_TensorHandleResolve(h, status.get());
      CHECK_STATUS(call->env, status.get(), void());
      call->outputs.push_back(t);
    }
    call->env->ReleaseLongArrayElements(call_outputs, outputs_array, 0);
  }

  // Calls the registered JVM function through the registry.
  void CallJVMFunction(JVMCall* call) {
    // Prepare the call arguments.
    jlongArray call_inputs = MakeInputs(call);

    // Invoke the registry 'call' method.
    if (call->registry != nullptr && call->call_method_id != nullptr) {
      auto outputs = (jlongArray) call->env->CallStaticObjectMethod(
          call->registry, call->call_method_id, call->id, call_inputs);
      jthrowable exc(call->env->ExceptionOccurred());
      if (exc) {
        // Get the exception string representation to use as the error message.
        jclass excObjCls(call->env->GetObjectClass(exc));
        jmethodID toString = call->env->GetMethodID(excObjCls, "toString", "()Ljava/lang/String;");
        jstring excString = (jstring) call->env->CallObjectMethod(exc, toString);
        const char* excCString = call->env->GetStringUTFChars(excString, 0);
        std::string excCppString(excCString);
        call->env->ReleaseStringUTFChars(excString, excCString);

        // Get the exception class name and convert it to a TensorFlow error code.
        jclass classCls(call->env->FindClass("java/lang/Class"));
        jmethodID getName(call->env->GetMethodID(classCls, "getName", "()Ljava/lang/String;"));
        jstring clsName(static_cast<jstring>(call->env->CallObjectMethod(excObjCls, getName)));
        const char* clsNameCString = call->env->GetStringUTFChars(clsName, 0);
        std::string clsNameCppString(clsNameCString);
        TF_Code error_code = tf_error_code(clsNameCppString);
        call->env->ReleaseStringUTFChars(clsName, clsNameCString);
        call->env->ExceptionClear();

        TF_Status* status = TF_NewStatus();
        TF_SetStatus(status, error_code, excCppString.c_str());
        TF_OpKernelContext_Failure(call->ctx, status);
        return;
      }

      if (outputs == nullptr) {
        TF_Status* status = TF_NewStatus();
        TF_SetStatus(
          status, TF_UNKNOWN, "Failed to run JVM callback function.");
        TF_OpKernelContext_Failure(call->ctx, status);
        return;
      }

      // Process the return values and convert them back to TensorFlow tensors.
      ProcessOutputs(call, outputs);
    } else {
      TF_Status* status = TF_NewStatus();
      TF_SetStatus(
        status, TF_UNKNOWN, "Failed to run JVM callback function. Could not find registry class or its 'call' method.");
      TF_OpKernelContext_Failure(call->ctx, status);
      return;
    }
  }

  struct JVMWrapper {
    mutex lock;
    JavaVM* jvm_ TF_GUARDED_BY(lock);
	  JVMWrapper(JavaVM* jvm_) : jvm_(jvm_) { }
  };

  static std::map<JavaVM*, JVMWrapper*> jvms;
  static mutex lock;

  struct JVMThreadHelper {
    JNIEnv* env;
	  JavaVM* jvm_;

    JVMThreadHelper() : jvm_(nullptr) { }

    ~JVMThreadHelper() {
      if (jvm_ != nullptr) {
        jvm_->DetachCurrentThread();
      }
    }

    void set_jvm(JavaVM* jvm) {
      if (jvm_ != nullptr) {
        if (jvm_ != jvm)
          throw "Multiple JVMs detected per thread.";
	       return;
      }
      jvm_ = jvm;
      jvm_->GetEnv((void**) &env, JNI_VERSION_1_6);
      jvm_->AttachCurrentThread((void**) &env, nullptr);
	  }

	  void unset_jvm(JavaVM* jvm) {
	    if (jvm_ != nullptr) {
        if (jvm_ != jvm)
          throw "Multiple JVMs detected per thread.";
         return;
      }
      jvm_ = jvm;
      jvm_->DetachCurrentThread();
	  }
  };

  JVMWrapper* get_jvm_wrapper(JavaVM* jvm_) {
    mutex_lock guard(lock);
    std::map<JavaVM*, JVMWrapper*>::iterator it = jvms.find(jvm_);
    if (it != jvms.end())
      return it->second;
    JVMWrapper* jvm_wrapper_ = new JVMWrapper(jvm_);
    jvms[jvm_] = jvm_wrapper_;
    return jvm_wrapper_;
  }

  void delete_jvm_wrapper(JavaVM* jvm_) {
    mutex_lock guard(lock);
    std::map<JavaVM*, JVMWrapper*>::iterator it = jvms.find(jvm_);
    if (it != jvms.end()) {
      jvms.erase(jvm_);
      delete it->second;
    }
  }

  thread_local JVMThreadHelper jvm_thread;
}  // namespace


struct JVMCallbackKernel {
  int id_;
  JVMWrapper* jvm_wrapper_;
  jclass registry_ TF_GUARDED_BY(jvm_wrapper_->lock);
  jmethodID call_method_id_ TF_GUARDED_BY(jvm_wrapper_->lock);
};

static void* JVMCallbackKernel_Create(TF_OpKernelConstruction* ctx) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  struct JVMCallbackKernel* k = new struct JVMCallbackKernel;

  TF_OpKernelConstruction_GetAttrInt32(ctx, "id", &(k->id_), status.get());
  // CHECK_STATUS(env, status.get(), nullptr);

  // Get the JVM pointer.
  int32_t jvm_pointer_upper;
  int32_t jvm_pointer_lower;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "jvm_pointer_upper", &jvm_pointer_upper, status.get());
  // CHECK_STATUS(env, status.get(), nullptr);
  TF_OpKernelConstruction_GetAttrInt32(ctx, "jvm_pointer_lower", &jvm_pointer_lower, status.get());
  // CHECK_STATUS(env, status.get(), nullptr);
  int64_t jvm_pointer = (((int64_t) jvm_pointer_upper) << 32) | (int64_t) jvm_pointer_lower;
  JavaVM* jvm_ = reinterpret_cast<JavaVM*>(jvm_pointer);
  k->jvm_wrapper_ = get_jvm_wrapper(jvm_);
  mutex_lock guard(k->jvm_wrapper_->lock);
  jvm_thread.set_jvm(k->jvm_wrapper_->jvm_);

  // Obtain a handle to the JVM environment and the callbacks registry class.
  JNIEnv* env = jvm_thread.env;
  k->registry_ = env->FindClass("org/platanios/tensorflow/jni/ScalaCallbacksRegistry");
  k->call_method_id_ = env->GetStaticMethodID(k->registry_, "call", "(I[J)[J");
  return k;
};

static void JVMCallbackKernel_Compute(void* kernel, TF_OpKernelContext* ctx) {
  struct JVMCallbackKernel* k = static_cast<struct JVMCallbackKernel*>(kernel);
  if (ctx != nullptr) {
    mutex_lock guard(k->jvm_wrapper_->lock);
    jvm_thread.set_jvm(k->jvm_wrapper_->jvm_);
    JNIEnv* env = jvm_thread.env;

    JVMCall call;
    call.env = env;
    call.registry = k->registry_;
    call.call_method_id = k->call_method_id_;
    call.ctx = ctx;
    call.id = k->id_;

    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

    for (int i = 0; i < TF_NumInputs(ctx); ++i) {
      TF_Tensor* tensor;
      TF_GetInput(ctx, i, &tensor, status.get());
      CHECK_STATUS(env, status.get(), void());
      call.inputs.push_back(tensor);
    }

    CallJVMFunction(&call);

    if (static_cast<int32>(call.outputs.size()) != TF_NumOutputs(ctx)) {
      TF_SetStatus(
        status.get(), TF_INVALID_ARGUMENT,
        (std::to_string(k->id_) + " returns " +
        std::to_string(call.outputs.size()) +
        " values, but expects to see " +
        std::to_string(TF_NumOutputs(ctx)) + " values.").c_str());
      TF_OpKernelContext_Failure(ctx, status.get());
    }

    for (size_t i = 0; i < call.outputs.size(); ++i) {
      const TF_Tensor* t = call.outputs[i];
      if (TF_TensorType(t) != TF_ExpectedOutputDataType(call.ctx, i)) {
        TF_SetStatus(
          status.get(), TF_INVALID_ARGUMENT,
          (std::to_string(i) + " -th value returned by " +
          std::to_string(k->id_) + " is " +
          std::to_string(TF_TensorType(t)) + ", but expects " +
          std::to_string(TF_ExpectedOutputDataType(call.ctx, i))).c_str());
        TF_OpKernelContext_Failure(ctx, status.get());
      }

      TF_SetOutput(ctx, i, t, status.get());
      CHECK_STATUS(env, status.get(), void());
    }
  }
};

static void JVMCallbackKernel_Delete(void* kernel) {
  struct JVMCallbackKernel* k = static_cast<struct JVMCallbackKernel*>(kernel);
  delete_jvm_wrapper(k->jvm_wrapper_->jvm_);
  delete k;
};

TF_ATTRIBUTE_UNUSED static bool IsJVMCallbackOpRegistered = []() {
  if (SHOULD_REGISTER_OP("JVMCallback")) {
    TF_Status* status = TF_NewStatus();
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("JVMCallback");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tin");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder, "id: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "jvm_pointer_upper: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "jvm_pointer_lower: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tin: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: list(type) >=0");
    TF_OpDefinitionBuilderSetIsStateful(op_builder, true);
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder, &TF_ShapeInferenceContextSetUnknownShape);
    TF_RegisterOpDefinition(op_builder, status);
    CHECK_EQ(TF_GetCode(status), TF_OK) << "JVM callback op registration failed: " << TF_Message(status);
    TF_DeleteStatus(status);
  }
  return true;
}();

TF_ATTRIBUTE_UNUSED static bool IsJVMCallbackStatelessOpRegistered = []() {
  if (SHOULD_REGISTER_OP("JVMCallbackStateless")) {
    TF_Status* status = TF_NewStatus();
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("JVMCallbackStateless");
    TF_OpDefinitionBuilderAddInput(op_builder, "input: Tin");
    TF_OpDefinitionBuilderAddOutput(op_builder, "output: Tout");
    TF_OpDefinitionBuilderAddAttr(op_builder, "id: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "jvm_pointer_upper: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "jvm_pointer_lower: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tin: list(type) >= 0");
    TF_OpDefinitionBuilderAddAttr(op_builder, "Tout: list(type) >=0");
    TF_OpDefinitionBuilderSetIsStateful(op_builder, false);
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder, &TF_ShapeInferenceContextSetUnknownShape);
    TF_RegisterOpDefinition(op_builder, status);
    CHECK_EQ(TF_GetCode(status), TF_OK) << "JVM callback stateless op registration failed: " << TF_Message(status);
    TF_DeleteStatus(status);
  }
  return true;
}();

// A dummy static variable initialized by a lambda whose side-effect is to
// register the JVM callback op kernel.
TF_ATTRIBUTE_UNUSED static bool IsJVMCallbackOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("JVMCallback")) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
    TF_KernelBuilder* builder_cpu = TF_NewKernelBuilder(
      "JVMCallback", "CPU", &JVMCallbackKernel_Create,
      &JVMCallbackKernel_Compute, &JVMCallbackKernel_Delete);
    TF_RegisterKernelBuilder("JVMCallbackKernel", builder_cpu, status.get());
    // CHECK_STATUS(env, status.get(), 0);

    // TODO: [CALLBACK] Register GPU kernel.
    // TF_KernelBuilder* builder_gpu = TF_NewKernelBuilder(
    //   "JVMCallback", "GPU", &JVMCallbackKernel_Create,
    //   &JVMCallbackKernel_Compute, &JVMCallbackKernel_Delete);
    // TF_RegisterKernelBuilder("JVMCallbackKernel", builder_gpu, status.get());
    // CHECK_STATUS(env, status.get(), 0);
  }
  return true;
}();

TF_ATTRIBUTE_UNUSED static bool IsJVMCallbackStatelessOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("JVMCallbackStateless")) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
    TF_KernelBuilder* builder_cpu = TF_NewKernelBuilder(
      "JVMCallbackStateless", "CPU", &JVMCallbackKernel_Create,
      &JVMCallbackKernel_Compute, &JVMCallbackKernel_Delete);
    TF_RegisterKernelBuilder("JVMCallbackStatelessKernel", builder_cpu, status.get());
    // CHECK_STATUS(env, status.get(), 0);

    // TODO: [CALLBACK] Register GPU kernel.
    // TF_KernelBuilder* builder_gpu = TF_NewKernelBuilder(
    //   "JVMCallbackStateless", "GPU", &JVMCallbackKernel_Create,
    //   &JVMCallbackKernel_Compute, &JVMCallbackKernel_Delete);
    // TF_RegisterKernelBuilder("JVMCallbackStatelessKernel", builder_gpu, status.get());
    // CHECK_STATUS(env, status.get(), 0);
  }
  return true;
}();

}  // namespace tensorflow
