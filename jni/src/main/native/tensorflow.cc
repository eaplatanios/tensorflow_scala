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

#include "exception.h"
#include "tensorflow.h"
#include "utilities.h"

#include <cstring>
#include <iostream>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/python_api.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Operation* require_operation_handle(JNIEnv *env, jlong handle) {
  return requireHandleImpl<TF_Operation>(env, handle);
}

TF_Graph* require_graph_handle(JNIEnv *env, jlong handle) {
  return requireHandleImpl<TF_Graph>(env, handle);
}
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_jvmPointer(
    JNIEnv* env, jobject object) {
  JavaVM* jvm;
  env->GetJavaVM(&jvm);
  std::string pointer = pointerToString<JavaVM*>(jvm);
  return env->NewStringUTF(pointer.c_str());
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_callbackRegistryPointer(
    JNIEnv* env, jobject object) {
  jclass registry = env->FindClass("org/platanios/tensorflow/jni/ScalaCallbacksRegistry");
  std::string pointer = pointerToString<jobject>(env->NewGlobalRef(reinterpret_cast<jobject>(registry)));
  return env->NewStringUTF(pointer.c_str());
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_callbackRegistryCallMethodPointer(
    JNIEnv* env, jobject object) {
  jclass registry = env->FindClass("org/platanios/tensorflow/jni/ScalaCallbacksRegistry");
  jmethodID registry_call = env->GetStaticMethodID(registry, "call", "(I[J)[J");
  std::string pointer = pointerToString<jobject>(env->NewGlobalRef(reinterpret_cast<jobject>(registry_call)));
  return env->NewStringUTF(pointer.c_str());
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_version(
  JNIEnv* env, jobject object) {
  return env->NewStringUTF(TF_Version());
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_dataTypeSize(
  JNIEnv* env, jobject object, jint data_type_c_value) {
  return (jint) TF_DataTypeSize((TF_DataType) data_type_c_value);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_loadOpLibrary(
  JNIEnv* env, jobject object, jstring library_filename) {
  const char *c_library_filename = env->GetStringUTFChars(library_filename, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_Library* library = TF_LoadLibrary(c_library_filename, status.get());
  CHECK_STATUS(env, status.get(), nullptr);
  TF_Buffer op_list_buffer = TF_GetOpList(library);
  jbyteArray op_list = env->NewByteArray(op_list_buffer.length);
  jbyte* op_list_elems = env->GetByteArrayElements(op_list, nullptr);
  std::memcpy(op_list_elems, op_list_buffer.data, op_list_buffer.length);
  env->ReleaseByteArrayElements(op_list, op_list_elems, 0);
  TF_DeleteLibraryHandle(library);
  env->ReleaseStringUTFChars(library_filename, c_library_filename);
  return op_list;
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_updateInput(
  JNIEnv* env, jobject object, jlong graph_handle, jlong input_op_handle, jint input_index, jlong output_op_handle,
  jint output_index) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  TF_Operation* input_op = require_operation_handle(env, input_op_handle);
  TF_Operation* output_op = require_operation_handle(env, output_op_handle);
  if (graph == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Graph could not be found.");
  if (input_op == nullptr || output_op == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Operation could not be found.");
  TF_Output output{output_op, output_index};
  TF_Input input{input_op, input_index};
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  tensorflow::UpdateEdge(graph, output, input, status.get());
  CHECK_STATUS(env, status.get(), 1);
  return 0;
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_addControlInput(
  JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jlong input_op_handle) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  TF_Operation* op = require_operation_handle(env, op_handle);
  TF_Operation* input_op = require_operation_handle(env, input_op_handle);
  if (graph == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Graph could not be found.");
  if (op == nullptr || input_op == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Operation could not be found.");
  tensorflow::AddControlInput(graph, op, input_op);
  return 0;
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_clearControlInputs(
  JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (graph == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Graph could not be found.");
  if (op == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Operation could not be found.");
  tensorflow::ClearControlInputs(graph, op);
  return 0;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_setRequestedDevice(
  JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jstring device) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (graph == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Graph could not be found.");
  if (op == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Operation could not be found.");
  const char *c_device = env->GetStringUTFChars(device, nullptr);
  tensorflow::SetRequestedDevice(graph, op, c_device);
  env->ReleaseStringUTFChars(device, c_device);
  return;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_setAttributeProto(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jstring name, jbyteArray value) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (graph == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Graph could not be found.");
  if (op == nullptr)
    throw_exception(env, tf_invalid_argument_exception, "Operation could not be found.");
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  jbyte *c_value = env->GetByteArrayElements(value, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_Buffer* proto_buffer = TF_NewBufferFromString(c_value, static_cast<size_t>(env->GetArrayLength(value)));
  tensorflow::SetAttribute(graph, op, c_name, proto_buffer, status.get());
  TF_DeleteBuffer(proto_buffer);
  env->ReleaseByteArrayElements(value, c_value, JNI_ABORT);
  env->ReleaseStringUTFChars(name, c_name);
  CHECK_STATUS(env, status.get(), void());
}
