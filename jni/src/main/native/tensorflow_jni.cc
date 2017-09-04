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

#include "tensorflow_jni.h"

#include <dlfcn.h>
#include <iostream>
#include <memory>

#include "c_api.h"
//#include "python_api.h"
#include "exception_jni.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throw_exception(
        env, jvm_null_pointer_exception,
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

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_loadGlobal(
  JNIEnv* env, jobject object, jstring lib_path) {
  const char *c_lib_path = env->GetStringUTFChars(lib_path, nullptr);
  void* h = dlopen(c_lib_path, RTLD_LAZY | RTLD_GLOBAL);
  if (h) {
    std::cout << "Loading dynamic library '" << c_lib_path << "' succeeded." << std::endl;
  } else {
    std::cout << "Loading dynamic library '" << c_lib_path << "' failed." << std::endl;
  }
  env->ReleaseStringUTFChars(lib_path, c_lib_path);
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_version(
  JNIEnv* env, jobject object) {
  return env->NewStringUTF(TF_Version());
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_dataTypeSize(
  JNIEnv* env, jobject object, jint data_type_c_value) {
  return (jint) TF_DataTypeSize((TF_DataType) data_type_c_value);
}

//JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_updateInput(
//  JNIEnv* env, jobject object, jlong graph_handle, jlong input_op_handle, jint input_index, jlong output_op_handle,
//  jint output_index) {
//  TF_Graph* graph = require_graph_handle(env, graph_handle);
//  TF_Operation* input_op = require_operation_handle(env, input_op_handle);
//  TF_Operation* output_op = require_operation_handle(env, output_op_handle);
//  if (graph == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Graph could not be found.");
//  if (input_op == nullptr || output_op == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Operation could not be found.");
//  TF_Output output{output_op, output_index};
//  // tensorflow::UpdateInput(graph, input_op, input_index, output);
//  return 0;
//}
//
//JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_addControlInput(
//  JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jlong input_op_handle) {
//  TF_Graph* graph = require_graph_handle(env, graph_handle);
//  TF_Operation* op = require_operation_handle(env, op_handle);
//  TF_Operation* input_op = require_operation_handle(env, input_op_handle);
//  if (graph == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Graph could not be found.");
//  if (op == nullptr || input_op == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Operation could not be found.");
//  // tensorflow::AddControlInput(graph, op, input_op);
//  return 0;
//}
//
//JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_clearControlInputs(
//  JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle) {
//  TF_Graph* graph = require_graph_handle(env, graph_handle);
//  TF_Operation* op = require_operation_handle(env, op_handle);
//  if (graph == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Graph could not be found.");
//  if (op == nullptr)
//    throw_exception(env, jvm_illegal_state_exception, "Operation could not be found.");
//  // tensorflow::ClearControlInputs(graph, op);
//  return 0;
//}
