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

#include "function.h"

#include <limits>
#include <memory>

#include "c_api.h"
#include "exception.h"
#include "utilities.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Function_00024_graphToFunction(
  JNIEnv* env, jobject object, jlong fn_body_graph_handle, jstring fn_name, jlongArray op_handles,
  jlongArray input_op_handles, jintArray input_op_indices,
  jlongArray output_op_handles, jintArray output_op_indices,
  jobjectArray output_names) {
  REQUIRE_HANDLE(fn_body_graph, TF_Graph, fn_body_graph_handle, 0);

  const char *c_fn_name = env->GetStringUTFChars(fn_name, nullptr);

  const int num_ops = op_handles == NULL ? 0 : env->GetArrayLength(op_handles);
  const int num_inputs = env->GetArrayLength(input_op_handles);
  const int num_outputs = env->GetArrayLength(output_op_handles);

  std::unique_ptr<TF_Operation* []> ops(new TF_Operation* [num_ops]);
  std::unique_ptr<TF_Output[]> inputs(new TF_Output[num_inputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[num_outputs]);

  if (num_ops > 0) {
    REQUIRE_HANDLES(op_handles, ops.get(), num_ops, 0);
  }

  REQUIRE_OUTPUTS(input_op_handles, input_op_indices, inputs.get(), num_inputs, 0);
  REQUIRE_OUTPUTS(output_op_handles, output_op_indices, outputs.get(), num_outputs, 0);

  int num_output_names = env->GetArrayLength(output_names);
  std::unique_ptr<jstring[]> j_output_names(new jstring[num_output_names]);
  const char **c_output_names = new const char *[num_output_names];
  for (int i = 0; i < num_output_names; i++) {
    j_output_names[i] = (jstring) env->GetObjectArrayElement(output_names, i);
    c_output_names[i] = env->GetStringUTFChars(j_output_names[i], nullptr);
  }

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_Function* function = TF_GraphToFunction(
    fn_body_graph, c_fn_name, op_handles == NULL ? -1 : num_ops, ops.get(), num_inputs, inputs.get(), num_outputs,
    outputs.get(), c_output_names, /*opts=*/nullptr, status.get());
  CHECK_STATUS(env, status.get(), 0);

  env->ReleaseStringUTFChars(fn_name, c_fn_name);
  for (int i = 0; i < num_output_names; i++) {
    env->ReleaseStringUTFChars(j_output_names[i], c_output_names[i]);
  }

  return reinterpret_cast<jlong>(function);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Function_00024_addToGraph(
  JNIEnv* env, jobject object, jlong graph_handle, jlong function_handle) {
  REQUIRE_HANDLE(graph, TF_Graph, graph_handle, void());
  REQUIRE_HANDLE(function, TF_Function, function_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_GraphAddFunction(graph, function, status.get());
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Function_00024_toFunctionDef(
    JNIEnv* env, jobject object, jlong function_handle) {
  REQUIRE_HANDLE(function, TF_Function, function_handle, nullptr);

  // Call the C API "TF_GraphToGraphDef" function and throw an exception if an error occurs
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  jbyteArray return_array = nullptr;
  TF_Buffer *buffer = TF_NewBuffer();
  TF_FunctionToFunctionDef(function, buffer, status.get());
  CHECK_STATUS(env, status.get(), nullptr);
  
  // sizeof(jsize) is less than sizeof(size_t) on some platforms.
  if (buffer->length > std::numeric_limits<jint>::max()) {
    throw_exception(env, jvm_index_out_of_bounds_exception,
                    "GraphDef is too large to serialize into a Java byte array.");
  } else {
    static_assert(sizeof(jbyte) == 1, "Unexpected size of the Java byte type.");
    jint return_array_length = static_cast<jint>(buffer->length);
    return_array = env->NewByteArray(return_array_length);
    env->SetByteArrayRegion(return_array, 0, return_array_length, static_cast<const jbyte *>(buffer->data));
  }

  // Clean up and return the byte array
  TF_DeleteBuffer(buffer);
  return return_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Function_00024_delete(
    JNIEnv* env, jobject object, jlong function_handle) {
  REQUIRE_HANDLE(function, TF_Function, function_handle, void());
  TF_DeleteFunction(function);
}
