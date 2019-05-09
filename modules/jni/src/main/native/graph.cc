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

#include "exception.h"
#include "graph.h"
#include "utilities.h"

#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/c/c_api.h"

namespace {
  std::unique_ptr<TF_Output[]> to_tf_output_array(
    JNIEnv *env,
    jobjectArray
    java_array,
    jfieldID output_op_handle_field_id,
    jfieldID output_op_index_field_id
  ) {
    if (java_array == nullptr) return nullptr;
    int array_length = env->GetArrayLength(java_array);
    std::unique_ptr<TF_Output[]> array(new TF_Output[array_length]);
    for (int i = 0; i < array_length; ++i) {
      jobject object = env->GetObjectArrayElement(java_array, i);
      jlong op_handle = env->GetLongField(object, output_op_handle_field_id);
      jint output_index = env->GetIntField(object, output_op_index_field_id);
      REQUIRE_HANDLE(op, TF_Operation, op_handle, nullptr);
      array[i] = {op, output_index};
    }
    return array;
  }
}  // namespace

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_allocate(
  JNIEnv* env,
  jobject object
) {
  return reinterpret_cast<jlong>(TF_NewGraph());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_delete(
  JNIEnv* env,
  jobject object,
  jlong graph_handle
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, void());
  TF_DeleteGraph(g);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_findOp(
  JNIEnv* env,
  jobject object,
  jlong graph_handle,
  jstring op_name
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, 0);
  const char *op_name_c_string = env->GetStringUTFChars(op_name, nullptr);
  TF_Operation *op = TF_GraphOperationByName(g, op_name_c_string);
  env->ReleaseStringUTFChars(op_name, op_name_c_string);
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_ops(
  JNIEnv* env,
  jobject object,
  jlong graph_handle
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, nullptr);

  // Call the C API "TF_GraphNextOperation" repeatedly to obtain all ops in the graph.
  std::vector<TF_Operation *> ops;
  size_t pos = 0;
  TF_Operation *op;
  while ((op = TF_GraphNextOperation(g, &pos)) != nullptr)
    ops.push_back(op);

  // Construct the return array.
  jlongArray return_array = env->NewLongArray(static_cast<jsize>(ops.size()));
  jlong *op_handles_array = env->GetLongArrayElements(return_array, nullptr);
  for (int i = 0; i < ops.size(); ++i)
    op_handles_array[i] = reinterpret_cast<jlong>(ops[i]);
  env->ReleaseLongArrayElements(return_array, op_handles_array, 0);
  return return_array;
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_addGradients(
  JNIEnv* env,
  jobject object,
  jlong graph_handle,
  jobjectArray y_array,
  jobjectArray x_array,
  jobjectArray dx_array
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, nullptr);

  // Convert the inputs to their C API equivalent data structures
  jclass output_class = env->FindClass("org/platanios/tensorflow/jni/Output");
  jfieldID output_op_handle_field_id = env->GetFieldID(output_class, "opHandle", "J");
  jfieldID output_op_index_field_id = env->GetFieldID(output_class, "outputIndex", "I");

  int ny = env->GetArrayLength(y_array);
  std::unique_ptr<TF_Output[]> y = to_tf_output_array(
      env, y_array, output_op_handle_field_id, output_op_index_field_id);
  if (y == nullptr) return nullptr;
  int nx = env->GetArrayLength(x_array);
  std::unique_ptr<TF_Output[]> x = to_tf_output_array(
      env, x_array, output_op_handle_field_id, output_op_index_field_id);
  if (x == nullptr) return nullptr;
  std::unique_ptr<TF_Output[]> dx = to_tf_output_array(
      env, dx_array, output_op_handle_field_id, output_op_index_field_id);
  std::unique_ptr<TF_Output[]> dy(new TF_Output[nx]);

  // Call the C API "TF_AddGradients" function and throw an exception if an error occurs.
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_AddGradients(g, y.get(), ny, x.get(), nx, dx.get(), status.get(), dy.get());
  CHECK_STATUS(env, status.get(), nullptr);

  // Construct the return gradients array
  jmethodID output_class_constructor = env->GetStaticMethodID(
      output_class, "apply", "(JI)Lorg/platanios/tensorflow/jni/Output;");
  jobjectArray gradients_array = env->NewObjectArray(nx, output_class, NULL);
  for (int i = 0; i < nx; ++i) {
    env->SetObjectArrayElement(
        gradients_array, i, env->CallStaticObjectMethod(
            output_class, output_class_constructor,
            reinterpret_cast<jlong>(dy[i].oper), dy[i].index));
  }
  return gradients_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_importGraphDef(
  JNIEnv* env,
  jobject object,
  jlong graph_handle,
  jbyteArray graph_def,
  jstring name_prefix,
  jobjectArray input_map_key_ops,
  jintArray input_map_key_outputs,
  jlongArray input_map_value_ops,
  jintArray input_map_value_outputs,
  jobjectArray control_dependency_map_key_ops,
  jlongArray control_dependency_map_value_ops,
  jlongArray control_dependencies
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, void());

  TF_ImportGraphDefOptions *options = TF_NewImportGraphDefOptions();

  // Handle the name prefix argument.
  const char *name_prefix_c_string = env->GetStringUTFChars(name_prefix, nullptr);
  TF_ImportGraphDefOptionsSetPrefix(options, name_prefix_c_string);
  env->ReleaseStringUTFChars(name_prefix, name_prefix_c_string);

  // Handle the input map arguments.
  if (input_map_key_ops != nullptr) {
    int input_map_length = env->GetArrayLength(input_map_key_ops);
    if (input_map_length != env->GetArrayLength(input_map_key_outputs) ||
          input_map_length != env->GetArrayLength(input_map_value_ops) ||
          input_map_length != env->GetArrayLength(input_map_value_outputs))
      throw_exception(env, tf_invalid_argument_exception, "All input map arguments must have the same length.");
    jint *input_map_key_outputs_elements = env->GetIntArrayElements(input_map_key_outputs, 0);
    jlong *input_map_value_ops_elements = env->GetLongArrayElements(input_map_value_ops, 0);
    jint *input_map_value_outputs_elements = env->GetIntArrayElements(input_map_value_outputs, 0);
    for (int i = 0; i < input_map_length; ++i) {
      jstring input_map_key_op = reinterpret_cast<jstring>(env->GetObjectArrayElement(input_map_key_ops, i));
      const char *input_map_key_op_c_string = env->GetStringUTFChars(input_map_key_op, nullptr);
      int input_map_key_output = reinterpret_cast<int>(input_map_key_outputs_elements[i]);
      REQUIRE_HANDLE(op, TF_Operation, input_map_value_ops_elements[i], void());
      int output_index = reinterpret_cast<int>(input_map_value_outputs_elements[i]);
      TF_Output output{op, output_index};
      TF_ImportGraphDefOptionsAddInputMapping(options, input_map_key_op_c_string, input_map_key_output, output);
      env->ReleaseStringUTFChars(input_map_key_op, input_map_key_op_c_string);
    }
    env->ReleaseIntArrayElements(input_map_key_outputs, input_map_key_outputs_elements, 0);
    env->ReleaseLongArrayElements(input_map_value_ops, input_map_value_ops_elements, 0);
    env->ReleaseIntArrayElements(input_map_value_outputs, input_map_value_outputs_elements, 0);
  }

  // Handle the control dependency map arguments
  if (control_dependency_map_key_ops != nullptr) {
    int control_dependency_map_length = env->GetArrayLength(control_dependency_map_key_ops);
    if (control_dependency_map_length != env->GetArrayLength(control_dependency_map_value_ops))
      throw_exception(
        env, tf_invalid_argument_exception, "All control dependency map arguments must have the same length.");
     jlong *control_dependency_map_value_ops_elements = env->GetLongArrayElements(control_dependency_map_value_ops, 0);
     for (int i = 0; i < control_dependency_map_length; ++i) {
      jstring control_dependency_map_key_op =
        reinterpret_cast<jstring>(env->GetObjectArrayElement(control_dependency_map_key_ops, i));
      const char *control_dependency_map_key_op_c_string =
        env->GetStringUTFChars(control_dependency_map_key_op, nullptr);
      REQUIRE_HANDLE(op, TF_Operation, control_dependency_map_value_ops_elements[i], void());
      TF_ImportGraphDefOptionsRemapControlDependency(options, control_dependency_map_key_op_c_string, op);
      env->ReleaseStringUTFChars(control_dependency_map_key_op, control_dependency_map_key_op_c_string);
    }
    env->ReleaseLongArrayElements(control_dependency_map_value_ops, control_dependency_map_value_ops_elements, 0);
  }

  // Handle the control dependencies argument
  if (control_dependencies != nullptr) {
    int control_dependencies_length = env->GetArrayLength(control_dependencies);
    jlong *control_dependencies_elements = env->GetLongArrayElements(control_dependencies, 0);
    for (int i = 0; i < control_dependencies_length; ++i) {
      REQUIRE_HANDLE(op, TF_Operation, control_dependencies_elements[i], void());
      TF_ImportGraphDefOptionsAddControlDependency(options, op);
    }
    env->ReleaseLongArrayElements(control_dependencies, control_dependencies_elements, 0);
  }

  // Call the C API "TF_GraphImportGraphDef" function and throw an exception if an error occurs
  static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
  jbyte *bytes = env->GetByteArrayElements(graph_def, nullptr);
  TF_Buffer *buffer = TF_NewBufferFromString(bytes, static_cast<size_t>(env->GetArrayLength(graph_def)));
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_GraphImportGraphDef(g, buffer, options, status.get());
  CHECK_STATUS(env, status.get(), void());

  // Continue cleaning up resources even if an exception was thrown.
  TF_DeleteBuffer(buffer);
  env->ReleaseByteArrayElements(graph_def, bytes, JNI_ABORT);
  TF_DeleteImportGraphDefOptions(options);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_toGraphDef(
  JNIEnv* env,
  jobject object,
  jlong graph_handle
) {
  REQUIRE_HANDLE(g, TF_Graph, graph_handle, nullptr);

  // Call the C API "TF_GraphToGraphDef" function and throw an exception if an error occurs
  jbyteArray return_array = nullptr;
  TF_Buffer *buf = TF_NewBuffer();
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_GraphToGraphDef(g, buf, status.get());
  if (throw_exception_if_not_ok(env, status.get())) {
    // sizeof(jsize) is less than sizeof(size_t) on some platforms.
    if (buf->length > std::numeric_limits<jint>::max()) {
      throw_exception(env, tf_invalid_argument_exception,
                      "GraphDef is too large to serialize into a Java byte array.");
    } else {
      static_assert(sizeof(jbyte) == 1, "Unexpected size of the Java byte type.");
      jint return_array_length = static_cast<jint>(buf->length);
      return_array = env->NewByteArray(return_array_length);
      env->SetByteArrayRegion(return_array, 0, return_array_length, static_cast<const jbyte *>(buf->data));
    }
  }

  // Clean up and return the byte array.
  TF_DeleteBuffer(buf);
  return return_array;
}
