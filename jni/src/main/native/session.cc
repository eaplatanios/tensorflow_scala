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

#include "session.h"

#include <string.h>
#include <memory>

#include "exception.h"
#include "tf_c_api.h"
#include "utilities.h"

namespace {
  void TF_MaybeDeleteBuffer(TF_Buffer* buffer) {
    if (buffer == nullptr) return;
    TF_DeleteBuffer(buffer);
  }

  typedef std::unique_ptr<TF_Buffer, decltype(&TF_MaybeDeleteBuffer)> unique_tf_buffer;

  unique_tf_buffer MakeUniqueBuffer(TF_Buffer* buffer) {
    return unique_tf_buffer(buffer, (void (&&)(TF_Buffer*)) TF_MaybeDeleteBuffer);
  }
}  // namespace

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Session_00024_allocate(
    JNIEnv* env, jobject object, jlong graph_handle, jstring target, jbyteArray config_proto) {
  REQUIRE_HANDLE(graph, TF_Graph, graph_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  TF_SessionOptions* options = TF_NewSessionOptions();

  // Set the target, if one has been provided.
  const char* c_target;
  if (target != nullptr) {
    c_target = env->GetStringUTFChars(target, nullptr);
    TF_SetTarget(options, c_target);
  }

  // Set the configuration proto, if one has been provided.
  jbyte* c_config_proto;
  if (config_proto != nullptr) {
    c_config_proto = env->GetByteArrayElements(config_proto, nullptr);
    TF_SetConfig(options, c_config_proto, static_cast<size_t>(env->GetArrayLength(config_proto)), status.get());
    CHECK_STATUS(env, status.get(), 0);
  }

  TF_Session* session = TF_NewSession(graph, options, status.get());
  CHECK_STATUS(env, status.get(), 0);

  TF_DeleteSessionOptions(options);

  if (target != nullptr) {
    env->ReleaseStringUTFChars(target, c_target);
  }

  if (config_proto != nullptr) {
    env->ReleaseByteArrayElements(config_proto, c_config_proto, JNI_ABORT);
  }

  return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Session_00024_delete(
    JNIEnv* env, jobject object, jlong handle) {
  REQUIRE_HANDLE(session, TF_Session, handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_CloseSession(session, status.get());
  CHECK_STATUS(env, status.get(), void());
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(session, status.get());
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Session_00024_run(
    JNIEnv* env, jobject object, jlong handle, jbyteArray jrun_options, jlongArray input_tensor_handles,
    jlongArray input_op_handles, jintArray input_op_indices, jlongArray output_op_handles, jintArray output_op_indices,
    jlongArray target_op_handles, jboolean want_run_metadata, jlongArray output_tensor_handles) {
  REQUIRE_HANDLE(session, TF_Session, handle, nullptr);

  const jint num_inputs = env->GetArrayLength(input_tensor_handles);
  const jint num_outputs = env->GetArrayLength(output_tensor_handles);
  const jint num_targets = env->GetArrayLength(target_op_handles);

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[num_inputs]);
  std::unique_ptr<TF_Tensor* []> input_values(new TF_Tensor* [num_inputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[num_outputs]);
  std::unique_ptr<TF_Tensor* []> output_values(new TF_Tensor* [num_outputs]);
  std::unique_ptr<TF_Operation* []> targets(new TF_Operation* [num_targets]);
  unique_tf_buffer run_metadata(MakeUniqueBuffer(want_run_metadata ? TF_NewBuffer() : nullptr));

  REQUIRE_HANDLES(input_tensor_handles, input_values.get(), num_inputs, nullptr);
  REQUIRE_OUTPUTS(input_op_handles, input_op_indices, inputs.get(), num_inputs, nullptr);
  REQUIRE_OUTPUTS(output_op_handles, output_op_indices, outputs.get(), num_outputs, nullptr);
  REQUIRE_HANDLES(target_op_handles, targets.get(), num_targets, nullptr);

  unique_tf_buffer run_options(MakeUniqueBuffer(nullptr));
  jbyte* jrun_options_data = nullptr;
  if (jrun_options != nullptr) {
    size_t sz = (size_t) env->GetArrayLength(jrun_options);
    if (sz > 0) {
      jrun_options_data = env->GetByteArrayElements(jrun_options, nullptr);
      run_options.reset(TF_NewBufferFromString(static_cast<void*>(jrun_options_data), sz));
    }
  }

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_SessionRun(
      session, run_options.get(), inputs.get(), input_values.get(), static_cast<int>(num_inputs), outputs.get(),
      output_values.get(), static_cast<int>(num_outputs), reinterpret_cast<const TF_Operation* const*>(targets.get()),
      static_cast<int>(num_targets), run_metadata.get(), status.get());
  CHECK_STATUS(env, status.get(), nullptr);

  if (jrun_options_data != nullptr)
    env->ReleaseByteArrayElements(jrun_options, jrun_options_data, JNI_ABORT);

  jlong* output_tensor_handles_array = env->GetLongArrayElements(output_tensor_handles, nullptr);
  for (int i = 0; i < num_outputs; ++i)
    output_tensor_handles_array[i] = reinterpret_cast<jlong>(output_values[i]);
  env->ReleaseLongArrayElements(output_tensor_handles, output_tensor_handles_array, 0);

  jbyteArray return_array = nullptr;
  if (run_metadata != nullptr) {
    return_array = env->NewByteArray(static_cast<jsize>(run_metadata->length));
    jbyte* elements = env->GetByteArrayElements(return_array, nullptr);
    memcpy(elements, run_metadata->data, run_metadata->length);
    env->ReleaseByteArrayElements(return_array, elements, JNI_COMMIT);
  }

  return return_array;
}
