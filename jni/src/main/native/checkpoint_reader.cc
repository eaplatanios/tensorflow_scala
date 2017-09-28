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

#include "checkpoint_reader.h"

#include <string.h>

#include "tf_c_eager_api.h"
#include "tf_checkpoint_reader.h"
#include "utilities.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_newCheckpointReader(
    JNIEnv* env, jobject object, jstring file_pattern) {
  const char* c_file_pattern = env->GetStringUTFChars(file_pattern, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  auto* reader = new tensorflow::checkpoint::CheckpointReader(std::string(c_file_pattern), status.get());
  CHECK_STATUS(env, status.get(), 0);
  env->ReleaseStringUTFChars(file_pattern, c_file_pattern);
  return reinterpret_cast<jlong>(reader);
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_debugString(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, nullptr);
  return env->NewStringUTF(reader->DebugString().c_str());
}

JNIEXPORT jboolean JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_hasTensor(
    JNIEnv* env, jobject object, jlong reader_handle, jstring name) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, false);
  const char* c_name = env->GetStringUTFChars(name, nullptr);
  bool has_tensor = reader->HasTensor(std::string(c_name));
  env->ReleaseStringUTFChars(name, c_name);
  return (jboolean) has_tensor;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_getTensor(
    JNIEnv* env, jobject object, jlong reader_handle, jstring name) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, 0);
  const char* c_name = env->GetStringUTFChars(name, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<tensorflow::Tensor> tensor;
  reader->GetTensor(c_name, &tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  TFE_TensorHandle* tfe_tensor = new TFE_TensorHandle(*tensor.get(), nullptr);
  return (jlong) tfe_tensor;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_delete(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, void());
  delete reader;
}
