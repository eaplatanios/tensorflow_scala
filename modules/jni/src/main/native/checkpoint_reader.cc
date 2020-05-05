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

#include "checkpoint_reader.h"
#include "checkpoint_reader_internal.h"
#include "utilities.h"

#include <string.h>

#include "tensorflow/c/eager/c_api.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_newCheckpointReader(
    JNIEnv* env, jobject object, jstring file_pattern) {
  const char* c_file_pattern = env->GetStringUTFChars(file_pattern, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  auto* reader = new tensorflow::checkpoint::CheckpointReader(std::string(c_file_pattern), status.get());
  if (!throw_exception_if_not_ok(env, status.get())) {
      delete reader;
      return 0;
  }
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
  TFE_TensorHandle* tfe_tensor = nullptr; // TODO: !!! TFE_NewTensorHandle(*tensor.get(), status.get());
  return (jlong) tfe_tensor;
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_variableShapes(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  const tensorflow::checkpoint::TensorSliceReader::VarToShapeMap map = reader->GetVariableToShapeMap();

  jobjectArray jvariables = env->NewObjectArray(map.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));
  jobjectArray jshapes = env->NewObjectArray(map.size(), env->FindClass("[J"), env->NewLongArray(0));
  int i = 0;
  for(const auto& pair : map) {
    char *value = new char[pair.first.size() + 1];
    strncpy(value, reinterpret_cast<const char *>(pair.first.c_str()), pair.first.size());
    value[pair.first.size()] = '\0';
    env->SetObjectArrayElement(jvariables, i, env->NewStringUTF(value));

    tensorflow::TensorShape shape = pair.second;
    jlongArray jshape = env->NewLongArray(shape.dims());
    jlong *jdims = env->GetLongArrayElements(jshape, nullptr);
    for (int j = 0; j < shape.dims(); ++j) {
      jdims[j] = static_cast<jlong>(shape.dim_size(j));
    }
    env->ReleaseLongArrayElements(jshape, jdims, 0);
    env->SetObjectArrayElement(jshapes, i, jshape);
    i += 1;
  }

  jclass variableShapesClass = env->FindClass("org/platanios/tensorflow/jni/VariableShapes");
  jmethodID classConstructor = env->GetStaticMethodID(
    variableShapesClass, "apply", "([Ljava/lang/String;[[J)Lorg/platanios/tensorflow/jni/VariableShapes;");
  jobject jvariableShapes = env->CallStaticObjectMethod(variableShapesClass, classConstructor, jvariables, jshapes);
  CHECK_STATUS(env, status.get(), nullptr);
  return jvariableShapes;
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_variableDataTypes(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  const tensorflow::checkpoint::TensorSliceReader::VarToDataTypeMap map = reader->GetVariableToDataTypeMap();

  jobjectArray jvariables = env->NewObjectArray(map.size(), env->FindClass("java/lang/String"), env->NewStringUTF(""));
  jintArray jtypes = env->NewIntArray(map.size());
  jint *types = env->GetIntArrayElements(jtypes, nullptr);
  int i = 0;
  for(const auto& pair : map) {
    char *value = new char[pair.first.size() + 1];
    strncpy(value, reinterpret_cast<const char *>(pair.first.c_str()), pair.first.size());
    value[pair.first.size()] = '\0';
    env->SetObjectArrayElement(jvariables, i, env->NewStringUTF(value));
    types[i] = static_cast<jint>(pair.second);
    i += 1;
  }
  env->ReleaseIntArrayElements(jtypes, types, 0);

  jclass variableDataTypesClass = env->FindClass("org/platanios/tensorflow/jni/VariableDataTypes");
  jmethodID classConstructor = env->GetStaticMethodID(
    variableDataTypesClass, "apply", "([Ljava/lang/String;[I)Lorg/platanios/tensorflow/jni/VariableDataTypes;");
  jobject jvariableDataTypes = env->CallStaticObjectMethod(variableDataTypesClass, classConstructor, jvariables, jtypes);
  CHECK_STATUS(env, status.get(), nullptr);
  return jvariableDataTypes;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_CheckpointReader_00024_delete(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::checkpoint::CheckpointReader, reader_handle, void());
  delete reader;
}
