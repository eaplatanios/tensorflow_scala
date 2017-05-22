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

#include "include/tensor_jni.h"

#include <algorithm>
#include <memory>

#include "include/c_api.h"
#include "include/exception_jni.h"

namespace {
  TF_Tensor* require_graph_handle(JNIEnv *env, jlong handle) {
    static_assert(sizeof(jlong) >= sizeof(TF_Tensor*),
                  "Scala \"Long\" cannot be used to represent TensorFlow C API pointers.");
    if (handle == 0) {
      throw_exception(env, jvm_null_pointer_exception, "This tensor has already been disposed.");
      return nullptr;
    }
    return reinterpret_cast<TF_Tensor*>(handle);
  }
}  // namespace

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_fromBuffer(
    JNIEnv* env, jobject object, jobject buffer, jint dtype, jlongArray shape, jlong sizeInBytes) {
  int num_dims = static_cast<int>(env->GetArrayLength(shape));
  jlong* dims = nullptr;
  if (num_dims > 0) {
    jboolean is_copy;
    dims = env->GetLongArrayElements(shape, &is_copy);
  }
  static_assert(sizeof(jlong) == sizeof(int64_t), "Scala \"Long\" is not compatible with the TensorFlow C API.");
  // On some platforms "jlong" is a "long" while "int64_t" is a "long long".
  //
  // Thus, static_cast<int64_t*>(dims) will trigger a compiler error:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
  // *') is not allowed
  //
  // Since this array is typically very small, use the guaranteed safe scheme of
  // creating a copy.
  int64_t* dims_copy = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i)
    dims_copy[i] = static_cast<int64_t>(dims[i]);
  // Notifying the JVM of the existence of this reference to the byte buffer, to avoid garbage collection.
  // More details can be found here: http://docs.oracle.com/javase/6/docs/technotes/guides/jni/spec/design.html#wp1242
  jobject buffer_ref = env->NewGlobalRef(buffer);
  std::pair<JNIEnv*, jobject>* deallocator_arg = new std::pair<JNIEnv*, jobject>(env, buffer_ref);
  TF_Tensor* tensor_handle = TF_NewTensor(
      static_cast<TF_DataType>(dtype), dims_copy, num_dims,
      env->GetDirectBufferAddress(buffer), static_cast<size_t>(sizeInBytes),
      [](void* data, size_t len, void* arg) {
        std::pair<JNIEnv*, jobject>* pair = reinterpret_cast<std::pair<JNIEnv*, jobject>*>(arg);
        pair->first->DeleteGlobalRef(reinterpret_cast<jobject>(pair->second));
        delete (pair);
      }, deallocator_arg);
  delete[] dims_copy;
  if (dims != nullptr)
    env->ReleaseLongArrayElements(shape, dims, JNI_ABORT);
  if (tensor_handle == nullptr) {
    throw_exception(env, jvm_null_pointer_exception, "Unable to create new native Tensor.");
    return 0;
  }
  return reinterpret_cast<jlong>(tensor_handle);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_dataType(
    JNIEnv* env, jobject object, jlong handle) {
  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "\"TF_DataType\" in C cannot be represented as an \"Int\" in Scala.");
  TF_Tensor* tensor = require_graph_handle(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(TF_TensorType(tensor));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_shape(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Tensor* tensor = require_graph_handle(env, handle);
  if (tensor == nullptr) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t), "Scala \"Long\" is not compatible with the TensorFlow C API.");
  const jsize num_dims = TF_NumDims(tensor);
  jlongArray return_array = env->NewLongArray(num_dims);
  jlong* shape = env->GetLongArrayElements(return_array, nullptr);
  for (int i = 0; i < num_dims; ++i)
    shape[i] = static_cast<jlong>(TF_Dim(tensor, i));
  env->ReleaseLongArrayElements(return_array, shape, 0);
  return return_array;
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_buffer(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Tensor* tensor = require_graph_handle(env, handle);
  if (tensor == nullptr) return nullptr;
  void* data = TF_TensorData(tensor);
  const size_t byte_size = TF_TensorByteSize(tensor);
  return env->NewDirectByteBuffer(data, static_cast<jlong>(byte_size));
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_delete(
    JNIEnv* env, jobject object, jlong handle) {
  if (handle == 0) return;
  TF_DeleteTensor(reinterpret_cast<TF_Tensor*>(handle));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getEncodedStringSize(
    JNIEnv* env, jobject object, jint string_num_bytes) {
  return static_cast<jint>(TF_StringEncodedSize(static_cast<size_t>(string_num_bytes)));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_setStringBytes(
    JNIEnv* env, jobject object, jbyteArray string, jobject string_buffer) {
  char* dst_buffer = reinterpret_cast<char*>(env->GetDirectBufferAddress(string_buffer));
  size_t src_len = static_cast<size_t>(env->GetArrayLength(string));
  size_t dst_len = TF_StringEncodedSize(src_len);
  std::unique_ptr<char[]> buffer(new char[src_len]);
  env->GetByteArrayRegion(string, 0, static_cast<jsize>(src_len), reinterpret_cast<jbyte*>(buffer.get()));
  TF_Status* status = TF_NewStatus();
  size_t num_bytes_written = TF_StringEncode(buffer.get(), src_len, dst_buffer, dst_len, status);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  return static_cast<jsize>(num_bytes_written);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getStringBytes(
    JNIEnv* env, jobject object, jobject string_buffer) {
  char* src_buffer = reinterpret_cast<char*>(env->GetDirectBufferAddress(string_buffer));
  jbyteArray return_array = nullptr;
  const char* dst = nullptr;
  size_t dst_len = 0;
  TF_Status* status = TF_NewStatus();
  size_t src_len = static_cast<size_t>(env->GetDirectBufferCapacity(string_buffer));
  TF_StringDecode(src_buffer, src_len, &dst, &dst_len, status);
  if (throw_exception_if_not_ok(env, status)) {
    return_array = env->NewByteArray(static_cast<jsize>(dst_len));
    env->SetByteArrayRegion(return_array, 0, static_cast<jsize>(dst_len), reinterpret_cast<const jbyte*>(dst));
  }
  TF_DeleteStatus(status);
  return return_array;
}
