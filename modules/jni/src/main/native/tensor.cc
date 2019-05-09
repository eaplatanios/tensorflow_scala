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
#include "tensor.h"
#include "utilities.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

//region TensorFlow C Tensors

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_allocate(
  JNIEnv* env,
  jobject object,
  jint data_type,
  jlongArray shape,
  jlong num_bytes
) {
  TF_DataType dtype = static_cast<TF_DataType>(data_type);
  const int num_dims = env->GetArrayLength(shape);
  std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
  if (num_dims > 0) {
    jlong *shape_elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i)
      dims[i] = static_cast<int64_t>(shape_elems[i]);
    env->ReleaseLongArrayElements(shape, shape_elems, JNI_ABORT);
  }
  TF_Tensor* tensor = TF_AllocateTensor(dtype, dims.get(), num_dims, static_cast<size_t>(num_bytes));
  return reinterpret_cast<jlong>(tensor);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_fromBuffer(
  JNIEnv* env,
  jobject object,
  jint data_type,
  jlongArray shape,
  jlong num_bytes,
  jobject buffer
) {
  TF_DataType dtype = static_cast<TF_DataType>(data_type);
  const int num_dims = env->GetArrayLength(shape);
  std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
  if (num_dims > 0) {
    jlong *shape_elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i)
      dims[i] = static_cast<int64_t>(shape_elems[i]);
    env->ReleaseLongArrayElements(shape, shape_elems, JNI_ABORT);
  }
  size_t c_num_bytes = static_cast<size_t>(num_bytes);
  TF_Tensor* tensor = TF_AllocateTensor(dtype, dims.get(), num_dims, c_num_bytes);
  memcpy(TF_TensorData(tensor), env->GetDirectBufferAddress(buffer), c_num_bytes);
  return reinterpret_cast<jlong>(tensor);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_dataType(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "\"TF_DataType\" in C cannot be represented as an \"Int\" in Scala.");
  REQUIRE_HANDLE(tensor, TF_Tensor, handle, 0);
  return static_cast<jint>(TF_TensorType(tensor));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_shape(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(tensor, TF_Tensor, handle, nullptr);
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
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(tensor, TF_Tensor, handle, nullptr);
  void* data = TF_TensorData(tensor);
  const size_t byte_size = TF_TensorByteSize(tensor);
  return env->NewDirectByteBuffer(data, static_cast<jlong>(byte_size));
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_delete(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(tensor, TF_Tensor, handle, void());
  TF_DeleteTensor(tensor);
}

//endregion TensorFlow C Tensors

//region TensorFlow Eager Tensors

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocateContext(
  JNIEnv* env,
  jobject object,
  jbyteArray config_proto
) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  TFE_ContextOptions* options = TFE_NewContextOptions();

  // Set the configuration proto, if one has been provided.
  jbyte* c_config_proto;
  if (config_proto != nullptr) {
    c_config_proto = env->GetByteArrayElements(config_proto, nullptr);
    TFE_ContextOptionsSetConfig(
      options, c_config_proto, static_cast<size_t>(env->GetArrayLength(config_proto)), status.get());
    CHECK_STATUS(env, status.get(), 0);
  }

  TFE_ContextOptionsSetAsync(options, 0);
  TFE_Context* context = TFE_NewContext(options, status.get());

  TFE_DeleteContextOptions(options);
  if (config_proto != nullptr) {
    env->ReleaseByteArrayElements(config_proto, c_config_proto, JNI_ABORT);
  }

  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDeleteContext(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(context, TFE_Context, handle, void());
  TFE_DeleteContext(context);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocate(
  JNIEnv* env,
  jobject object,
  jlong tensor_handle
) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  REQUIRE_HANDLE(tensor, TF_Tensor, tensor_handle, 0);
  TFE_TensorHandle* eager_tensor = TFE_NewTensorHandle(tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(eager_tensor);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDataType(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "\"TF_DataType\" in C cannot be represented as an \"Int\" in Scala.");
  REQUIRE_HANDLE(tensor, TFE_TensorHandle, handle, 0);
  return static_cast<jint>(TFE_TensorHandleDataType(tensor));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerShape(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(tensor, TFE_TensorHandle, handle, nullptr);
  static_assert(sizeof(jlong) == sizeof(int64_t), "Scala \"Long\" is not compatible with the TensorFlow C API.");
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  const jsize num_dims = TFE_TensorHandleNumDims(tensor, status.get());
  CHECK_STATUS(env, status.get(), nullptr);
  jlongArray return_array = env->NewLongArray(num_dims);
  jlong* shape = env->GetLongArrayElements(return_array, nullptr);
  for (int i = 0; i < num_dims; ++i) {
    shape[i] = static_cast<jlong>(TFE_TensorHandleDim(tensor, i, status.get()));
    CHECK_STATUS(env, status.get(), nullptr);
  }
  env->ReleaseLongArrayElements(return_array, shape, 0);
  return return_array;
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDevice(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(tensor, TFE_TensorHandle, handle, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  jstring device = env->NewStringUTF(TFE_TensorHandleDeviceName(tensor, status.get()));
  CHECK_STATUS(env, status.get(), nullptr);
  return device;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDelete(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(eager_tensor, TFE_TensorHandle, handle, void());
  TFE_DeleteTensorHandle(eager_tensor);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerResolve(
  JNIEnv* env,
  jobject object,
  jlong handle
) {
  REQUIRE_HANDLE(eager_tensor, TFE_TensorHandle, handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* tensor = TFE_TensorHandleResolve(eager_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(tensor);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerCopyToDevice(
  JNIEnv* env,
  jobject object,
  jlong tensor_handle,
  jlong context_handle,
  jstring device
) {
  REQUIRE_HANDLE(tensor, TFE_TensorHandle, tensor_handle, 0);
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  const char* c_device = env->GetStringUTFChars(device, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* eager_tensor = TFE_TensorHandleCopyToDevice(tensor, context, c_device, status.get());
  env->ReleaseStringUTFChars(device, c_device);
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(eager_tensor);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerNewOp(
  JNIEnv* env,
  jobject object,
  jlong context_handle,
  jstring op_or_function_name
) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  const char* c_op_or_function_name = env->GetStringUTFChars(op_or_function_name, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TFE_Op* op = TFE_NewOp(context, c_op_or_function_name, status.get());
  env->ReleaseStringUTFChars(op_or_function_name, c_op_or_function_name);
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDeleteOp(
  JNIEnv* env,
  jobject object,
  jlong op_handle
) {
  REQUIRE_HANDLE(op, TFE_Op, op_handle, void());
  TFE_DeleteOp(op);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpDevice(
  JNIEnv* env,
  jobject object,
  jlong op_handle,
  jstring device
) {
  REQUIRE_HANDLE(op, TFE_Op, op_handle, void());
  const char* c_device = env->GetStringUTFChars(device, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TFE_OpSetDevice(op, c_device, status.get());
  env->ReleaseStringUTFChars(device, c_device);
  CHECK_STATUS(env, status.get(), void());
}

//endregion TensorFlow Eager Tensors

//region String Helpers

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_setStringBytes(
  JNIEnv* env,
  jobject object,
  jbyteArray string,
  jobject string_buffer
) {
  char* dst_buffer = reinterpret_cast<char*>(env->GetDirectBufferAddress(string_buffer));
  size_t src_len = static_cast<size_t>(env->GetArrayLength(string));
  size_t dst_len = TF_StringEncodedSize(src_len);
  std::unique_ptr<char[]> buffer(new char[src_len]);
  // jbyte is a signed char, while the C standard doesn't require char and
  // signed char to be the same. As a result, static_cast<char*>(src) will
  // complain. We copy the string instead.
  jbyte* src = env->GetByteArrayElements(string, nullptr);
  static_assert(sizeof(jbyte) == sizeof(char), "Cannot convert Java byte to a C char.");
  std::memcpy(buffer.get(), src, src_len);
  env->ReleaseByteArrayElements(string, src, JNI_ABORT);
  // env->GetByteArrayRegion(string, 0, static_cast<jsize>(src_len), reinterpret_cast<jbyte*>(buffer.get()));
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  size_t num_bytes_written = TF_StringEncode(buffer.get(), src_len, dst_buffer, dst_len, status.get());
  CHECK_STATUS(env, status.get(), 0);
  return static_cast<jsize>(num_bytes_written);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getStringBytes(
  JNIEnv* env,
  jobject object,
  jobject string_buffer
) {
  jbyteArray return_array = nullptr;

  // Call TF_StringDecode(...).
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  char* src_buffer = reinterpret_cast<char*>(env->GetDirectBufferAddress(string_buffer));
  size_t src_len = static_cast<size_t>(env->GetDirectBufferCapacity(string_buffer));
  const char* dst = nullptr;
  size_t dst_len = 0;
  TF_StringDecode(src_buffer, src_len, &dst, &dst_len, status.get());

  if (throw_exception_if_not_ok(env, status.get())) {
    return_array = env->NewByteArray(static_cast<jsize>(dst_len));
    env->SetByteArrayRegion(return_array, 0, static_cast<jsize>(dst_len), reinterpret_cast<const jbyte*>(dst));
  }
  return return_array;
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getEncodedStringSize(
  JNIEnv* env,
  jobject object,
  jint string_num_bytes
) {
  return static_cast<jint>(TF_StringEncodedSize(static_cast<size_t>(string_num_bytes)));
}

//endregion String Helpers
