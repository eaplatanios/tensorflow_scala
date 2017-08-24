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
#include <cstring>
#include <memory>

#include "include/c_api.h"
#include "include/c_eager_api.h"
#include "include/exception_jni.h"
#include "include/utilities.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_allocate(
    JNIEnv* env, jobject object, jint data_type, jlongArray shape, jlong num_bytes) {
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
    JNIEnv* env, jobject object, jint data_type, jlongArray shape, jlong num_bytes, jobject buffer) {
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

//JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_fromBuffer(
//    JNIEnv* env, jobject object, jobject buffer, jint dtype, jlongArray shape, jlong sizeInBytes) {
//  int num_dims = static_cast<int>(env->GetArrayLength(shape));
//  jlong* dims = nullptr;
//  if (num_dims > 0) {
//    jboolean is_copy;
//    dims = env->GetLongArrayElements(shape, &is_copy);
//  }
//  static_assert(sizeof(jlong) == sizeof(int64_t), "Scala \"Long\" is not compatible with the TensorFlow C API.");
//  // On some platforms "jlong" is a "long" while "int64_t" is a "long long".
//  //
//  // Thus, static_cast<int64_t*>(dims) will trigger a compiler error:
//  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
//  // *') is not allowed
//  //
//  // Since this array is typically very small, use the guaranteed safe scheme of
//  // creating a copy.
//  int64_t* dims_copy = new int64_t[num_dims];
//  for (int i = 0; i < num_dims; ++i)
//    dims_copy[i] = static_cast<int64_t>(dims[i]);
//  // Notifying the JVM of the existence of this reference to the byte buffer, to avoid garbage collection.
//  // More details can be found here: http://docs.oracle.com/javase/6/docs/technotes/guides/jni/spec/design.html#wp1242
//  jobject buffer_ref = env->NewGlobalRef(buffer);
//  std::pair<JNIEnv*, jobject>* deallocator_arg = new std::pair<JNIEnv*, jobject>(env, buffer_ref);
//  TF_Tensor* tensor_handle = TF_NewTensor(
//      static_cast<TF_DataType>(dtype), dims_copy, num_dims,
//      env->GetDirectBufferAddress(buffer), static_cast<size_t>(sizeInBytes),
//      [](void* data, size_t len, void* arg) {
//        std::pair<JNIEnv*, jobject>* pair = reinterpret_cast<std::pair<JNIEnv*, jobject>*>(arg);
//        pair->first->DeleteGlobalRef(reinterpret_cast<jobject>(pair->second));
//        delete (pair);
//      }, deallocator_arg);
//  delete[] dims_copy;
//  if (dims != nullptr)
//    env->ReleaseLongArrayElements(shape, dims, JNI_ABORT);
//  if (tensor_handle == nullptr) {
//    throw_exception(env, jvm_null_pointer_exception, "Unable to create new native Tensor.");
//    return 0;
//  }
//  return reinterpret_cast<jlong>(tensor_handle);
//}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_dataType(
    JNIEnv* env, jobject object, jlong handle) {
  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "\"TF_DataType\" in C cannot be represented as an \"Int\" in Scala.");
  TF_Tensor* tensor = require_handle<TF_Tensor>(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(TF_TensorType(tensor));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_shape(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Tensor* tensor = require_handle<TF_Tensor>(env, handle);
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
  TF_Tensor* tensor = require_handle<TF_Tensor>(env, handle);
  if (tensor == nullptr) return nullptr;
  void* data = TF_TensorData(tensor);
  const size_t byte_size = TF_TensorByteSize(tensor);
  return env->NewDirectByteBuffer(data, static_cast<jlong>(byte_size));
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_delete(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Tensor* tensor = require_handle<TF_Tensor>(env, handle);
  if (tensor == nullptr) return;
  TF_DeleteTensor(tensor);
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
  // jbyte is a signed char, while the C standard doesn't require char and
  // signed char to be the same. As a result, static_cast<char*>(src) will
  // complain. We copy the string instead.
  jbyte* src = env->GetByteArrayElements(string, nullptr);
  static_assert(sizeof(jbyte) == sizeof(char), "Cannot convert Java byte to a C char.");
  std::memcpy(buffer.get(), src, src_len);
  env->ReleaseByteArrayElements(string, src, JNI_ABORT);
  // env->GetByteArrayRegion(string, 0, static_cast<jsize>(src_len), reinterpret_cast<jbyte*>(buffer.get()));
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

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocateContext(
    JNIEnv* env, jobject object) {
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* options = TF_NewSessionOptions();
  TFE_Context* context = TFE_NewContext(options, status);
  TF_DeleteSessionOptions(options);
  bool ok = throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  return ok ? reinterpret_cast<jlong>(context) : 0;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDeleteContext(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_Context* context = require_handle<TFE_Context>(env, handle);
  if (context == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TFE_DeleteContext(context, status);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocate(
    JNIEnv* env, jobject object, jlong tensor_handle) {
  TF_Tensor* tensor = require_handle<TF_Tensor>(env, tensor_handle);
  if (tensor == nullptr) return 0;
  TFE_TensorHandle* eager_tensor = TFE_NewTensorHandle(tensor);
  return reinterpret_cast<jlong>(eager_tensor);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDataType(
    JNIEnv* env, jobject object, jlong handle) {
  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "\"TF_DataType\" in C cannot be represented as an \"Int\" in Scala.");
  TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(TFE_TensorHandleDataType(tensor));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerShape(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, handle);
  if (tensor == nullptr) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t), "Scala \"Long\" is not compatible with the TensorFlow C API.");
  const jsize num_dims = TFE_TensorHandleNumDims(tensor);
  jlongArray return_array = env->NewLongArray(num_dims);
  jlong* shape = env->GetLongArrayElements(return_array, nullptr);
  for (int i = 0; i < num_dims; ++i)
    shape[i] = static_cast<jlong>(TFE_TensorHandleDim(tensor, i));
  env->ReleaseLongArrayElements(return_array, shape, 0);
  return return_array;
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDevice(
    JNIEnv* env,  jobject object,  jlong handle) {
  TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, handle);
  if (tensor == nullptr) return nullptr;
  return env->NewStringUTF(TFE_TensorHandleDeviceName(tensor));
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDelete(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_TensorHandle* eager_tensor = require_handle<TFE_TensorHandle>(env, handle);
  if (eager_tensor == nullptr) return;
  TFE_DeleteTensorHandle(eager_tensor);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerResolve(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_TensorHandle* eager_tensor = require_handle<TFE_TensorHandle>(env, handle);
  if (eager_tensor == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  TF_Tensor* tensor = TFE_TensorHandleResolve(eager_tensor, status);
  bool ok = throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  return ok ? reinterpret_cast<jlong>(tensor) : 0;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerCopyToDevice(
    JNIEnv* env,  jobject object,  jlong tensor_handle, jlong context_handle, jstring device) {
  TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, tensor_handle);
  TFE_Context* context = require_handle<TFE_Context>(env, context_handle);
  if (tensor == nullptr || context == nullptr) return 0;
  const char* c_device = env->GetStringUTFChars(device, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* eager_tensor = TFE_TensorHandleCopyToDevice(tensor, context, c_device, status);
  env->ReleaseStringUTFChars(device, c_device);
  bool ok = throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  return ok ? reinterpret_cast<jlong>(eager_tensor) : 0;
}

//JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerExecuteOp(
//    JNIEnv* env,  jobject object,  jlong context_handle, jstring op_type, jstring device, jlongArray input_handles,
//    jstringArray string_attr_names, jstringArray string_attr_values) {
//  std::vector<std::function<void()>> deallocators;
//
//  auto deallocate = [&] {
//    for (std::function<void()> &deallocator : deallocators) {
//      deallocator->();
//    }
//  };
//
//  // Allocate op.
//  TFE_Context* context = require_handle<TFE_Context>(env, context_handle);
//  if (context == nullptr) return 0;
//  const char* c_op_type = env->GetStringUTFChars(op_type, nullptr);
//  deallocators.push_back([&] { env->ReleaseStringUTFChars(op_type, c_op_type); });
//  TF_Status* status = TF_NewStatus();
//  deallocators.push_back([&] { TF_DeleteStatus(status); });
//  TFE_Op* op = TFE_NewOp(context, c_op_type, status);
//  if (!throw_exception_if_not_ok(env, status)) {
//    deallocate();
//    return nullptr;
//  }
//  deallocators.push_back([&] { TFE_DeleteOp(op); });
//
//  // Set the op device.
//  char* c_device = nullptr;
//  if (device != NULL) {
//    c_device = env->GetStringUTFChars(device, nullptr);
//    status = TF_NewStatus();
//    TFE_OpSetDevice(op, context, const_cast<const char*>(c_device), status);
//    if (!throw_exception_if_not_ok(env, status)) {
//      deallocate();
//      return nullptr;
//    }
//    deallocators.push_back([&] { env->ReleaseStringUTFChars(device, c_device); });
//  }
//
//  // Add all inputs.
//  if (input_handles != nullptr) {
//    const int num_inputs = env->GetArrayLength(input_handles);
//    jlong* i_handles = env->GetLongArrayElements(input_handles, nullptr);
//    deallocators.push_back([&] { env->ReleaseLongArrayElements(input_handles, i_handles, JNI_ABORT); });
//    for (int i = 0; i < num_inputs; ++i) {
//      TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, i_handles[i]);
//      if (tensor == nullptr) return nullptr;
//      TFE_OpAddInput(op, tensor, status);
//    }
//  }
//
//  // Set all string-valued attributes.
//  if (string_attr_names != nullptr) {
//    const int num_string_attrs = env->GetArrayLength(string_attr_names);
//    jstring* s_attr_names = env->GetStringArrayElements(string_attr_names, nullptr);
//    jstring* s_attr_values = env->GetStringArrayElements(string_attr_values, nullptr);
//    const char** s_attr_c_names = new const char*[num_string_attrs];
//    jbyte** s_attr_c_values = new jbyte*[num_string_attrs];
//    for (int i = 0; i < num_string_attrs; ++i) {
//      s_attr_c_names[i] = env->GetStringUTFChars(name, nullptr);
//      s_attr_c_values[i] = env->GetByteArrayElements(s_attr_values[i], nullptr);
//      TFE_OpSetAttrString(op, s_attr_c_names[i], reinterpret_cast<const char*>(s_attr_c_values[i]));
//    }
//  }
//
//  // Set all string-list-valued attributes.
//  const int num_string_list_attrs = env->GetArrayLength(string_list_attr_names);
//  jstring* sl_attr_names = env->GetStringArrayElements(string_list_attr_names, nullptr);
//  jobject* sl_attr_values = env->GetObjectArrayElements(string_list_attr_values, nullptr);
//  const char** sl_attr_c_names = new const char*[num_string_list_attrs];
//  jbyte** sl_attr_c_values = new jbyte*[num_string_list_attrs];
//  for (int i = 0; i < num_string_attrs; ++i) {
//    s_attr_c_names[i] = env->GetStringUTFChars(name, nullptr);
//    s_attr_c_values[i] = env->GetByteArrayElements(s_attr_values[i], nullptr);
//    TFE_OpSetAttrString(op, s_attr_c_names[i], reinterpret_cast<const char*>(s_attr_c_values[i]));
//  }
//
//
//  const char *c_name = env->GetStringUTFChars(name, nullptr);
//    int num_strings = env->GetArrayLength(values);
//    size_t *lengths = new size_t[num_strings];
//    jbyte **strings = new jbyte *[num_strings];
//    for (int i = 0; i < num_strings; i++) {
//      jbyteArray value = (jbyteArray) env->GetObjectArrayElement(values, i);
//      lengths[i] = static_cast<size_t>(env->GetArrayLength(value));
//      strings[i] = env->GetByteArrayElements(value, nullptr);
//      // TODO: We do not release the array elements because we assume the arrays will be small enough.
//    }
//    TFE_OpSetAttrStringList(op, c_name, const_cast<const char **>(reinterpret_cast<char **>(strings)), num_strings);
//    env->ReleaseStringUTFChars(name, c_name);
//
//  // Release all resources.
//  deallocate();
//}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocateOp(
    JNIEnv* env,  jobject object,  jlong context_handle, jstring op_type) {
  TFE_Context* context = require_handle<TFE_Context>(env, context_handle);
  if (context == nullptr) return 0;
  const char* c_op_type = env->GetStringUTFChars(op_type, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(context, c_op_type, status);
  env->ReleaseStringUTFChars(op_type, c_op_type);
  bool ok = throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  return ok ? reinterpret_cast<jlong>(op) : 0;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDeleteOp(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return;
  TFE_DeleteOp(op);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpDevice(
    JNIEnv* env, jobject object, jlong op_handle, jlong context_handle, jstring device) {
  TFE_Op* op = require_handle<TFE_Op>(env, op_handle);
  TFE_Context* context = require_handle<TFE_Context>(env, context_handle);
  if (op == nullptr || context == nullptr) return;
  const char* c_device = env->GetStringUTFChars(device, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetDevice(op, context, c_device, status);
  env->ReleaseStringUTFChars(device, c_device);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerOpAddInput(
    JNIEnv* env, jobject object, jlong handle, jlong tensor_handle) {
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  TFE_TensorHandle* tensor = require_handle<TFE_TensorHandle>(env, tensor_handle);
  if (op == nullptr || tensor == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TFE_OpAddInput(op, tensor, status);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttrString(
    JNIEnv* env, jobject object, jlong handle, jstring name, jbyteArray value) {
  static_assert(sizeof(jbyte) == 1, "Require Java byte to be represented as a single byte.");
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  jbyte *c_value = env->GetByteArrayElements(value, nullptr);
  TFE_OpSetAttrString(op, c_name, reinterpret_cast<const char *>(c_value));
  env->ReleaseByteArrayElements(value, c_value, JNI_ABORT);
  env->ReleaseStringUTFChars(name, c_name);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttrStringList(
        JNIEnv* env, jobject object, jlong handle, jstring name, jobjectArray values) {
  static_assert(sizeof(jbyte) == 1, "Require Java byte to be represented as a single byte.");
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  int num_strings = env->GetArrayLength(values);
  size_t *lengths = new size_t[num_strings];
  jbyte **strings = new jbyte *[num_strings];
  for (int i = 0; i < num_strings; i++) {
    jbyteArray value = (jbyteArray) env->GetObjectArrayElement(values, i);
    lengths[i] = static_cast<size_t>(env->GetArrayLength(value));
    strings[i] = env->GetByteArrayElements(value, nullptr);
    // TODO: We do not release the array elements because we assume the arrays will be small enough.
  }
  TFE_OpSetAttrStringList(op, c_name, const_cast<const char **>(reinterpret_cast<char **>(strings)), num_strings);
  env->ReleaseStringUTFChars(name, c_name);
}

#define DEFINE_SET_ATTR_SCALAR(atype, jtype, ctype)                                            \
  JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttr##atype( \
      JNIEnv* env, jobject object, jlong handle, jstring name, jtype value) {                  \
    static_assert(                                                                             \
        sizeof(ctype) >= sizeof(jtype),                                                        \
        "Information loss when converting between Java and C types");                          \
    TFE_Op* op = require_handle<TFE_Op>(env, handle);                                          \
    if (op == nullptr) return;                                                                 \
    const char* c_name = env->GetStringUTFChars(name, nullptr);                                \
    TFE_OpSetAttr##atype(op, c_name, static_cast<ctype>(value));                               \
    env->ReleaseStringUTFChars(name, c_name);                                                  \
  }

#define DEFINE_SET_ATTR_LIST(atype, jname, jtype, ctype)                           \
  JNIEXPORT void JNICALL                                                           \
      Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttr##atype##List(  \
          JNIEnv* env, jobject object, jlong handle, jstring name,                 \
          jtype##Array value) {                                                    \
    TFE_Op* op = require_handle<TFE_Op>(env, handle);                              \
    if (op == nullptr) return;                                                     \
    const char *c_name = env->GetStringUTFChars(name, nullptr);                    \
    /* Make a copy of the array to paper over any differences */                   \
    /* in byte representations of the jtype and ctype         */                   \
    /* For example, jint vs TF_DataType.                      */                   \
    /* If this copy turns out to be a problem in practice     */                   \
    /* can avoid it for many types.                           */                   \
    const int n = env->GetArrayLength(value);                                      \
    std::unique_ptr<ctype[]> c_value(new ctype[n]);                                \
    jtype* elems = env->Get##jname##ArrayElements(value, nullptr);                 \
    for (int i = 0; i < n; ++i) {                                                  \
      c_value[i] = static_cast<ctype>(elems[i]);                                   \
    }                                                                              \
    TFE_OpSetAttr##atype##List(op, c_name, c_value.get(), n);                      \
    env->Release##jname##ArrayElements(value, elems, JNI_ABORT);                   \
    env->ReleaseStringUTFChars(name, c_name);                                      \
  }

#define DEFINE_SET_ATTR(atype, jname, jtype, ctype) \
  DEFINE_SET_ATTR_SCALAR(atype, jtype, ctype)       \
  DEFINE_SET_ATTR_LIST(atype, jname, jtype, ctype)

DEFINE_SET_ATTR(Int, Long, jlong, int64_t);
DEFINE_SET_ATTR(Float, Float, jfloat, float);
DEFINE_SET_ATTR(Bool, Boolean, jboolean, unsigned char);
DEFINE_SET_ATTR(Type, Int, jint, TF_DataType);
#undef DEFINE_SET_ATTR
#undef DEFINE_SET_ATTR_LIST
#undef DEFINE_SET_ATTR_SCALAR

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttrShape(
    JNIEnv* env, jobject object, jlong handle, jstring name, jlongArray shape,
    jint num_dims) {
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return;
  std::unique_ptr<int64_t[]> c_value;
  // num_dims and env->GetArrayLength(shape) are assumed to be consistent.
  // i.e., either num_dims < 0 or num_dims == env->GetArrayLength(shape).
  if (num_dims > 0) {
    c_value.reset(new int64_t[num_dims]);
    jlong *elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      c_value[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetAttrShape(op, c_name, c_value.get(), static_cast<int>(num_dims), status);
  env->ReleaseStringUTFChars(name, c_name);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpAttrShapeList(
    JNIEnv* env, jobject object, jlong handle, jstring name, jobjectArray shapes, jintArray num_dims, jint num_shapes) {
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return;
  std::unique_ptr<int[]> c_num_dims;
  std::unique_ptr<int64_t*[]> c_shapes;
  int c_num_shapes = static_cast<int>(num_shapes);
  // num_dims[i] and env->GetArrayLength(shapes[i]) are assumed to be consistent.
  // i.e., either num_dims[i] < 0 or num_dims[i] == env->GetArrayLength(shapes[i]).
  if (c_num_shapes > 0) {
    c_num_dims.reset(new int[c_num_shapes]);
    c_shapes.reset(new int64_t*[c_num_shapes]);
    jint *num_dims_elems = env->GetIntArrayElements(num_dims, nullptr);
    for (int j = 0; j < c_num_shapes; ++j) {
      c_num_dims[j] = static_cast<int>(num_dims_elems[j]);
      if (c_num_dims[j] > -1) {
        c_shapes[j] = new int64_t[c_num_dims[j]];
        jlongArray shapes_elems = (jlongArray) env->GetObjectArrayElement(shapes, j);
        jlong *shape_elems = env->GetLongArrayElements(shapes_elems, nullptr);
        for (int i = 0; i < c_num_dims[j]; ++i) {
          c_shapes[j][i] = static_cast<int64_t>(shape_elems[i]);
        }
        env->ReleaseLongArrayElements(shapes_elems, shape_elems, JNI_ABORT);
      } else {
        c_shapes[j] = new int64_t[0];
      }
    }
    env->ReleaseIntArrayElements(num_dims, num_dims_elems, JNI_ABORT);
  }
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetAttrShapeList(
    op, c_name, const_cast<const int64_t **>(c_shapes.get()), c_num_dims.get(), c_num_shapes, status);
  env->ReleaseStringUTFChars(name, c_name);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerExecuteOp(
    JNIEnv* env, jobject object, jlong handle) {
  TFE_Op* op = require_handle<TFE_Op>(env, handle);
  if (op == nullptr) return nullptr;
  std::unique_ptr<TFE_TensorHandle*[]> outputs(new TFE_TensorHandle*[1]);
  std::unique_ptr<int[]> num_outputs(new int[1]);
  TF_Status* status = TF_NewStatus();
  TFE_Execute(op, outputs.get(), num_outputs.get(), status);
  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);
  jlongArray output_tensor_handles = env->NewLongArray(num_outputs[0]);
  jlong* output_tensor_handles_array = env->GetLongArrayElements(output_tensor_handles, nullptr);
  for (int i = 0; i < num_outputs[0]; ++i)
    output_tensor_handles_array[i] = reinterpret_cast<jlong>(outputs[i]);
  env->ReleaseLongArrayElements(output_tensor_handles, output_tensor_handles_array, 0);
  return output_tensor_handles;
}

TFE_TensorHandle* TestStridesTensorHandle() {
  int64_t dims[] = {2};
  int data[] = {1, 1};
  TF_Tensor* t = TF_AllocateTensor(
      TF_INT32, &dims[0], sizeof(dims) / sizeof(int64_t), sizeof(data));
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandle(t);
  TF_DeleteTensor(t);
  return th;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_cast(
    JNIEnv* env, jobject object, jlong context_handle, jlong tensor_handle, jint data_type) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context, "Cast", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(tensor, TFE_TensorHandle, tensor_handle, 0);
  TFE_OpAddInput(op.get(), tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetAttrType(op.get(), "SrcT", TFE_TensorHandleDataType(tensor));
  TFE_OpSetAttrType(op.get(), "DstT", static_cast<TF_DataType>(data_type));

  std::unique_ptr<TFE_TensorHandle*[]> outputs(new TFE_TensorHandle*[1]);
  std::unique_ptr<int[]> num_outputs(new int[1] {1});
  TFE_Execute(op.get(), outputs.get(), num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_pack(
    JNIEnv* env, jobject object, jlong context_handle, jlongArray tensor_handles, jlong axis) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context, "Pack", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);

  const int num_tensors = env->GetArrayLength(tensor_handles);
  jlong *tensor_handle_elems = env->GetLongArrayElements(tensor_handles, nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    REQUIRE_HANDLE(tensor, TFE_TensorHandle, tensor_handle_elems[i], 0);
    TFE_OpAddInput(op.get(), tensor, status.get());
    CHECK_STATUS(env, status.get(), 0);
    if (i == 0) TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(tensor));
  }
  env->ReleaseLongArrayElements(tensor_handles, tensor_handle_elems, JNI_ABORT);

  TFE_OpSetAttrInt(op.get(), "N", static_cast<int64_t>(num_tensors));
  TFE_OpSetAttrInt(op.get(), "axis", static_cast<int64_t>(axis));

  std::unique_ptr<TFE_TensorHandle*[]> outputs(new TFE_TensorHandle*[1]);
  std::unique_ptr<int[]> num_outputs(new int[1] {1});
  TFE_Execute(op.get(), outputs.get(), num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_stridedSlice(
    JNIEnv* env, jobject object, jlong context_handle, jlong tensor_handle, jlong begin_tensor_handle,
    jlong end_tensor_handle, jlong strides_tensor_handle, jlong begin_mask, jlong end_mask, jlong ellipsis_mask,
    jlong new_axis_mask, jlong shrink_axis_mask) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context, "StridedSlice", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(tensor, TFE_TensorHandle, tensor_handle, 0);
  TFE_OpAddInput(op.get(), tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(begin_tensor, TFE_TensorHandle, begin_tensor_handle, 0);
  TFE_OpAddInput(op.get(), begin_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(end_tensor, TFE_TensorHandle, end_tensor_handle, 0);
  TFE_OpAddInput(op.get(), end_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(strides_tensor, TFE_TensorHandle, strides_tensor_handle, 0);
  TFE_OpAddInput(op.get(), strides_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);

  TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(tensor));
  TFE_OpSetAttrType(op.get(), "Index", TFE_TensorHandleDataType(begin_tensor));

  TFE_OpSetAttrInt(op.get(), "begin_mask", static_cast<int64_t>(begin_mask));
  TFE_OpSetAttrInt(op.get(), "end_mask", static_cast<int64_t>(end_mask));
  TFE_OpSetAttrInt(op.get(), "ellipsis_mask", static_cast<int64_t>(ellipsis_mask));
  TFE_OpSetAttrInt(op.get(), "new_axis_mask", static_cast<int64_t>(new_axis_mask));
  TFE_OpSetAttrInt(op.get(), "shrink_axis_mask", static_cast<int64_t>(shrink_axis_mask));

  std::unique_ptr<TFE_TensorHandle*[]> outputs(new TFE_TensorHandle*[1]);
  std::unique_ptr<int[]> num_outputs(new int[1] {1});
  TFE_Execute(op.get(), outputs.get(), num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_reshape(
    JNIEnv* env, jobject object, jlong context_handle, jlong tensor_handle, jlong shape_tensor_handle) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context, "Reshape", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(tensor, TFE_TensorHandle, tensor_handle, 0);
  TFE_OpAddInput(op.get(), tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  REQUIRE_HANDLE(shape_tensor, TFE_TensorHandle, shape_tensor_handle, 0);
  TFE_OpAddInput(op.get(), shape_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(tensor));
  TFE_OpSetAttrType(op.get(), "Tshape", TFE_TensorHandleDataType(shape_tensor));

  std::unique_ptr<TFE_TensorHandle*[]> outputs(new TFE_TensorHandle*[1]);
  std::unique_ptr<int[]> num_outputs(new int[1] {1});
  TFE_Execute(op.get(), outputs.get(), num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_add(
    JNIEnv* env, jobject object, jlong context_handle, jlong tensor1_handle, jlong tensor2_handle) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context, "Add", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(tensor1, TFE_TensorHandle, tensor1_handle, 0);
  TFE_OpAddInput(op.get(), tensor1, status.get());
  CHECK_STATUS(env, status.get(), 0);
  REQUIRE_HANDLE(tensor2, TFE_TensorHandle, tensor2_handle, 0);
  TFE_OpAddInput(op.get(), tensor2, status.get());
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(tensor1));

  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [1]);
  std::unique_ptr<int[]> num_outputs(new int[1] {1});
  TFE_Execute(op.get(), outputs.get(), num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(outputs[0]);
}
