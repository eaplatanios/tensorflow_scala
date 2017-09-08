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

#ifndef TENSORFLOW_JNI_UTILITIES_H_
#define TENSORFLOW_JNI_UTILITIES_H_

#include <sstream>

#include "exception.h"

namespace {
  template <typename T>
  std::string pointerToString(T pointer, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr) {
    std::stringstream ss;
    ss << pointer;
    return ss.str();
  }

  template <typename T>
  T pointerFromString(const std::string &text, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr) {
    std::stringstream ss(text);
    void* pointer;
    ss >> pointer;
    return (T) pointer;
  }

  template <class T>
  inline T* require_handle(JNIEnv* env, jlong handle, const char* object_name) {
    static_assert(sizeof(jlong) >= sizeof(T*), "Cannot package C object pointers as a Java long");
    if (handle == 0) {
      std::stringstream msg;
      msg << "Object '" << object_name << "' has been disposed already.";
      throw_exception(env, jvm_null_pointer_exception, msg.str().c_str());
      return nullptr;
    }
    return reinterpret_cast<T*>(handle);
  }
  
  template<class T>
  inline void require_handles(JNIEnv* env, jlongArray src_array, T** dst_array, jint src_array_length) {
    jint len = env->GetArrayLength(src_array);
    if (len != src_array_length) {
      std::stringstream msg;
      msg << "Expected " << src_array_length << " handles, but got " << len << " handles, instead.";
      throw_exception(env, jvm_illegal_argument_exception, msg.str().c_str());
      return;
    }
    jlong* src_start = env->GetLongArrayElements(src_array, nullptr);
    jlong* src = src_start;
    for (int i = 0; i < src_array_length; ++i, ++src, ++dst_array) {
      if (*src == 0) {
        std::stringstream msg;
        msg << "Invalid handle (# " << i << " of " << src_array_length << ").";
        throw_exception(env, jvm_null_pointer_exception, msg.str().c_str());
        return;
      }
      *dst_array = reinterpret_cast<T*>(*src);
    }
    env->ReleaseLongArrayElements(src_array, src_start, JNI_ABORT);
  }

  inline void require_outputs(
      JNIEnv* env, jlongArray src_ops, jintArray src_indices, TF_Output* dst_array, jint src_ops_length) {
    jint len = env->GetArrayLength(src_ops);
    if (len != src_ops_length) {
      std::stringstream msg;
      msg << "Expected " << src_ops_length << " ops, but got " << len << ", instead.";
      throw_exception(env, jvm_illegal_argument_exception, msg.str().c_str());
      return;
    }
    len = env->GetArrayLength(src_indices);
    if (len != src_ops_length) {
      std::stringstream msg;
      msg << "Expected " << src_ops_length << " op output indices, but got " << len << ", instead.";
      throw_exception(env, jvm_illegal_argument_exception, msg.str().c_str());
      return;
    }
    jlong* op_handles = env->GetLongArrayElements(src_ops, nullptr);
    jint* indices = env->GetIntArrayElements(src_indices, nullptr);
    for (int i = 0; i < src_ops_length; ++i) {
      if (op_handles[i] == 0) {
        std::stringstream msg;
        msg << "Invalid op handle (# " << i << " of " << src_ops_length << ").";
        throw_exception(env, jvm_null_pointer_exception, msg.str().c_str());
        return;
      }
      dst_array[i] = TF_Output{reinterpret_cast<TF_Operation*>(op_handles[i]), static_cast<int>(indices[i])};
    }
    env->ReleaseIntArrayElements(src_indices, indices, JNI_ABORT);
    env->ReleaseLongArrayElements(src_ops, op_handles, JNI_ABORT);
  }
}  // namespace

#define REQUIRE_HANDLE(name, type, variable_name, null_return_value)       \
  type* name = require_handle<type>(env, variable_name, #variable_name);   \
  if (name == nullptr) return null_return_value;

#define REQUIRE_HANDLES(src_array, dst_array, src_array_length, null_return_value)   \
  require_handles(env, src_array, dst_array, src_array_length);                      \
  if (env->ExceptionCheck()) return null_return_value;

#define REQUIRE_OUTPUTS(src_ops, src_indices, dst_array, src_ops_length, null_return_value)   \
  require_outputs(env, src_ops, src_indices, dst_array, src_ops_length);                      \
  if (env->ExceptionCheck()) return null_return_value;

#endif  // TENSORFLOW_JNI_UTILITIES_H_
