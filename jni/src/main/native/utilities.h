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

namespace {
  template <class T>
  inline T* require_handle(JNIEnv* env, jlong handle) {
    static_assert(sizeof(jlong) >= sizeof(T*), "Cannot package C object pointers as a Java long");
    if (handle == 0) {
      throw_exception(env, jvm_null_pointer_exception, "This object has been disposed/deallocated already.");
      return nullptr;
    }
    return reinterpret_cast<T*>(handle);
  }
}  // namespace

#define REQUIRE_HANDLE(name, type, variable_name, null_return_value)     \
  type* name = require_handle<type>(env, variable_name);                 \
  if (name == nullptr) return null_return_value;

#endif  // TENSORFLOW_JNI_UTILITIES_H_
