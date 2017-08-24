#ifndef TENSORFLOW_JNI_UTILITIES_H_
#define TENSORFLOW_JNI_UTILITIES_H_

#include "include/c_api.h"
#include "include/c_eager_api.h"

namespace {
  template <class T>
  T* require_handle(JNIEnv* env, jlong handle) {
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
