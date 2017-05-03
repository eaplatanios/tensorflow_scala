#include "include/exception_jni.h"

#include <stdlib.h>

#include "include/c_api.h"

const char jvm_default_exception[] = "org/platanios/tensorflow/jni/TensorFlow$NativeException";
const char jvm_illegal_argument_exception[] = "java/lang/IllegalArgumentException";
const char jvm_security_exception[] = "java/lang/SecurityException";
const char jvm_illegal_state_exception[] = "java/lang/IllegalStateException";
const char jvm_null_pointer_exception[] = "java/lang/NullPointerException";
const char jvm_index_out_of_bounds_exception[] = "java/lang/IndexOutOfBoundsException";
const char jvm_unsupported_operation_exception[] = "java/lang/UnsupportedOperationException";

void throw_exception(JNIEnv *env, const char *clazz, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  // Using vsnprintf() instead of vasprintf() because the latter doesn't seem to be easily available on Windows
  const size_t max_msg_len = 512;
  char *message = static_cast<char *>(malloc(max_msg_len));
  if (vsnprintf(message, max_msg_len, fmt, args) >= 0)
    env->ThrowNew(env->FindClass(clazz), message);
  else
    env->ThrowNew(env->FindClass(clazz), "");
  free(message);
  va_end(args);
}

namespace {
// Map TF_Codes to unchecked exceptions.
  const char *jvm_exception_class_name(TF_Code code) {
    switch (code) {
      case TF_OK:
        return nullptr;
      case TF_INVALID_ARGUMENT:
        return jvm_illegal_argument_exception;
      case TF_UNAUTHENTICATED:
      case TF_PERMISSION_DENIED:
        return jvm_security_exception;
      case TF_RESOURCE_EXHAUSTED:
      case TF_FAILED_PRECONDITION:
        return jvm_illegal_state_exception;
      case TF_OUT_OF_RANGE:
        return jvm_index_out_of_bounds_exception;
      case TF_UNIMPLEMENTED:
        return jvm_unsupported_operation_exception;
      default:
        return jvm_default_exception;
    }
  }
}  // namespace

bool throw_exception_if_not_ok(JNIEnv *env, const TF_Status *status) {
  const char *clazz = jvm_exception_class_name(TF_GetCode(status));
  if (clazz == nullptr) return true;
  env->ThrowNew(env->FindClass(clazz), TF_Message(status));
  return false;
}
