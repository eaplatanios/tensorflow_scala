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

#ifndef TENSORFLOW_JAVA_EXCEPTION_JNI_H_
#define TENSORFLOW_JAVA_EXCEPTION_JNI_H_

#include <jni.h>
#include <stdlib.h>
#include <string>

#include "tensorflow/c/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TF_Status;

const char tf_cancelled_exception[] = "org/platanios/tensorflow/jni/CancelledException";
const char tf_unknown_exception[] = "org/platanios/tensorflow/jni/UnknownException";
const char tf_invalid_argument_exception[] = "org/platanios/tensorflow/jni/InvalidArgumentException";
const char tf_deadline_exceeded_exception[] = "org/platanios/tensorflow/jni/DeadlineExceededException";
const char tf_not_found_exception[] = "org/platanios/tensorflow/jni/NotFoundException";
const char tf_already_exists_exception[] = "org/platanios/tensorflow/jni/AlreadyExistsException";
const char tf_permission_denied_exception[] = "org/platanios/tensorflow/jni/PermissionDeniedException";
const char tf_unauthenticated_exception[] = "org/platanios/tensorflow/jni/UnauthenticatedException";
const char tf_resource_exhausted_exception[] = "org/platanios/tensorflow/jni/ResourceExhaustedException";
const char tf_failed_precondition_exception[] = "org/platanios/tensorflow/jni/FailedPreconditionException";
const char tf_aborted_exception[] = "org/platanios/tensorflow/jni/AbortedException";
const char tf_out_of_range_exception[] = "org/platanios/tensorflow/jni/OutOfRangeException";
const char tf_unimplemented_exception[] = "org/platanios/tensorflow/jni/UnimplementedException";
const char tf_internal_exception[] = "org/platanios/tensorflow/jni/InternalException";
const char tf_unavailable_exception[] = "org/platanios/tensorflow/jni/UnavailableException";
const char tf_data_loss_exception[] = "org/platanios/tensorflow/jni/DataLossException";

const char jvm_illegal_argument_exception[] = "java/lang/IllegalArgumentException";
const char jvm_security_exception[] = "java/lang/SecurityException";
const char jvm_illegal_state_exception[] = "java/lang/IllegalStateException";
const char jvm_null_pointer_exception[] = "java/lang/NullPointerException";
const char jvm_index_out_of_bounds_exception[] = "java/lang/IndexOutOfBoundsException";
const char jvm_unsupported_operation_exception[] = "java/lang/UnsupportedOperationException";

// Map TF_Codes to unchecked exceptions.
inline const char *jvm_exception_class_name(TF_Code code) {
  switch (code) {
    case TF_OK:
      return nullptr;
    case TF_CANCELLED:
      return tf_cancelled_exception;
    case TF_UNKNOWN:
      return tf_unknown_exception;
    case TF_INVALID_ARGUMENT:
      return tf_invalid_argument_exception;
    case TF_DEADLINE_EXCEEDED:
      return tf_deadline_exceeded_exception;
    case TF_NOT_FOUND:
      return tf_not_found_exception;
    case TF_ALREADY_EXISTS:
      return tf_already_exists_exception;
    case TF_PERMISSION_DENIED:
      return tf_permission_denied_exception;
    case TF_UNAUTHENTICATED:
      return tf_unauthenticated_exception;
    case TF_RESOURCE_EXHAUSTED:
      return tf_resource_exhausted_exception;
    case TF_FAILED_PRECONDITION:
      return tf_failed_precondition_exception;
    case TF_ABORTED:
      return tf_aborted_exception;
    case TF_OUT_OF_RANGE:
      return tf_out_of_range_exception;
    case TF_UNIMPLEMENTED:
      return tf_unimplemented_exception;
    case TF_INTERNAL:
      return tf_internal_exception;
    case TF_UNAVAILABLE:
      return tf_unavailable_exception;
    case TF_DATA_LOSS:
      return tf_data_loss_exception;
    default:
      return tf_unknown_exception;
  }
}

// Map unchecked exceptions to TF_Codes.
inline TF_Code tf_error_code(std::string jvm_name) {
  if (jvm_name == "org.platanios.tensorflow.jni.CancelledException") {
    return TF_CANCELLED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.UnknownException") {
    return TF_UNKNOWN;
  } else if (jvm_name == "org.platanios.tensorflow.jni.InvalidArgumentException" ||
      jvm_name == "java.lang.IllegalArgumentException") {
    return TF_INVALID_ARGUMENT;
  } else if (jvm_name == "org.platanios.tensorflow.jni.DeadlineExceededException") {
    return TF_DEADLINE_EXCEEDED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.NotFoundException") {
    return TF_NOT_FOUND;
  } else if (jvm_name == "org.platanios.tensorflow.jni.AlreadyExistsException") {
    return TF_ALREADY_EXISTS;
  } else if (jvm_name == "org.platanios.tensorflow.jni.PermissionDeniedException" ||
      jvm_name == "java.lang.SecurityException") {
    return TF_PERMISSION_DENIED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.UnauthenticatedException") {
    return TF_UNAUTHENTICATED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.ResourceExhaustedException") {
    return TF_RESOURCE_EXHAUSTED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.FailedPreconditionException") {
    return TF_FAILED_PRECONDITION;
  } else if (jvm_name == "org.platanios.tensorflow.jni.AbortedException") {
    return TF_ABORTED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.OutOfRangeException" ||
      jvm_name == "java.lang.IndexOutOfBoundsException") {
    return TF_OUT_OF_RANGE;
  } else if (jvm_name == "org.platanios.tensorflow.jni.UnimplementedException") {
    return TF_UNIMPLEMENTED;
  } else if (jvm_name == "org.platanios.tensorflow.jni.InternalException") {
    return TF_INTERNAL;
  } else if (jvm_name == "org.platanios.tensorflow.jni.UnavailableException") {
    return TF_UNAVAILABLE;
  } else if (jvm_name == "org.platanios.tensorflow.jni.DataLossException") {
    return TF_DATA_LOSS;
  } else {
    return TF_UNKNOWN;
  }
}

inline void throw_exception(JNIEnv *env, const char *clazz, const char *fmt, ...) {
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

// If status is not TF_OK, then throw an appropriate exception.
// Returns true iff TF_GetCode(status) == TF_OK.
inline bool throw_exception_if_not_ok(JNIEnv *env, const TF_Status *status) {
  const char *clazz = jvm_exception_class_name(TF_GetCode(status));
  if (clazz == nullptr) return true;
  env->ThrowNew(env->FindClass(clazz), TF_Message(status));
  return false;
}

#define CHECK_STATUS(env, status, null_return_value)     \
  if (!throw_exception_if_not_ok(env, status))           \
    return null_return_value;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_EXCEPTION_JNI_H_
