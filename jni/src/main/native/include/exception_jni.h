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

#include "c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TF_Status;

const char jvm_default_exception[] = "org/platanios/tensorflow/jni/TensorFlow$NativeException";
const char jvm_illegal_argument_exception[] = "java/lang/IllegalArgumentException";
const char jvm_security_exception[] = "java/lang/SecurityException";
const char jvm_illegal_state_exception[] = "java/lang/IllegalStateException";
const char jvm_null_pointer_exception[] = "java/lang/NullPointerException";
const char jvm_index_out_of_bounds_exception[] = "java/lang/IndexOutOfBoundsException";
const char jvm_unsupported_operation_exception[] = "java/lang/UnsupportedOperationException";

// Map TF_Codes to unchecked exceptions.
const char *jvm_exception_class_name(TF_Code code);

void throw_exception(JNIEnv *env, const char *clazz, const char *fmt, ...);

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
