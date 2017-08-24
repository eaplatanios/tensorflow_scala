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

#ifdef __cplusplus
extern "C" {
#endif

struct TF_Status;

extern const char jvm_default_exception[];
extern const char jvm_illegal_argument_exception[];
extern const char jvm_security_exception[];
extern const char jvm_illegal_state_exception[];
extern const char jvm_null_pointer_exception[];
extern const char jvm_index_out_of_bounds_exception[];
extern const char jvm_unsupported_operation_exception[];

void throw_exception(JNIEnv *env, const char *clazz, const char *fmt, ...);

// If status is not TF_OK, then throw an appropriate exception.
// Returns true iff TF_GetCode(status) == TF_OK.
bool throw_exception_if_not_ok(JNIEnv *env, const TF_Status *status);

#define CHECK_STATUS(env, status, null_return_value)     \
  if (!throw_exception_if_not_ok(env, status))           \
    return null_return_value;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_EXCEPTION_JNI_H_
