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

#include "include/exception_jni.h"

#include <stdlib.h>

#include "include/c_api.h"

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
