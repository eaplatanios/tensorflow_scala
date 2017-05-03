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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_EXCEPTION_JNI_H_
