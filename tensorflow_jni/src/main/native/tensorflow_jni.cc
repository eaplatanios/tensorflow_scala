#include "include/tensorflow_jni.h"

#include <memory>

#include "include/c_api.h"

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_version(JNIEnv* env,
                                                                                 jobject object) {
  return env->NewStringUTF(TF_Version());
}
