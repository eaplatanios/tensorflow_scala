#include "include/tensorflow_jni.h"

#include <memory>

#include "include/c_api.h"

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_version(JNIEnv* env,
                                                                                 jobject object) {
  return env->NewStringUTF(TF_Version());
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_TensorFlow_00024_dataTypeSize(
  JNIEnv* env, jobject object, jint data_type_c_value) {
  return (jint) TF_DataTypeSize((TF_DataType) data_type_c_value);
}
