/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_platanios_tensorflow_jni_generated_tensors_Linalg__ */

#ifndef _Included_org_platanios_tensorflow_jni_generated_tensors_Linalg__
#define _Included_org_platanios_tensorflow_jni_generated_tensors_Linalg__
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    cholesky
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_cholesky
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    choleskyGrad
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_choleskyGrad
  (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    logMatrixDeterminant
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_logMatrixDeterminant
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    matrixDeterminant
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixDeterminant
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    matrixInverse
 * Signature: (JJZ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixInverse
  (JNIEnv *, jobject, jlong, jlong, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    matrixSolve
 * Signature: (JJJZ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixSolve
  (JNIEnv *, jobject, jlong, jlong, jlong, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    matrixSolveLs
 * Signature: (JJJJZ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixSolveLs
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    matrixTriangularSolve
 * Signature: (JJJZZ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixTriangularSolve
  (JNIEnv *, jobject, jlong, jlong, jlong, jboolean, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    qr
 * Signature: (JJZ)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_qr
  (JNIEnv *, jobject, jlong, jlong, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    selfAdjointEigV2
 * Signature: (JJZ)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_selfAdjointEigV2
  (JNIEnv *, jobject, jlong, jlong, jboolean);

/*
 * Class:     org_platanios_tensorflow_jni_generated_tensors_Linalg__
 * Method:    svd
 * Signature: (JJZZ)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_svd
  (JNIEnv *, jobject, jlong, jlong, jboolean, jboolean);

#ifdef __cplusplus
}
#endif
#endif
