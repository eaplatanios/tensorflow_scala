/* DO NOT EDIT THIS FILE - it is machine generated */

/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

#include "tensor_linalg_ops.h"
#include "exception.h"
#include "utilities.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_logMatrixDeterminant(
    JNIEnv* env, jobject object, jlong context_handle, jlong input) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "LogMatrixDeterminant", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), nullptr);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), nullptr);

  REQUIRE_HANDLE(input_handle, TFE_TensorHandle, input, nullptr);
  TFE_OpAddInput(op.get(), input_handle, status.get());
  CHECK_STATUS(env, status.get(), nullptr);

  REQUIRE_HANDLE(attr_T_input_handle, TFE_TensorHandle, input, nullptr);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_input_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  const int num_outputs = 2;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), nullptr);

  jlongArray outputs_array = env->NewLongArray(static_cast<jsize>(num_outputs));
  jlong* output_elems = env->GetLongArrayElements(outputs_array, nullptr);
  for (int i = 0; i < num_outputs; ++i) {
    output_elems[i] = reinterpret_cast<jlong>(outputs[i]);
  }
  env->ReleaseLongArrayElements(outputs_array, output_elems, 0);
  return outputs_array;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixDeterminant(
    JNIEnv* env, jobject object, jlong context_handle, jlong input) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "MatrixDeterminant", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(input_handle, TFE_TensorHandle, input, 0);
  TFE_OpAddInput(op.get(), input_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(attr_T_input_handle, TFE_TensorHandle, input, 0);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_input_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  const int num_outputs = 1;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);

  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixInverse(
    JNIEnv* env, jobject object, jlong context_handle, jlong input, jboolean adjoint) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "MatrixInverse", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(input_handle, TFE_TensorHandle, input, 0);
  TFE_OpAddInput(op.get(), input_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(attr_T_input_handle, TFE_TensorHandle, input, 0);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_input_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  TFE_OpSetAttrBool(op.get(), "adjoint", static_cast<unsigned char>(adjoint));

  const int num_outputs = 1;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);

  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixSolve(
    JNIEnv* env, jobject object, jlong context_handle, jlong matrix, jlong rhs, jboolean adjoint) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "MatrixSolve", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(matrix_handle, TFE_TensorHandle, matrix, 0);
  TFE_OpAddInput(op.get(), matrix_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(rhs_handle, TFE_TensorHandle, rhs, 0);
  TFE_OpAddInput(op.get(), rhs_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(attr_T_matrix_handle, TFE_TensorHandle, matrix, 0);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_matrix_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  REQUIRE_HANDLE(attr_T_rhs_handle, TFE_TensorHandle, rhs, 0);
  const TF_DataType attr_T_rhs = TFE_TensorHandleDataType(attr_T_rhs_handle);
  if (attr_T != attr_T_rhs) {
      std::stringstream error_msg;
      error_msg
          << "Argument 'rhs' of 'matrixSolve' op with data type '"
          << attr_T_rhs
          << "' must match data type '"
          << attr_T
          << "' of argument 'matrix'";
      throw_exception(env, tf_invalid_argument_exception, error_msg.str().c_str());
  }

  TFE_OpSetAttrBool(op.get(), "adjoint", static_cast<unsigned char>(adjoint));

  const int num_outputs = 1;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);

  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixSolveLs(
    JNIEnv* env, jobject object, jlong context_handle, jlong matrix, jlong rhs, jlong l2_regularizer, jboolean fast) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "MatrixSolveLs", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(matrix_handle, TFE_TensorHandle, matrix, 0);
  TFE_OpAddInput(op.get(), matrix_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(rhs_handle, TFE_TensorHandle, rhs, 0);
  TFE_OpAddInput(op.get(), rhs_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(l2_regularizer_handle, TFE_TensorHandle, l2_regularizer, 0);
  TFE_OpAddInput(op.get(), l2_regularizer_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(attr_T_matrix_handle, TFE_TensorHandle, matrix, 0);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_matrix_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  REQUIRE_HANDLE(attr_T_rhs_handle, TFE_TensorHandle, rhs, 0);
  const TF_DataType attr_T_rhs = TFE_TensorHandleDataType(attr_T_rhs_handle);
  if (attr_T != attr_T_rhs) {
      std::stringstream error_msg;
      error_msg
          << "Argument 'rhs' of 'matrixSolveLs' op with data type '"
          << attr_T_rhs
          << "' must match data type '"
          << attr_T
          << "' of argument 'matrix'";
      throw_exception(env, tf_invalid_argument_exception, error_msg.str().c_str());
  }

  TFE_OpSetAttrBool(op.get(), "fast", static_cast<unsigned char>(fast));

  const int num_outputs = 1;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);

  return reinterpret_cast<jlong>(outputs[0]);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_generated_tensors_Linalg_00024_matrixTriangularSolve(
    JNIEnv* env, jobject object, jlong context_handle, jlong matrix, jlong rhs, jboolean lower, jboolean adjoint) {
  REQUIRE_HANDLE(context, TFE_Context, context_handle, 0);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "MatrixTriangularSolve", status.get()), TFE_DeleteOp);
  CHECK_STATUS(env, status.get(), 0);
  TFE_OpSetDevice(op.get(), "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(matrix_handle, TFE_TensorHandle, matrix, 0);
  TFE_OpAddInput(op.get(), matrix_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(rhs_handle, TFE_TensorHandle, rhs, 0);
  TFE_OpAddInput(op.get(), rhs_handle, status.get());
  CHECK_STATUS(env, status.get(), 0);

  REQUIRE_HANDLE(attr_T_matrix_handle, TFE_TensorHandle, matrix, 0);
  const TF_DataType attr_T = TFE_TensorHandleDataType(attr_T_matrix_handle);
  TFE_OpSetAttrType(op.get(), "T", attr_T);

  REQUIRE_HANDLE(attr_T_rhs_handle, TFE_TensorHandle, rhs, 0);
  const TF_DataType attr_T_rhs = TFE_TensorHandleDataType(attr_T_rhs_handle);
  if (attr_T != attr_T_rhs) {
      std::stringstream error_msg;
      error_msg
          << "Argument 'rhs' of 'matrixTriangularSolve' op with data type '"
          << attr_T_rhs
          << "' must match data type '"
          << attr_T
          << "' of argument 'matrix'";
      throw_exception(env, tf_invalid_argument_exception, error_msg.str().c_str());
  }

  TFE_OpSetAttrBool(op.get(), "lower", static_cast<unsigned char>(lower));

  TFE_OpSetAttrBool(op.get(), "adjoint", static_cast<unsigned char>(adjoint));

  const int num_outputs = 1;
  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
  std::unique_ptr<int[]> actual_num_outputs(new int[1] {num_outputs});
  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
  CHECK_STATUS(env, status.get(), 0);

  return reinterpret_cast<jlong>(outputs[0]);
}
