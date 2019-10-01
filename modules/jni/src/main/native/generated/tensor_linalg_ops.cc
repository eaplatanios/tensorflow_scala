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
