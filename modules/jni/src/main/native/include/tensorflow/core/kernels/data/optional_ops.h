/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {
namespace data {

// Stores a DT_VARIANT value representing an Optional with the given value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalWithValueToOutput(OpKernelContext* ctx, int output_index,
                                      std::vector<Tensor> value);

// Stores a DT_VARIANT value representing an Optional with no value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
