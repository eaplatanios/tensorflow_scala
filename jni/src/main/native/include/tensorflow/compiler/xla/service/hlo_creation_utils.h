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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Some lightweight utilities intended to make HLO instruction creation more
// ergonomic.  We don't have a complete set of helpers yet -- I expect we'll
// expand this interface as needed on an ad-hoc basis.

// Creates a binary HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> MakeBinaryHlo(HloOpcode opcode, HloInstruction* lhs,
                                        HloInstruction* rhs);

// Creates a pad HLO instruction and adds it to the computation containing
// `operand` and `padding_value` (`operand` and `padding_value` must be in the
// same computation).
StatusOr<HloInstruction*> MakePadHlo(HloInstruction* operand,
                                     HloInstruction* padding_value,
                                     const PaddingConfig& padding_config);

// Creates a slice HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeSliceHlo(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides);

// Creates a convolution HLO instruction and adds it to the computation
// containing `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> MakeConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers);

// Creates a transpose HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeTransposeHlo(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> dimensions);

// Creates a reshape HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeReshapeHlo(const Shape& result_shape,
                                         HloInstruction* operand);

StatusOr<HloInstruction*> MakeReshapeHlo(
    tensorflow::gtl::ArraySlice<int64> result_shape_dim_bounds,
    HloInstruction* operand);

// Creates a dynamic-slice HLO instruction and adds it to the computation
// containing `operand` and `start_indices` (`operand` and `start_indices` must
// be in the same computation).
StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, HloInstruction* start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes);

// Creates a dynamic-update-slice HLO instruction and adds it to the computation
// containing `operand`, `update` and `start_indices` (`operand`, `update` and
// `start_indices` must be in the same computation).
StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices);

// Creates a broadcast HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeBroadcastHlo(
    HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions,
    tensorflow::gtl::ArraySlice<int64> result_shape_bounds);

// Creates a GetTupleElement HLO instruction and adds it to the computation
// containing `operand`.
StatusOr<HloInstruction*> MakeGetTupleElementHlo(HloInstruction* operand,
                                                 int64 index);

// Creates a Concatenate HLO instruction and adds it to the computation
// containing `operands` (`operands` must be non-empty and every element must be
// contained in the same computation).
StatusOr<HloInstruction*> MakeConcatHlo(
    tensorflow::gtl::ArraySlice<HloInstruction*> operands, int64 dimension);

// -----------------------------------------------------------------------------
// Some other miscellaneous helpers to generate common HLO patterns.  All of
// these add all the instructions they generate into the computation containing
// their operand(s).

// Collapses (via reshape) the first N (logical) dimensions of `operand` into a
// single leading dimension.  `operand` must have rank > n.
//
// For instance if `operand` has shape f32[7,8,9] and n is 2 then the output is
// the `operand` reshaped to [56,9].
StatusOr<HloInstruction*> CollapseFirstNDims(HloInstruction* operand, int64 n);

// Expands (via reshape) the first (logical) dimension of `operand` into a
// sequence of `expanded_dims` dimensions.  `operand` must at least be of rank 1
// and the number of elements in its first dimension must be equal to the
// product of `expanded_dims`.
//
// For instance if `operand` has shape f32[200,9,7] and expanded_dims is
// {2,5,20} the result is `operand` reshaped to [2,5,20,9,7].
StatusOr<HloInstruction*> ExpandFirstDimIntoNDims(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> expanded_dims);

// Elides (via reshape) a set of degenerate dimensions (dimensions containing
// exactly one element), `dims_to_elide` from `operand`.  Every dimension in
// `dims_to_elide` must be a degenerate dimension.  `dims_to_elide` must be
// sorted and not contain duplicates.
//
// For example if `operand` is of shape f32[19,1,20,1,7,1,9] and dims_to_elide
// is {1,5} then the result is `operand` reshaped to [19,20,1,7,9].
StatusOr<HloInstruction*> ElideDegenerateDims(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> dims_to_elide);

// Pads `operand` (which must have rank 1) with `zeros_to_prepend` zeros in the
// front and `zeros_to_append` zeros in the back.
StatusOr<HloInstruction*> PadVectorWithZeros(HloInstruction* operand,
                                             int64 zeros_to_prepend,
                                             int64 zeros_to_append);

// Broadcasts a zero value of type `element_type` into a tensor with element
// type `element_type` and dimension bounds `broadcast_dimensions`.  The
// broadcast instruction is emitted into `computation`.
StatusOr<HloInstruction*> BroadcastZeros(
    HloComputation* computation, PrimitiveType element_type,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

// Creates a HLO computation that takes arguments of type `domain` and produces
// a value of type `range`.
StatusOr<std::unique_ptr<HloComputation>> CreateComputationWithSignature(
    tensorflow::gtl::ArraySlice<const Shape*> domain, const Shape& range,
    tensorflow::StringPiece name);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
