/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_H_

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Class describing a contiguous sequence of elements (ie, C array) which form
// the components of Shaped values in XLA. XLA arrays are trivially a
// single LogicalBuffer. Tuple values are made up of more than one
// LogicalBuffer: a LogicalBuffer for the pointers to elements, and a
// LogicalBuffer for each child element.
//
// Every buffer is defined by a particular instruction and most instructions
// define only a single buffer. Instructions which define a single buffer
// include array-shaped instructions such as Add but also includes Tuple-shaped
// instructions such as Tuple. The Tuple instruction defines a single buffer
// which is a vector of pointers to the buffers containing the Tuple
// instruction's operands. Though the result of the Tuple instruction includes
// multiple buffers only the top-level buffer (the vector of pointers) is
// defined by the Tuple instruction. The buffers containing the tuple elements
// are defined by earlier instructions, usually the operands of the Tuple
// instruction.
//
// Instructions which construct both the tuple *and* the tuple elements define
// more than one buffer. This includes (at least) tuple-shaped Constant,
// Parameter, Infeed and While instructions. The tuple-shaped instructions do
// not assemble a tuple from existing buffers like the Tuple instruction does,
// but rather define the entire tuple.
//
// Some instructions, such as Bitcast, define no buffers. These instructions
// simply forward buffers from their operands.
//
// The LogicalBuffer object describes which HLO instruction defines a buffer and
// where within that instruction's output shape the buffer is defined. The
// location within the output shape is indicated by LogicalBuffer::index() which
// is defined identically to the index used in
// ShapeUtil::GetSubshape(). Examples:
//
// %add = Add(%foo, %bar)
// %tuple_constant = Constant({1, {42, 43}})
//
// %add defines a single array-shaped buffer LogicalBuffer(%add, {}) which holds
// the array result of the add operation. The nested-tuple-shaped
// %tuple_constant defines 5 buffers described by the following LogicalBuffer
// objects:
//
//   LogicalBuffer(%tuple_constant, {})      // "Top-level" buffer: vector of
//                                           //  pointers to LogicalBuffers at
//                                           //  indices {0} and {1}
//   LogicalBuffer(%tuple_constant, {0})     // Holds value "1"
//   LogicalBuffer(%tuple_constant, {1})     // Holds nested tuple: vector of
//                                           //  pointers to LogicalBuffers at
//                                           //  indices {1, 0} and {1, 1}
//   LogicalBuffer(%tuple_constant, {1, 0})  // Holds value "42"
//   LogicalBuffer(%tuple_constant, {1, 1})  // Holds value "43"
class LogicalBuffer {
 public:
  TF_LIB_GTL_DEFINE_INT_TYPE(Color, int64);

  // Id is a unique identifier for the LogicalBuffer to facilitate efficient
  // collections of LogicalBuffers with stable iteration order.
  // LogicalBuffers are typically created and accessed through
  // TuplePointsToAnalysis, and points-to analysis assigns each LogicalBuffer a
  // unique value.
  using Id = int64;

  // Functions which return the size and alignment of a logical buffer in bytes.
  using SizeFunction = std::function<int64(const LogicalBuffer&)>;
  using AlignmentFunction = std::function<int64(LogicalBuffer::Color)>;

  LogicalBuffer(HloInstruction* instruction, const ShapeIndex& index, Id id);

  Id id() const { return id_; }

  // Return the instruction that defines the buffer.
  HloInstruction* instruction() const { return instruction_; }

  // Return the index within the output of the instruction where the buffer is
  // defined. Index used defined as in ShapeUtil::GetSubshape()
  const ShapeIndex& index() const { return index_; }

  // Return the color of the logical buffer. Differently colored buffers can
  // not be parts of the same allocation.
  Color color() const {
    CHECK_NE(color_, kInvalidColor)
        << "Should not query the color of a buffer that was never colored";
    return color_;
  }

  void set_color(Color color) {
    CHECK_NE(color, kInvalidColor)
        << "Should not set the color of a buffer to the invalid color";
    color_ = color;
  }

  bool has_color() const { return color_ != kInvalidColor; }

  // Return the shape of the buffer. This reference points into the shape field
  // of the instruction defining the buffer.  Therefore, the returned shape will
  // contain the layout of instruction, if any.
  const Shape& shape() const {
    return ShapeUtil::GetSubshape(instruction_->shape(), index_);
  }

  // Returns true if this buffer is the top-level output buffer of the defining
  // HLO instruction. This is equivalent to index == {}.
  bool IsTopLevel() const { return index_.empty(); }

  // Whether this buffer contains a tuple.
  bool IsTuple() const { return is_tuple_; }

  // Whether this buffer contains an array.
  bool IsArray() const { return is_array_; }

  // operator< is required for std::set.
  bool operator<(const LogicalBuffer& other) const { return id_ < other.id_; }

  string ToString() const;
  LogicalBufferProto ToProto(const SizeFunction& size_fn) const;

  // Returns the LogicalBufferProto::Location that serializes the given
  // instruction and index.
  static LogicalBufferProto::Location ToLocationProto(
      const HloInstruction& instruction, const ShapeIndex& index);

  const Color kInvalidColor = Color(-1);

 private:
  HloInstruction* instruction_;
  Id id_ : 62;
  bool is_array_ : 1;
  bool is_tuple_ : 1;
  Color color_;
  ShapeIndex index_;

  // Similar to HLO constructs (HloInstruction, etc), pointers are used for
  // comparison to equality, so disable all copying.
  TF_DISALLOW_COPY_AND_ASSIGN(LogicalBuffer);
};

std::ostream& operator<<(std::ostream& out, const LogicalBuffer& buffer);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_H_
