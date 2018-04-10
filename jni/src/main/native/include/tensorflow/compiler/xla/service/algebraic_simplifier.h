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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which performs algebraic simplifications.
class AlgebraicSimplifier : public HloPassInterface {
 public:
  // Given shapes 'from_shape' and 'to_shape', determines if it is valid to
  // bitcast from 'from_shape' to 'to_shape' after considering platform
  // dependent effects on layout like alignment restrictions. Precondition: the
  // two shapes have layouts, the same number of elements and
  // ShapeUtil::ReshapeIsBitcast returns true.
  using ValidBitcastCallback =
      std::function<bool(const Shape& from_shape, const Shape& to_shape)>;

  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored. If valid_bitcast_callback
  // returns true, then the pass will replace reshapes and transposes with
  // bitcasts.
  AlgebraicSimplifier(bool is_layout_sensitive,
                      ValidBitcastCallback valid_bitcast_callback,
                      bool enable_dot_strength_reduction = true,
                      bool enable_conv_simplification = true)
      : is_layout_sensitive_(is_layout_sensitive),
        valid_bitcast_callback_(std::move(valid_bitcast_callback)),
        enable_dot_strength_reduction_(enable_dot_strength_reduction),
        enable_conv_simplification_(enable_conv_simplification) {}
  ~AlgebraicSimplifier() override = default;
  tensorflow::StringPiece name() const override { return "algsimp"; }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool is_layout_sensitive_;
  ValidBitcastCallback valid_bitcast_callback_;

  // Enable dot simplification on platforms where it is profitable.
  bool enable_dot_strength_reduction_;

  // Enable convolution simplification on platforms where it is profitable.
  bool enable_conv_simplification_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
