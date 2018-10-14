/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.types._

/** Contains functions for constructing general cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Cast {
  /** $OpDocCastCast
    *
    * @group CastOps
    *
    * @param  input Tensor to cast.
    * @tparam R Target data type.
    * @return Created op output.
    */
  private[ops] def cast[T, R: TF, OL[TT] <: OutputLike[TT]](
      input: OL[T],
      truncate: Boolean = false
  )(implicit ev: OutputOps.Aux[OL, T]): OL[R] = {
    val dataType = TF[R].dataType
    if (input.dataType == dataType) {
      input.asInstanceOf[OL[R]]
    } else {
      Op.nameScope(s"${input.name}/CastTo$dataType") {
        ev.applyUnary(input, o => {
          Op.Builder[Output[T], Output[R]](
            opType = "Cast",
            name = "Cast",
            input = o
          ).setAttribute("DstT", dataType)
              .setAttribute("Truncate", truncate)
              .setGradientFn(castGradient(_, _)(TF[R]))
              .build().output
        })
      }
    }
  }

  protected def castGradient[T, R: TF](
      op: Op[Output[T], Output[R]],
      outputGradient: Output[R]
  ): Output[T] = {
    val supportedDataTypes = Seq(FLOAT16, FLOAT32, FLOAT64, BFLOAT16, COMPLEX64, COMPLEX128)
    val sourceDataType = op.input.dataType
    val destinationDataType = outputGradient.dataType
    if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType)) {
      cast[R, T, Output](outputGradient)(TF.fromDataType(sourceDataType), implicitly[OutputOps.Aux[Output, R]])
    } else {
      null
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    *
    * @param  input Input tensor.
    * @tparam R Target data type.
    * @return Created op output.
    */
  private[ops] def bitcast[T: IsNumeric, R: TF, OL[TT] <: OutputLike[TT]](
      input: OL[T]
  )(implicit ev: OutputOps.Aux[OL, T]): OL[R] = {
    val dataType = TF[R].dataType
    Op.nameScope(s"${input.name}/BitcastTo$dataType") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[T], Output[R]](
          opType = "Bitcast",
          name = "Bitcast",
          input = o
        ).setAttribute("type", dataType)
            .build().output
      })
    }
  }
}

object Cast extends Cast {
  /** @define OpDocCastCast
    *   The `cast` op casts a tensor to a new data type.
    *
    *   The op casts `x` to the provided data type.
    *
    *   For example:
    *   {{{
    *     // `a` is a tensor with values [1.8, 2.2], and data type FLOAT32
    *     a.castTo[Int] ==> [1, 2] // with data type Int
    *   }}}
    *
    *   **NOTE**: Only a smaller number of types are supported by the `cast` op. The exact casting rule is TBD. The
    *   current implementation uses C++ static cast rules for numeric types, which may be changed in the future.
    *
    * @define OpDocCastBitcast
    *   The `bitcast` op bitcasts a tensor from one type to another without copying data.
    *
    *   Given a tensor `input`, the op returns a tensor that has the same buffer data as `input`, but with data type
    *   `dataType`. If the input data type `T` is larger (in terms of number of bytes), then the output data type
    *   `dataType`, then the shape changes from `[...]` to `[..., sizeof(T)/sizeof(dataType)]`. If `T` is smaller than
    *   `dataType`, then the op requires that the rightmost dimension be equal to `sizeof(dataType)/sizeof(T)`. The
    *   shape then changes from `[..., sizeof(type)/sizeof(T)]` to `[...]`.
    *
    *   *NOTE*: Bitcast is implemented as a low-level cast, so machines with different endian orderings will give
    *   different results.
    */
  private[ops] trait Documentation
}
