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

import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types._

/** Contains functions for constructing general cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Cast {
  /** $OpDocCastCast
    *
    * @group CastOps
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def cast[T <: OutputLike : OutputOps](x: T, dataType: DataType, truncate: Boolean = false, name: String = "Cast"): T = {
    if (x.dataType == dataType) {
      x
    } else {
      if (x.dataType.isComplex && !dataType.isComplex)
        logger.warn("Casting complex tensors to real tensors discards the imaginary part.")
      implicitly[OutputOps[T]]
          .applyUnary(x, o => Op.Builder(opType = "Cast", name = name)
              .addInput(o)
              .setAttribute("DstT", dataType)
              .setAttribute("Truncate", truncate)
              .build().outputs(0))
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    * @param  input    Input tensor.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def bitcast(input: Output, dataType: DataType, name: String = "Bitcast"): Output = {
    Op.Builder(opType = "Bitcast", name = name)
        .addInput(input)
        .setAttribute("type", dataType)
        .build().outputs(0)
  }
}

object Cast extends Cast {
  case class CastOps(output: Output) {
    /** $OpDocMathCast
      *
      * @group MathOps
      * @param  dataType Target data type.
      * @return Result as a new tensor.
      */
    def cast(dataType: DataType): Output = Cast.cast(output, dataType)

    /** $OpDocMathBitcast
      *
      * @group MathOps
      * @param  dataType Target data type.
      * @return Result as a new tensor.
      */
    def bitcast(dataType: DataType): Output = Cast.bitcast(output, dataType)

    def toStringTensor: Output = cast(STRING)
    def toBoolean: Output = cast(BOOLEAN)
    def toFloat16: Output = cast(FLOAT16)
    def toFloat32: Output = cast(FLOAT32)
    def toFloat64: Output = cast(FLOAT64)
    def toBFloat16: Output = cast(BFLOAT16)
    def toComplex64: Output = cast(COMPLEX64)
    def toComplex128: Output = cast(COMPLEX128)
    def toInt8: Output = cast(INT8)
    def toInt16: Output = cast(INT16)
    def toInt32: Output = cast(INT32)
    def toInt64: Output = cast(INT64)
    def toUInt8: Output = cast(UINT8)
    def toUInt16: Output = cast(UINT16)
    def toUInt32: Output = cast(UINT32)
    def toUInt64: Output = cast(UINT64)
    def toQInt8: Output = cast(QINT8)
    def toQInt16: Output = cast(QINT16)
    def toQInt32: Output = cast(QINT32)
    def toQUInt8: Output = cast(QUINT8)
    def toQUInt16: Output = cast(QUINT16)
  }

  private[ops] object Gradients {
    GradientsRegistry.register("Cast", castGradient)

    private[this] def castGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val supportedDataTypes = Seq(FLOAT16, FLOAT32, FLOAT64, BFLOAT16, COMPLEX64, COMPLEX128)
      val sourceDataType = op.inputs(0).dataType
      val destinationDataType = outputGradients.head.dataType
      if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType))
        Seq(cast(outputGradients.head, sourceDataType))
      else
        Seq(null)
    }
  }

  /** @define OpDocCastCast
    *   The `cast` op casts a tensor to a new data type.
    *
    *   The op casts `x` to the provided data type.
    *
    *   For example:
    *   {{{
    *     // `a` is a tensor with values [1.8, 2.2], and data type FLOAT32
    *     cast(a, INT32) ==> [1, 2] // with data type INT32
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
