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

import org.platanios.tensorflow.api.types._

/** Contains functions for constructing general cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Cast {
  /** $OpDocCastCast
    *
    * @group CastOps
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def cast[T, R, OL[TT] <: OutputLike[TT]](
      x: OL[T],
      dataType: DataType[R],
      truncate: Boolean = false,
      name: String = "Cast"
  )(implicit ev: OutputOps.Aux[OL, T]): OL[R] = {
    if (x.dataType == dataType) {
      x.asInstanceOf[OL[R]]
    } else {
      ev.applyUnary(x, o => {
        Op.Builder[Output[T], Output[R]](
          opType = "Cast",
          name = name,
          input = o
        ).setAttribute("DstT", dataType)
            .setAttribute("Truncate", truncate)
            .setGradientFn(castGradient)
            .build().output
      })
    }
  }

  protected def castGradient[T, R](
      op: Op[Output[T], Output[R]],
      outputGradient: Output[R]
  ): Output[T] = {
    val supportedDataTypes = Seq(FLOAT16, FLOAT32, FLOAT64, BFLOAT16, COMPLEX64, COMPLEX128)
    val sourceDataType = op.input.dataType
    val destinationDataType = outputGradient.dataType
    if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType))
      cast(outputGradient, sourceDataType)
    else
      null
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    *
    * @param  input    Input tensor.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def bitcast[T: IsNumeric, R](
      input: Output[T],
      dataType: DataType[R],
      name: String = "Bitcast"
  ): Output[R] = {
    Op.Builder[Output[T], Output[R]](
      opType = "Bitcast",
      name = name,
      input = input
    ).setAttribute("type", dataType)
        .build().output
  }
}

object Cast extends Cast {
  private[ops] trait Implicits {
    implicit def outputConvertibleToCastOps[T, OC](
        value: OC
    )(implicit f: OC => Output[T]): CastOps[T] = {
      new CastOps(f(value))
    }

    implicit class CastOps[T](val output: Output[T]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def cast[R](dataType: DataType[R]): Output[R] = {
        Cast.cast(output, dataType)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        *
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def bitcast[R](
          dataType: DataType[R]
      )(implicit ev: IsNumeric[T]): Output[R] = {
        Cast.bitcast(output, dataType)
      }

      def toStringTensor: Output[String] = cast(STRING)
      def toBoolean: Output[Boolean] = cast(BOOLEAN)
      def toFloat16: Output[Half] = cast(FLOAT16)
      def toFloat32: Output[Float] = cast(FLOAT32)
      def toFloat64: Output[Double] = cast(FLOAT64)
      def toBFloat16: Output[TruncatedHalf] = cast(BFLOAT16)
      def toComplex64: Output[ComplexFloat] = cast(COMPLEX64)
      def toComplex128: Output[ComplexDouble] = cast(COMPLEX128)
      def toInt8: Output[Byte] = cast(INT8)
      def toInt16: Output[Short] = cast(INT16)
      def toInt32: Output[Int] = cast(INT32)
      def toInt64: Output[Long] = cast(INT64)
      def toUInt8: Output[UByte] = cast(UINT8)
      def toUInt16: Output[UShort] = cast(UINT16)
      def toUInt32: Output[UInt] = cast(UINT32)
      def toUInt64: Output[ULong] = cast(UINT64)
      def toQInt8: Output[QByte] = cast(QINT8)
      def toQInt16: Output[QShort] = cast(QINT16)
      def toQInt32: Output[QInt] = cast(QINT32)
      def toQUInt8: Output[QUByte] = cast(QUINT8)
      def toQUInt16: Output[QUShort] = cast(QUINT16)
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
