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
  private[api] def cast[T: TF, R: TF, OL[TT] <: OutputLike[TT]](
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
              .setGradientFn(castGradient(_, _)(TF[T], TF[R]))
              .build().output
        })
      }
    }
  }

  protected def castGradient[T: TF, R: TF](
      op: Op[Output[T], Output[R]],
      outputGradient: Output[R]
  ): Output[T] = {
    val supportedDataTypes = Seq(FLOAT16, FLOAT32, FLOAT64, BFLOAT16, COMPLEX64, COMPLEX128)
    val sourceDataType = op.input.dataType
    val destinationDataType = outputGradient.dataType
    if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType)) {
      cast[R, T, Output](outputGradient)
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
  private[api] def bitcast[T: IsNumeric, R: TF, OL[TT] <: OutputLike[TT]](
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
  private[ops] trait Implicits {
    implicit def outputConvertibleToCastOps[T: TF, OC, OL[TT] <: OutputLike[TT]](
        value: OC
    )(implicit
        f: OC => OL[T],
        ev: OutputOps.Aux[OL, T]
    ): CastOps[T, OL] = {
      new CastOps(f(value))
    }

    implicit class CastOps[T: TF, OL[TT] <: OutputLike[TT]](
        val output: OL[T]
    )(implicit evOps: OutputOps.Aux[OL, T]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF]: OL[R] = {
        Cast.cast[T, R, OL](output)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R]): OL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, OL](output)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF](truncate: Boolean): OL[R] = {
        Cast.cast[T, R, OL](output, truncate)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R], truncate: Boolean): OL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, OL](output, truncate)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        *
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R: TF](implicit ev: IsNumeric[T]): OL[R] = {
        Cast.bitcast[T, R, OL](output)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        *
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R](dataType: DataType[R])(implicit ev: IsNumeric[T]): OL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.bitcast[T, R, OL](output)
      }

      def toStringTensor: OL[String] = output.castTo[String]
      def toBoolean: OL[Boolean] = output.castTo[Boolean]
      def toHalf: OL[Half] = output.castTo[Half]
      def toFloat: OL[Float] = output.castTo[Float]
      def toDouble: OL[Double] = output.castTo[Double]
      def toTruncatedHalf: OL[TruncatedHalf] = output.castTo[TruncatedHalf]
      def toComplexFloat: OL[ComplexFloat] = output.castTo[ComplexFloat]
      def toComplexDouble: OL[ComplexDouble] = output.castTo[ComplexDouble]
      def toByte: OL[Byte] = output.castTo[Byte]
      def toShort: OL[Short] = output.castTo[Short]
      def toInt: OL[Int] = output.castTo[Int]
      def toLong: OL[Long] = output.castTo[Long]
      def toUByte: OL[UByte] = output.castTo[UByte]
      def toUShort: OL[UShort] = output.castTo[UShort]
      def toUInt: OL[UInt] = output.castTo[UInt]
      def toULong: OL[ULong] = output.castTo[ULong]
      def toQByte: OL[QByte] = output.castTo[QByte]
      def toQShort: OL[QShort] = output.castTo[QShort]
      def toQInt: OL[QInt] = output.castTo[QInt]
      def toQUByte: OL[QUByte] = output.castTo[QUByte]
      def toQUShort: OL[QUShort] = output.castTo[QUShort]
      def toResource: OL[Resource] = output.castTo[Resource]
      def toVariant: OL[Variant] = output.castTo[Variant]
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
