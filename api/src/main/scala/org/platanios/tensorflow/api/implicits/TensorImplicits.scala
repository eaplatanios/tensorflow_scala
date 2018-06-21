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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.tensors.{Tensor, TensorConvertible, TensorLike, TensorOps}
import org.platanios.tensorflow.api.tensors.ops.Basic._
import org.platanios.tensorflow.api.tensors.ops.{Math, NN}
import org.platanios.tensorflow.api.types._

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[implicits] trait TensorImplicits
    extends Math.Implicits
        with NN.Implicits {
  // implicit def tensorFromTensorLike[D <: DataType, T <: TensorLike[D]](value: T): Tensor[D] = value.toTensor

  implicit def tensorFromTensorConvertible[T, D <: DataType](value: T)(implicit ev: TensorConvertible.Aux[T, D]): Tensor[D] = {
    ev.toTensor(value)
  }

  // TODO: !!! [TYPES] Add support for more lossless conversions.

  //region FLOAT32 Conversions

  implicit def float32ToFloat64[TL[D] <: TensorLike[D]](value: TL[FLOAT32])(implicit
      ev: TensorOps.Aux[TL, FLOAT32]
  ): TL[FLOAT64] = Math.cast(value, FLOAT64)

  //endregion FLOAT32 Conversions

  implicit def tensorToBasicOps[D <: DataType](value: Tensor[D]): BasicOps[D] = new BasicOps(value)
  implicit def tensorToNumericBasicOps[D <: NumericDataType](value: Tensor[D]): NumericBasicOps[D] = new NumericBasicOps(value)
  implicit def tensorToDecimalBasicOps[D <: DecimalDataType](value: Tensor[D]): DecimalBasicOps[D] = new DecimalBasicOps(value)
  implicit def tensorToInt32OrInt64BasicOps[D <: Int32OrInt64](value: Tensor[D]): Int32OrInt64BasicOps[D] = new Int32OrInt64BasicOps(value)
  implicit def tensorToBooleanBasicOps(value: Tensor[BOOLEAN]): BooleanBasicOps = new BooleanBasicOps(value)
  implicit def tensorConvertibleToBasicOps[D <: DataType, T](value: T)(implicit f: T => Tensor[D]): BasicOps[D] = new BasicOps(f(value))
  implicit def tensorConvertibleToNumericBasicOps[D <: NumericDataType, T](value: T)(implicit f: T => Tensor[D]): NumericBasicOps[D] = new NumericBasicOps(f(value))
  implicit def tensorConvertibleToDecimalBasicOps[D <: DecimalDataType, T](value: T)(implicit f: T => Tensor[D]): DecimalBasicOps[D] = new DecimalBasicOps(f(value))
  implicit def tensorConvertibleToInt32OrInt64BasicOps[D <: Int32OrInt64, T](value: T)(implicit f: T => Tensor[D]): Int32OrInt64BasicOps[D] = new Int32OrInt64BasicOps(f(value))
  implicit def tensorConvertibleToBooleanBasicOps[T](value: T)(implicit f: T => Tensor[BOOLEAN]): BooleanBasicOps = new BooleanBasicOps(f(value))
}
