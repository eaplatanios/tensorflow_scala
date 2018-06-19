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

import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.TensorConvertible
import org.platanios.tensorflow.api.tensors.ops.Basic.BasicOps
import org.platanios.tensorflow.api.tensors.ops.Math.MathOps
import org.platanios.tensorflow.api.tensors.ops.NN.NNOps

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[implicits] trait TensorImplicits {
  implicit def tensorConvertibleToTensor[T](value: T)(implicit ev: TensorConvertible[T]): tensors.Tensor = ev.toTensor(value)

  implicit def tensorToBasicOps(value: tensors.Tensor): BasicOps = BasicOps(value)
  implicit def tensorConvertibleToBasicOps[T](value: T)(implicit f: T => tensors.Tensor): BasicOps = BasicOps(f(value))

  implicit def tensorToMathOps(value: tensors.Tensor): MathOps = MathOps(value)
  implicit def tensorConvertibleToMathOps[T](value: T)(implicit f: T => tensors.Tensor): MathOps = MathOps(f(value))

  implicit def tensorToNNOps(value: tensors.Tensor): NNOps = NNOps(value)
  implicit def tensorConvertibleToNNOps[T](value: T)(implicit f: T => tensors.Tensor): NNOps = NNOps(f(value))
}
