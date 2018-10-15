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

package org.platanios.tensorflow.api.ops.basic

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

/** Contains ops related to in-place tensor operations.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Inplace {
  def deepCopy[T: TF](
      x: Output[T],
      name: String = "DeepCopy"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "DeepCopy",
      input = x,
      name = name
    ).build().output
  }

  /** Applies an inplace update on `x`.
    *
    * If `i` is `None`, then `x` and `v` must have the same shape, and the op computes:
    *   `x opType v`
    * Else, if `i` is a scalar, `x` must have rank 1 higher than that of `v`, and the op computes:
    *   `x(i, :) opType v`
    * Else, `x` and `v` must have the same rank, and the op computes:
    *   `x(i, :) opType v`
    *
    * @param  x      Input tensor to be updated.
    * @param  i      Optional indices for the update.
    * @param  v      Values used for the update.
    * @param  opType Inplace op to use for the update.
    * @param  name   Name for the created op.
    * @tparam T Tensor data type.
    * @return Output of the update op, which is simply an alias for `x`.
    */
  private def inplaceHelper[T: TF](
      x: Output[T],
      i: Option[Output[Int]],
      v: Output[T],
      opType: String,
      name: String
  ): Output[T] = {
    i match {
      case Some(index) if index.rank == 0 =>
        // Single 0-dim update.
        val reshapedIndex = Basic.reshape(index, Tensor[Int](1))
        val reshapedValues = Basic.expandDims(v, axis = 0)
        Op.Builder[(Output[T], Output[Int], Output[T]), Output[T]](
          opType = opType,
          name = name,
          input = (x, reshapedIndex, reshapedValues)
        ).build().output
      case Some(index) =>
        Op.Builder[(Output[T], Output[Int], Output[T]), Output[T]](
          opType = opType,
          name = name,
          input = (x, index, v)
        ).build().output
      case None =>
        // Full tensor.
        val reshapedInput = Basic.reshape(x, Tensor[Int](1, -1))
        val reshapedValue = Basic.reshape(v, Tensor[Int](1, -1))
        val indices = Tensor[Int](0)
        val result = Op.Builder[(Output[T], Output[Int], Output[T]), Output[T]](
          opType = opType,
          name = name,
          input = (reshapedInput, indices, reshapedValue)
        ).build().output
        Basic.reshape(result, Basic.shape(x))
    }
  }

  /** Applies an inplace update on `x`.
    *
    * If `i` is `None`, then `x` and `v` must have the same shape, and the op computes:
    *   `x = v`
    * Else, if `i` is a scalar, `x` must have rank 1 higher than that of `v`, and the op computes:
    *   `x(i, :) = v`
    * Else, `x` and `v` must have the same rank, and the op computes:
    *   `x(i, :) = v`
    *
    * '''NOTE:''' If the purpose is simply to perform sparse updates and not avoid copying memory, then first perform
    * a `deepCopy` of `x` and then apply the inplace update on that copy.
    *
    * @param  x    Input tensor to be updated.
    * @param  i    Optional indices for the update.
    * @param  v    Values used for the update.
    * @param  name Name for the created op.
    * @tparam T Tensor data type.
    * @return Output of the update op, which is simply an alias for `x`.
    */
  def inplaceUpdate[T: TF](
      x: Output[T],
      i: Option[Output[Int]],
      v: Output[T],
      name: String = "InplaceUpdate"
  ): Output[T] = {
    inplaceHelper(x, i, v, opType = "InplaceUpdate", name = name)
  }

  /** Applies an inplace update on `x`.
    *
    * If `i` is `None`, then `x` and `v` must have the same shape, and the op computes:
    *   `x += v`
    * Else, if `i` is a scalar, `x` must have rank 1 higher than that of `v`, and the op computes:
    *   `x(i, :) += v`
    * Else, `x` and `v` must have the same rank, and the op computes:
    *   `x(i, :) += v`
    *
    * '''NOTE:''' If the purpose is simply to perform sparse updates and not avoid copying memory, then first perform
    * a `deepCopy` of `x` and then apply the inplace update on that copy.
    *
    * @param  x    Input tensor to be updated.
    * @param  i    Optional indices for the update.
    * @param  v    Values used for the update.
    * @param  name Name for the created op.
    * @tparam T Tensor data type.
    * @return Output of the update op, which is simply an alias for `x`.
    */
  def inplaceAdd[T: TF](
      x: Output[T],
      i: Option[Output[Int]],
      v: Output[T],
      name: String = "InplaceAdd"
  ): Output[T] = {
    inplaceHelper(x, i, v, opType = "InplaceAdd", name = name)
  }

  /** Applies an inplace update on `x`.
    *
    * If `i` is `None`, then `x` and `v` must have the same shape, and the op computes:
    *   `x -= v`
    * Else, if `i` is a scalar, `x` must have rank 1 higher than that of `v`, and the op computes:
    *   `x(i, :) -= v`
    * Else, `x` and `v` must have the same rank, and the op computes:
    *   `x(i, :) -= v`
    *
    * '''NOTE:''' If the purpose is simply to perform sparse updates and not avoid copying memory, then first perform
    * a `deepCopy` of `x` and then apply the inplace update on that copy.
    *
    * @param  x    Input tensor to be updated.
    * @param  i    Optional indices for the update.
    * @param  v    Values used for the update.
    * @param  name Name for the created op.
    * @tparam T Tensor data type.
    * @return Output of the update op, which is simply an alias for `x`.
    */
  def inplaceSubtract[T: TF](
      x: Output[T],
      i: Option[Output[Int]],
      v: Output[T],
      name: String = "InplaceSubtract"
  ): Output[T] = {
    inplaceHelper(x, i, v, opType = "InplaceSub", name = name)
  }
}

object Inplace extends Inplace
