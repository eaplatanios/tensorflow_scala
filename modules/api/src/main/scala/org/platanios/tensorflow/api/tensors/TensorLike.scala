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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types.DataType

/** Represents tensor-like objects.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorLike[T] {
  /** Data type of this tensor. */
  val dataType: DataType[T]

  /** Shape of this tensor. */
  val shape: Shape

  /** Device on which this tensor is stored. */
  val device: String

  /** Returns the tensor that this tensor-like object represents. */
  def toTensor: Tensor[T]

  /** Returns the tensor indexed slices that has the same value as this tensor-like object. */
  def toTensorIndexedSlices: TensorIndexedSlices[T]
}
