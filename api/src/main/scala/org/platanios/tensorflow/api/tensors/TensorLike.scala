/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

/**
  * @author Emmanouil Antonios Platanios
  */
trait TensorLike extends TensorConvertible {
  val dataType: DataType
  val shape   : Shape

  require(shape.isFullyDefined, s"The shape of a Tensor object must be fully defined. Shape '$shape' is not.")
  // require(shape.numElements > 0, "Empty tensors are not supported in the TensorFlow Scala API.")

  def rank: Int = shape.rank
  def numElements: Int = shape.numElements

  def summarize(maxEntries: Int = 6): String
}
