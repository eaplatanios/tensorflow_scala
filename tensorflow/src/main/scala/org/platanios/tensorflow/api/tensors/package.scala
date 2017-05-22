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

package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
package object tensors {
  private[api] trait API {
    type Tensor = tensors.Tensor
    type FixedSizeTensor = tensors.FixedSizeTensor
    type NumericTensor = tensors.NumericTensor

    val Tensor = tensors.Tensor

    type Order = tensors.Order
    val RowMajorOrder = tensors.RowMajorOrder
  }
}
