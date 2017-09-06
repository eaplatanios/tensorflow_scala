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

import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
package object ops {
  @inline private[ops] def castArgs(tensor1: Tensor, tensor2: Tensor): (Tensor, Tensor) = {
    val dataType = DataType.mostPrecise(tensor1.dataType, tensor2.dataType)
    (tensor1.cast(dataType), tensor2.cast(dataType))
  }

  @inline private[ops] def castArgs(tensor1: Tensor, tensor2: Tensor, tensor3: Tensor): (Tensor, Tensor, Tensor) = {
    val dataType = DataType.mostPrecise(tensor1.dataType, tensor2.dataType, tensor3.dataType)
    (tensor1.cast(dataType), tensor2.cast(dataType), tensor3.cast(dataType))
  }

  @inline private[ops] def castArgs(tensors: Seq[Tensor]): Seq[Tensor] = {
    val dataType = DataType.mostPrecise(tensors.map(_.dataType): _*)
    tensors.map(_.cast(dataType))
  }

  private[api] trait API
      extends Basic
          with Math
}
