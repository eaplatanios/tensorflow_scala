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

import org.platanios.tensorflow.api.tensors.{Tensor, TensorConvertible, TensorLike}
import org.platanios.tensorflow.api.types._

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits extends LowPriorityImplicits {
  implicit def outputConvertibleToOutput[OC, T](
      value: OC
  )(implicit ev: OutputConvertible.Aux[OC, T]): Output[T] = {
    ev.toOutput(value)
  }

  // TODO: !!! [TYPES] This does not currently lets us cast an S tensor to a T tensor, when S is preceding (e.g., `Tensor[S] + Tensor[T]` fails to compile).

  //  implicit def cast[SOURCE, TARGET, OL[A] <: OutputLike[A]](value: OL[SOURCE])(implicit
  //      evAllowedCast: AllowedCast.Aux[SOURCE, TARGET],
  //      ev: OutputOps.Aux[OL, SOURCE]
  //  ): OL[TARGET] = Cast.cast(value, evAllowedCast.targetDataType)
}

private[api] trait LowPriorityImplicits
    extends control_flow.Implicits
        with Basic.Implicits
        with Cast.Implicits
        with Clip.Implicits
        with Embedding.Implicits
        with Math.Implicits
        with NN.Implicits
        with Sparse.Implicits
        with Statistics.Implicits
        with Text.Implicits
