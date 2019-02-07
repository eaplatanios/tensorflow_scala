/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.jni.generated.tensors.{Random => NativeTensorOpsRandom}

/** Contains functions for executing ops related to random numbers and tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Random {
  /** $OpDocRandomRandomShuffle
    *
    * @group RandomOps
    * @param  value Tensor to be shuffled.
    * @param  seed  Optional random seed, used to generate a random seed pair for the random number generator, when
    *               combined with the graph-level seed.
    * @return Result as a new tensor.
    */
  def randomShuffle[T: TF](
      value: Tensor[T],
      seed: Option[Int] = None
  ): Tensor[T] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomShuffle(
      executionContext.value.nativeHandle, value.nativeHandle,
      graphSeed.getOrElse(0).toLong, opSeed.getOrElse(0).toLong))
  }
}

object Random extends Random
