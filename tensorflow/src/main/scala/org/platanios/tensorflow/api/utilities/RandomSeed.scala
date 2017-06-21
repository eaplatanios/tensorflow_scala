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

package org.platanios.tensorflow.api.utilities

import org.platanios.tensorflow.api.tf.defaultGraph

/**
  * @author Sören Brunk
  */
object RandomSeed {
  val defaultGraphSeed: Int = 87654321

  /**
    * Returns the local seeds an operation should use given an op-specific seed.
    *
    * Given operation-specific seed, opSeed, this helper function returns two
    * seeds derived from graph-level and op-level seeds. Many random operations
    * internally use the two seeds to allow user to change the seed globally for a
    * graph, or for only specific operations.
    *
    * @param  opSeed The op-specific seed
    *
    * @return A tuple of two integers that should be used for the local seed of this operation.
    */
  def getSeed(opSeed: Option[Int]): (Option[Int], Option[Int]) = {

    val graphSeed: Option[Int] = Some(defaultGraphSeed) // TODO use seed from current default graph once implemented

    if (graphSeed.isEmpty && opSeed.isEmpty)(None, None)
    else {
      val (seed1, seed2) = (graphSeed.getOrElse(defaultGraphSeed), opSeed.getOrElse(defaultGraph.ops.length))
      if ((seed1, seed2) == (0, 0)) (Some(seed1), Some(Int.MaxValue))
      else (Some(seed1), Some(seed2))
    }
  }

  /** Sets the graph-level random seed.
    *
    * Operations that rely on a random seed actually derive it from two seeds:
    * the graph-level and operation-level seeds. This sets the graph-level seed.
    *
    * Its interactions with operation-level seeds is as follows:
    *
    *   1. If neither the graph-level nor the operation seed is set:
    *     A random seed is used for this op.
    *   2. If the graph-level seed is set, but the operation seed is not:
    *     The system deterministically picks an operation seed in conjunction
    *     with the graph-level seed so that it gets a unique random sequence.
    *   3. If the graph-level seed is not set, but the operation seed is set:
    *     A default graph-level seed and the specified operation seed are used to
    *     determine the random sequence.
    *   4. If both the graph-level and the operation seed are set:
    *     Both seeds are used in conjunction to determine the random sequence.
    *
    * @param seed The new graph-level random seed.
    */
  def setRandomSeed(seed: Int): Unit = ???
}
