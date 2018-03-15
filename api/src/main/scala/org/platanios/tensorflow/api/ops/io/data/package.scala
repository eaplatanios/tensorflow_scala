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

package org.platanios.tensorflow.api.ops.io

import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.INT64

/**
  * @author Emmanouil Antonios Platanios
  */
package object data {
  /** Returns the local random seeds an op should use, given an optionally provided op-specific seed.
    *
    * @param  seed Optionally provided op-specific seed.
    * @param  name Name prefix for all created ops.
    * @return Local random seeds to use.
    */
  private[data] def randomSeeds(seed: Option[Int] = None, name: String = "RandomSeeds"): (Output, Output) = {
    Op.createWithNameScope(name) {
      val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
      val seed1 = Basic.constant(graphSeed.getOrElse(0), INT64)
      val seed2 = opSeed match {
        case None => Basic.constant(0, INT64)
        case Some(s) => Op.createWithNameScope("Seed2") {
          val seed2 = Basic.constant(s, INT64)
          Math.select(
            Math.logicalAnd(Math.equal(seed1, 0), Math.equal(seed2, 0)),
            Basic.constant(Int.MaxValue, INT64),
            seed2)
        }
      }
      (seed1, seed2)
    }
  }
}
