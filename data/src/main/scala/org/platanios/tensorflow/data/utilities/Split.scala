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

package org.platanios.tensorflow.data.utilities

import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
trait Split {
  val seed: Option[Long]

  protected lazy val random: Random = {
    seed.map(new Random(_)).getOrElse(new Random())
  }

  @throws[IllegalArgumentException]
  def apply(trainPortion: Float): (Seq[Int], Seq[Int])
}

case class UniformSplit(
    numSamples: Int,
    override val seed: Option[Long] = None
) extends Split {
  @throws[IllegalArgumentException]
  override def apply(trainPortion: Float): (Seq[Int], Seq[Int]) = {
    require(trainPortion >= 0.0f && trainPortion <= 1.0f, "'trainPortion' must be in [0.0f, 1.0f].")
    val permutedIndices = random.shuffle[Int, Seq](0 until numSamples)
    val numTrainSamples = math.floor(numSamples * trainPortion).toInt
    (permutedIndices.take(numTrainSamples), permutedIndices.drop(numTrainSamples))
  }
}

case class UniformStratifiedSplit(
    labels: Seq[Int],
    override val seed: Option[Long] = None
) extends Split {
  @throws[IllegalArgumentException]
  override def apply(trainPortion: Float): (Seq[Int], Seq[Int]) = {
    require(trainPortion >= 0.0f && trainPortion <= 1.0f, "'trainPortion' must be in [0.0f, 1.0f].")
    val (trainIndices, testIndices) = labels.zipWithIndex.groupBy(_._1).mapValues(_.map(_._2)).map({
      case (_, indices) =>
        val permutedIndices = random.shuffle[Int, Seq](indices.indices)
        val numTrainSamples = math.floor(indices.size * trainPortion).toInt
        (permutedIndices.take(numTrainSamples), permutedIndices.drop(numTrainSamples))
    }).unzip
    (trainIndices.flatten.toSeq, testIndices.flatten.toSeq)
  }
}
