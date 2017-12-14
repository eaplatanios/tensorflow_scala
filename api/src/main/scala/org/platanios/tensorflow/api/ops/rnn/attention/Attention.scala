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

package org.platanios.tensorflow.api.ops.rnn.attention

import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, NN, Op, Output}
import org.platanios.tensorflow.api.types.{DataType, INT32}

import scala.language.postfixOps

/** Base class for attention mechanisms.
  *
  * @param  memory                      Memory to query; usually the output of an RNN encoder. Each tensor in the memory
  *                                     should be shaped `[batchSize, maxTime, ...]`.
  * @param  memorySequenceLengths       Sequence lengths for the batch entries in the memory. If provided, the memory
  *                                     tensor rows are masked with zeros for values past the respective sequence
  *                                     lengths.
  * @param  checkInnerDimensionsDefined If `true`, the `memory` argument's shape is checked to ensure that all but the
  *                                     two outermost dimensions of each tensor are fully defined.
  * @param  scoreMaskValue              Scalar tensor containing the mask value to use for the attention scores before
  *                                     passing them to `probability`. Defaults to negative infinity. Note that this
  *                                     value is only used if `memorySequenceLengths` is not `null`.
  * @param  name                        Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Attention(
    protected val memory: Output,
    protected val memorySequenceLengths: Output = null,
    val checkInnerDimensionsDefined: Boolean = true,
    val scoreMaskValue: Output = Float.NegativeInfinity,
    val name: String = "Attention"
) {
  val values: Output = Op.createWithNameScope(s"$name/Initialization") {
    Attention.maybeMaskValues(memory, memorySequenceLengths, checkInnerDimensionsDefined)
  }

  val keys: Output = values

  val batchSize: Output = Op.createWithNameScope(s"$name/Initialization") {
    Attention.dimSize(keys, 0)
  }

  val alignmentSize: Output = Op.createWithNameScope(s"$name/Initialization") {
    Attention.dimSize(keys, 1)
  }

  val dataType: DataType = keys.dataType

  /** Initial alignment value.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return a tensor of all zeros.
    */
  val initialAlignment: Output = {
    Op.createWithNameScope(s"$name/InitialAlignments", Set(batchSize.op)) {
      val fullShape = Basic.concatenate(Seq(batchSize, alignmentSize.cast(batchSize.dataType)), axis = 0)
      Basic.zeros(dataType, fullShape)
    }
  }

  /** Computes an alignment tensor given the provided query and previous alignment tensor.
    *
    * The previous alignment tensor is important for attention mechanisms that use the previous alignment to calculate
    * the attention at the next time step, such as monotonic attention mechanisms.
    *
    * @param  query             Query tensor.
    * @param  previousAlignment Previous alignment tensor.
    * @return Alignment tensor.
    */
  final def alignment(query: Output, previousAlignment: Output): Output = Op.createWithNameScope(name) {
    val unmaskedScore = score(query, previousAlignment)
    val maskedScore = Attention.maybeMaskScore(unmaskedScore, memorySequenceLengths, scoreMaskValue)
    probability(maskedScore, previousAlignment)
  }

  /** Computes an alignment score for `query`.
    *
    * @param  query             Query tensor.
    * @param  previousAlignment Previous alignment tensor.
    * @return Score tensor.
    */
  protected def score(query: Output, previousAlignment: Output): Output

  /** Computes alignment probabilities for `score`.
    *
    * @param  score             Alignment score tensor.
    * @param  previousAlignment Previous alignment tensor.
    * @return Alignment probabilities tensor.
    */
  protected def probability(score: Output, previousAlignment: Output): Output = NN.softmax(score, name = "Probability")
}

object Attention {
  private[attention] def dimSize(value: Output, axis: Int): Output = {
    if (value.rank != -1 && value.shape(axis) != -1)
      Basic.constant(value.shape(axis))
    else
      Basic.shape(value)(axis)
  }

  /** Potentially masks the provided values tensor based on the provided sequence lengths. */
  @throws[InvalidShapeException]
  private[Attention] def maybeMaskValues(
      values: Output, sequenceLengths: Output, checkInnerDimensionsDefined: Boolean
  ): Output = {
    if (checkInnerDimensionsDefined && !values.shape(2 ::).isFullyDefined)
      throw InvalidShapeException(
        s"Expected memory '${values.name}' to have fully defined inner dimensions, but saw shape: ${values.shape}.")
    val (batchSize, sequenceMask) = {
      if (sequenceLengths == null) {
        (null, null)
      } else {
        val batchSize = {
          if (sequenceLengths.shape(0) != -1)
            Basic.constant(sequenceLengths.shape(0))
          else
            Basic.shape(sequenceLengths)(0)
        }
        (batchSize, Basic.sequenceMask(sequenceLengths, Basic.shape(values)(1), values.dataType))
      }
    }
    if (sequenceMask == null) {
      values
    } else {
      val rank = if (values.rank != -1) Basic.constant(values.rank) else Basic.rank(values)
      val extraOnes = Basic.ones(INT32, rank - 2)
      val mBatchSize = if (values.shape(0) != -1) Basic.constant(values.shape(0)) else Basic.shape(values)(0)
      Op.createWith(controlDependencies = Set(Checks.assertEqual(
        batchSize, mBatchSize,
        "The memory tensor batch sizes do not match with the provided sequence lengths batch size."))) {
        val mask = sequenceMask.reshape(Basic.concatenate(Seq(Basic.shape(sequenceMask), extraOnes), 0))
        values * mask
      }
    }
  }

  /** Potentially masks the provided score tensor based on the provided sequence lengths. */
  private[Attention] def maybeMaskScore(
      score: Output, sequenceLengths: Output, scoreMaskValue: Output
  ): Output = {
    if (sequenceLengths != null) {
      score
    } else {
      Op.createWith(controlDependencies = Set(Checks.assertNonPositive(
        sequenceLengths, "All provided in memory sequence lengths must greater than zero."))) {
        val scoreMask = Basic.sequenceMask(sequenceLengths, Basic.shape(score)(1))
        Math.select(scoreMask, score, scoreMaskValue * Basic.onesLike(score))
      }
    }
  }
}
