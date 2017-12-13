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
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, Op, Output, Symbol}
import org.platanios.tensorflow.api.types.{DataType, INT32}

import scala.language.postfixOps

/** Base class for attention mechanisms.
  *
  * This class provides some common functionality which includes:
  *
  *   - Storing the query and memory layers.
  *   - Pre-processing and storing the memory.
  *
  * @param  memory                      Memory to query; usually the output of an RNN encoder. Each tensor in the memory
  *                                     should be shaped `[batchSize, maxTime, ...]`.
  * @param  probabilityFn               Function that takes the scores and the previous alignment as inputs and produces
  *                                     probabilities.
  * @param  memorySequenceLengths       Sequence lengths for the batch entries in the memory. If provided, the memory
  *                                     tensor rows are masked with zeros for values past the respective sequence
  *                                     lengths.
  * @param  queryFn                     Function used to process the input queries, before using them to query the
  *                                     memory.
  * @param  checkInnerDimensionsDefined If `true`, the `memory` argument's shape is checked to ensure that all but the
  *                                     two outermost dimensions of each tensor are fully defined.
  * @param  scoreMaskValue              Mask value to use for the score before passing it to `probabilityFn`. Defaults
  *                                     to negative infinity. Note that this value is only used if
  *                                     `memorySequenceLengths` is not `null`.
  * @param  name                        Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class AttentionMechanism[M, MS](
    protected val memory: M,
    protected val probabilityFn: (Output, Output) => Output,
    val memorySequenceLengths: Output = null,
    val queryFn: Output => Output = (o: Output) => o,
    val checkInnerDimensionsDefined: Boolean = true,
    val scoreMaskValue: Float = Float.NegativeInfinity,
    val name: String = "AttentionMechanism"
)(implicit evM: WhileLoopVariable.Aux[M, MS]) {
  /** Function to apply on the memory before using it for queries. Defaults to the identity function. */
  val memoryFn: M => M = (m: M) => m

  val trainableVariables: Set[Variable] = Set.empty

  val nonTrainableVariables: Set[Variable] = Set.empty

  val (dataType, keys, values, batchSize, alignmentsSize): (DataType, M, M, Output, Output) = {
    Op.createWithNameScope(s"$name/Initialization") {
      val values = evM.map(
        memory, (s: Symbol) => {
          AttentionMechanism.maybeMaskMemory(
            s.asInstanceOf[Output], memorySequenceLengths, checkInnerDimensionsDefined).asInstanceOf[Symbol]
        })
      val keys = memoryFn(values)
      val flattenedKeys = evM.outputs(keys)
      val batchSize = {
        if (flattenedKeys.head.shape(0) != -1)
          Basic.constant(flattenedKeys.head.shape(0))
        else
          Basic.shape(flattenedKeys.head)(0)
      }
      val alignmentsSize = {
        if (flattenedKeys.head.shape(1) != -1)
          Basic.constant(flattenedKeys.head.shape(1))
        else
          Basic.shape(flattenedKeys.head)(1)
      }
      val dataType = flattenedKeys.head.dataType
      require(flattenedKeys.forall(_.dataType == dataType), "All memory keys need to have the same data type.")
      (dataType, keys, values, batchSize, alignmentsSize)
    }
  }

  protected val scoreMaskValueOutput: Output = Basic.constant(scoreMaskValue, dataType)

  protected val probability: (Output, Output) => Output = (score: Output, previousAlignment: Output) => {
    probabilityFn(
      AttentionMechanism.maybeMaskScore(score, memorySequenceLengths, scoreMaskValueOutput), previousAlignment)
  }

  /** Creates the initial alignment values.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return a tensor of all zeros.
    *
    * @param  batchSize `INT32` scalar containing the batch size.
    * @param  dataType  Data type for the alignments tensor.
    * @return Initial alignment tensor.
    */
  def initialAlignments(batchSize: Output, dataType: DataType): Output = {
    Op.createWithNameScope(s"$name/InitialAlignments", Set(batchSize.op)) {
      val fullShape = Basic.concatenate(Seq(batchSize, alignmentsSize.cast(batchSize.dataType)), axis = 0)
      Basic.zeros(dataType, fullShape)
    }
  }

  /** Computes the next alignments given the current query and the previous alignments.
    *
    * @param  query              Query.
    * @param  previousAlignments Previous alignment.
    * @return Next alignment.
    */
  def call(query: Output, previousAlignments: Output): Output

  /** Computes the next alignments given the current query and the previous alignments.
    *
    * @param  query              Query.
    * @param  previousAlignments Previous alignment.
    * @return Next alignment.
    */
  def apply(query: Output, previousAlignments: Output): Output = call(query, previousAlignments)
}

object AttentionMechanism {
  /** Potentially masks the provided memory tensor based on the provided sequence lengths. */
  @throws[InvalidShapeException]
  private[AttentionMechanism] def maybeMaskMemory(
      memory: Output, sequenceLengths: Output, checkInnerDimensionsDefined: Boolean
  ): Output = {
    if (checkInnerDimensionsDefined && !memory.shape(2 ::).isFullyDefined)
      throw InvalidShapeException(
        s"Expected memory '${memory.name}' to have fully defined inner dimensions, but saw shape: ${memory.shape}.")
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
        (batchSize, Basic.sequenceMask(sequenceLengths, Basic.shape(memory)(1), memory.dataType))
      }
    }
    if (sequenceMask == null) {
      memory
    } else {
      val rank = if (memory.rank != -1) Basic.constant(memory.rank) else Basic.rank(memory)
      val extraOnes = Basic.ones(INT32, rank - 2)
      val mBatchSize = if (memory.shape(0) != -1) Basic.constant(memory.shape(0)) else Basic.shape(memory)(0)
      Op.createWith(controlDependencies = Set(Checks.assertEqual(
        batchSize, mBatchSize,
        "The memory tensor batch sizes do not match with the provided sequence lengths batch size."))) {
        val mask = sequenceMask.reshape(Basic.concatenate(Seq(Basic.shape(sequenceMask), extraOnes), 0))
        memory * mask
      }
    }
  }

  /** Potentially masks the provided score tensor based on the provided sequence lengths. */
  private[AttentionMechanism] def maybeMaskScore(
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
