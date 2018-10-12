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

package org.platanios.tensorflow.api.ops.rnn.attention

import cats.data.Nested
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure.Aux
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output}

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
abstract class Attention[T: TF : IsDecimal, State](
    protected val memory: Output[T],
    protected val memorySequenceLengths: Output[Int] = null,
    val checkInnerDimensionsDefined: Boolean = true,
    val scoreMaskValue: Output[Float] = Float.MinValue,
    val name: String = "Attention"
) {
  lazy val values: Output[T] = {
    Op.nameScope(s"$name/Values") {
      Attention.maybeMaskValues(memory, memorySequenceLengths, checkInnerDimensionsDefined)
    }
  }

  lazy val keys: Output[T] = {
    values
  }

  lazy val batchSize: Output[Int] = {
    Op.nameScope(s"$name/BatchSize") {
      Attention.dimSize(keys, axis = 0)
    }
  }

  lazy val alignmentSize: Output[Int] = {
    Op.nameScope(s"$name/AlignmentSize") {
      Attention.dimSize(keys, axis = 1)
    }
  }

  def stateSize[V, D, S](implicit evStructureState: NestedStructure.Aux[State, V, D, S]): S

  lazy val dataType: DataType[T] = {
    keys.dataType
  }

  /** Initial alignment value.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return a tensor of all zeros.
    */
  lazy val initialAlignment: Output[T] = {
    Op.nameScope(s"$name/InitialAlignment") {
      val fullShape = Basic.stack(
        Seq(batchSize, alignmentSize.castTo[Int]),
        axis = 0)
      Basic.zeros[T, Int](fullShape)
    }
  }

  /** Initial state value.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return the same output as `initialAlignment`.
    */
  def initialState: State

  /** Computes an alignment tensor given the provided query and previous alignment tensor.
    *
    * The previous alignment tensor is important for attention mechanisms that use the previous alignment to calculate
    * the attention at the next time step, such as monotonic attention mechanisms.
    *
    * TODO: Figure out how to generalize the "next state" functionality.
    *
    * @param  query         Query tensor.
    * @param  previousState Previous alignment tensor.
    * @return Tuple containing the alignment tensor and the next attention state.
    */
  def alignment(query: Output[T], previousState: State): (Output[T], State)

  /** Computes an alignment score for `query`.
    *
    * @param  query Query tensor.
    * @param  state Current attention mechanism state (defaults to the previous alignment tensor). The data type of
    *               this tensor matches that of `values` and its shape is `[batchSize, alignmentSize]`, where
    *               `alignmentSize` is the memory's maximum time.
    * @return Score tensor.
    */
  protected def score(query: Output[T], state: State): Output[T]

  /** Computes alignment probabilities for `score`.
    *
    * @param  score Alignment score tensor.
    * @param  state Current attention mechanism state (defaults to the previous alignment tensor). The data type of
    *               this tensor matches that of `values` and its shape is `[batchSize, alignmentSize]`, where
    *               `alignmentSize` is the memory's maximum time.
    * @return Alignment probabilities tensor.
    */
  protected def probability(score: Output[T], state: State): Output[T] = {
    NN.softmax(score, name = "Probability")
  }
}

/** Base class for attention models that use as state the previous alignment. */
abstract class SimpleAttention[T: TF : IsDecimal](
    override protected val memory: Output[T],
    override protected val memorySequenceLengths: Output[Int] = null,
    override val checkInnerDimensionsDefined: Boolean = true,
    override val scoreMaskValue: Output[Float] = Float.MinValue,
    override val name: String = "SimpleAttention"
) extends Attention[T, Output[T]](
  memory = memory,
  memorySequenceLengths = memorySequenceLengths,
  checkInnerDimensionsDefined = checkInnerDimensionsDefined,
  scoreMaskValue = scoreMaskValue,
  name = name
) {
  override def stateSize[V, D, S](implicit evStructureState: Aux[Output[T], V, D, S]): S = {
    Output.constantValueAsShape(alignmentSize).getOrElse(Shape.unknown()).asInstanceOf[S]
  }

  override def initialState: Output[T] = {
    Op.nameScope(s"$name/InitialState") {
      Basic.identity(initialAlignment)
    }
  }

  override def alignment(
      query: Output[T],
      previousState: Output[T]
  ): (Output[T], Output[T]) = {
    Op.nameScope(name) {
      val unmaskedScore = score(query, previousState)
      val maskedScore = Attention.maybeMaskScore(
        unmaskedScore, memorySequenceLengths, scoreMaskValue.castTo[T])
      val alignment = probability(maskedScore, previousState)
      (alignment, alignment)
    }
  }
}

object Attention {
  private[attention] def dimSize[T: TF](
      value: Output[T],
      axis: Int
  ): Output[Int] = {
    if (value.rank != -1 && value.shape(axis) != -1)
      Basic.constant(value.shape(axis))
    else
      Basic.shape(value).castTo[Int].slice(axis)
  }

  /** Potentially masks the provided values tensor based on the provided sequence lengths. */
  @throws[InvalidShapeException]
  private[attention] def maybeMaskValues[T: TF : IsNotQuantized](
      values: Output[T],
      sequenceLengths: Output[Int],
      checkInnerDimensionsDefined: Boolean
  ): Output[T] = {
    if (checkInnerDimensionsDefined && !values.shape(2 ::).isFullyDefined) {
      throw InvalidShapeException(
        s"Expected memory '${values.name}' to have fully defined " +
            s"inner dimensions, but saw shape: ${values.shape}.")
    }
    val sequenceMask = {
      if (sequenceLengths == null) {
        null
      } else {
        Basic.sequenceMask(
          sequenceLengths,
          Basic.shape(values).castTo[Int].slice(1)
        ).castTo[T]
      }
    }
    if (sequenceMask == null) {
      values
    } else {
      val rank = if (values.rank != -1) Basic.constant(values.rank) else Basic.rank(values)
      val extraOnes = Basic.ones[Int, Int](Basic.expandDims(rank - 2, 0))
      val mask = sequenceMask.reshape(
        Basic.concatenate(Seq(
          Basic.shape(sequenceMask).castTo[Int],
          extraOnes
        ), axis = 0))
      values * mask
    }
  }

  /** Potentially masks the provided score tensor based on the provided sequence lengths. */
  private[attention] def maybeMaskScore[T: TF : IsNotQuantized](
      score: Output[T],
      sequenceLengths: Output[Int],
      scoreMaskValue: Output[T]
  ): Output[T] = {
    if (sequenceLengths != null) {
      score
    } else {
      val scoreMask = Basic.sequenceMask(
        sequenceLengths,
        Basic.shape(score).castTo[Int].slice(1))
      Math.select(scoreMask, score, scoreMaskValue * Basic.onesLike(score))
    }
  }
}
