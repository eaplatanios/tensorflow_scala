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

package org.platanios.tensorflow.api.ops.rnn.attention

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.OutputToShape
import org.platanios.tensorflow.api.ops.{NN, Op, Output}
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.math.Math

import scala.language.postfixOps

/** Base class for attention mechanisms.
  *
  * @param  memorySize     Size of the memory over which attention is defined.
  * @param  scoreMaskValue Scalar tensor containing the mask value to use for the attention scores before passing them
  *                        to `probability`. Defaults to negative infinity. Note that this value is only used if
  *                        `memorySequenceLengths` is not `null`.
  * @param  name           Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Attention[T: TF : IsDecimal, State, StateShape](
    val memorySize: Output[Int],
    val name: String = "Attention"
)(implicit evOutputToShapeState: OutputToShape.Aux[State, StateShape]) {
  def keysShape(valuesShape: Shape): Shape

  def stateShape: StateShape

  /** Initial alignment value.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return a tensor of all zeros.
    *
    * @param  batchSize Batch size.
    */
  def initialAlignment(batchSize: Output[Int]): Output[T] = {
    Op.nameScope(s"$name/InitialAlignment") {
      val fullShape = Basic.stack(Seq(batchSize, memorySize), axis = 0)
      Basic.zeros[T, Int](fullShape)
    }
  }

  /** Initial state value.
    *
    * This is important for attention mechanisms that use the previous alignment to calculate the alignment at the
    * next time step (e.g., monotonic attention).
    *
    * The default behavior is to return the same output as `initialAlignment`.
    *
    * @param  batchSize Batch size.
    */
  def initialState(
      batchSize: Output[Int],
      memory: Attention.Memory[T]
  ): Attention.State[T, State]

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
  def alignment(
      query: Output[T],
      previousState: Attention.State[T, State]
  ): (Output[T], Attention.State[T, State])
}

/** Base class for attention models that use as state the previous alignment. */
abstract class SimpleAttention[T: TF : IsDecimal](
    override val memorySize: Output[Int],
    val scoreMaskValue: Output[Float] = Float.MinValue,
    override val name: String = "SimpleAttention"
) extends Attention[T, Output[T], Shape](
  memorySize = memorySize,
  name = name
) {
  override def stateShape: Shape = {
    Output.constantValueAsShape(memorySize).getOrElse(Shape.unknown())
  }

  override def initialState(
      batchSize: Output[Int],
      memory: Attention.Memory[T]
  ): Attention.State[T, Output[T]] = {
    Op.nameScope(s"$name/InitialState") {
      val values = this.values(memory)
      val keys = this.keys(memory, values)
      val state = Basic.identity(initialAlignment(batchSize))
      Attention.State(keys, values, state, memory.lengths)
    }
  }

  override def alignment(
      query: Output[T],
      previousState: Attention.State[T, Output[T]]
  ): (Output[T], Attention.State[T, Output[T]]) = {
    Op.nameScope(name) {
      val unmaskedScore = score(query, previousState)
      val maskedScore = Attention.maybeMaskScore(
        unmaskedScore, scoreMaskValue.castTo[T], previousState.sequenceLengths)
      val alignment = probability(maskedScore, previousState)
      (alignment, previousState.copy(state = alignment))
    }
  }

  protected def values(
      memory: Attention.Memory[T]
  ): Output[T] = {
    Op.nameScope(s"$name/Values") {
      Attention.maybeMaskValues(memory.values, memory.lengths)
    }
  }

  protected def keys(
      memory: Attention.Memory[T],
      values: Output[T]
  ): Output[T]

  /** Computes an alignment score for `query`.
    *
    * @param  query Query tensor.
    * @param  state Current attention mechanism state (defaults to the previous alignment tensor). The data type of
    *               this tensor matches that of `values` and its shape is `[batchSize, alignmentSize]`, where
    *               `alignmentSize` is the memory's maximum time.
    * @return Score tensor.
    */
  protected def score(query: Output[T], state: Attention.State[T, Output[T]]): Output[T]

  /** Computes alignment probabilities for `score`.
    *
    * @param  score Alignment score tensor.
    * @param  state Current attention mechanism state (defaults to the previous alignment tensor). The data type of
    *               this tensor matches that of `values` and its shape is `[batchSize, alignmentSize]`, where
    *               `alignmentSize` is the memory's maximum time.
    * @return Alignment probabilities tensor.
    */
  protected def probability(score: Output[T], state: Attention.State[T, Output[T]]): Output[T] = {
    NN.softmax(score, name = "Probability")
  }
}

object Attention {
  /** Represents a memory that can be attended over.
    *
    * @param  values  Memory values that will queried; usually the output of an RNN encoder. This tensor should have
    *                 shape `[batchSize, maxTime, ...]`.
    * @param  lengths Sequence lengths for the batch entries in the memory values. If provided, the memory tensor rows
    *                 are masked with zeros for values past the respective sequence lengths.
    */
  case class Memory[T](
      values: Output[T],
      lengths: Option[Output[Int]] = None)

  case class State[T, S](
      keys: Output[T],
      values: Output[T],
      state: S,
      sequenceLengths: Option[Output[Int]] = None)

  type StateShape[S] = (Shape, Shape, S, Option[Shape])

  /** Potentially masks the provided values tensor based on the provided sequence lengths. */
  @throws[InvalidShapeException]
  private[attention] def maybeMaskValues[T: TF : IsNotQuantized](
      values: Output[T],
      sequenceLengths: Option[Output[Int]]
  ): Output[T] = {
    sequenceLengths match {
      case None => values
      case Some(lengths) =>
        val sequenceMask = Basic.sequenceMask(lengths, Basic.shape(values).slice(1)).castTo[T]
        val rank = if (values.rank != -1) Basic.constant(values.rank) else Basic.rank(values)
        val extraOnes = Basic.ones[Int](Basic.expandDims(rank - 2, 0))
        val mask = sequenceMask.reshape(
          Basic.concatenate(Seq(
            Basic.shape(sequenceMask),
            extraOnes
          ), axis = 0))
        values * mask
    }
  }

  /** Potentially masks the provided score tensor based on the provided sequence lengths. */
  private[attention] def maybeMaskScore[T: TF : IsNotQuantized](
      score: Output[T],
      scoreMaskValue: Output[T],
      sequenceLengths: Option[Output[Int]] = None
  ): Output[T] = {
    sequenceLengths match {
      case None => score
      case Some(lengths) =>
        val scoreMask = Basic.sequenceMask(
          lengths,
          Basic.shape(score).slice(1))
        Math.select(scoreMask, score, scoreMaskValue * Basic.onesLike(score))
    }
  }
}
