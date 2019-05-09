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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.tensors.Tensor

import scala.language.postfixOps

/** RNN cell that wraps another RNN cell and adds support for attention to it.
  *
  * @param  cell                   RNN cell being wrapped.
  * @param  attentions             Map from memories to attention mechanisms to use for those memories.
  * @param  attentionLayerWeights  Attention layer weights to use for projecting the computed attention.
  * @param  cellInputFn            Function that takes the original cell input tensor and the attention tensor as inputs
  *                                and returns the mixed cell input to use. Defaults to concatenating the two tensors
  *                                across their last axis.
  * @param  outputAttention        If `true` (the default), the output of this cell at each step is the attention value.
  *                                This is the behavior of Luong-style attention mechanisms. If `false`, the output at
  *                                each step is the output of `cell`. This is the behavior of Bhadanau-style attention
  *                                mechanisms. In both cases, the `attention` tensor is propagated to the next time step
  *                                via the state and is used there. This flag only controls whether the attention
  *                                mechanism is propagated up to the next cell in an RNN stack or to the top RNN output.
  * @param  storeAlignmentsHistory If `true`, the alignments history from all steps is stored in the final output state
  *                                (currently stored as a time major `TensorArray` on which you must call `stack()`).
  *                                Defaults to `false`.
  * @param  name                   Name prefix used for all new ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class AttentionWrapperCell[T: TF : IsNotQuantized, CellState: OutputStructure, AttentionState, CellStateShape, AttentionStateShape] private[attention] (
    val cell: RNNCell[Output[T], CellState, Shape, CellStateShape],
    val attentions: Seq[(Attention.Memory[T], Attention[T, AttentionState, AttentionStateShape])],
    val attentionLayerWeights: Seq[Output[T]] = null,
    val cellInputFn: (Output[T], Output[T]) => Output[T] = {
      (input: Output[T], attention: Output[T]) =>
        Basic.concatenate(Seq(input, attention), -1)(TF.fromDataType(input.dataType))
    },
    val outputAttention: Boolean = true,
    val storeAlignmentsHistory: Boolean = false,
    val name: String = "AttentionWrapperCell"
)(implicit
    evOutputToShapeCellState: OutputToShape.Aux[CellState, CellStateShape],
    evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
) extends RNNCell[Output[T], AttentionWrapperState[T, CellState, AttentionState], Shape, (CellStateShape, Shape, Shape, Seq[Shape], Seq[Shape], Seq[Attention.StateShape[AttentionStateShape]])] {
  private val attentionLayersSize: Int = {
    if (attentionLayerWeights != null) {
      require(attentionLayerWeights.lengthCompare(attentions.size) == 0,
        s"The number of attention layer weights (${attentionLayerWeights.size}) must match the number of " +
            s"attention mechanisms (${attentions.size}).")
      val sizes = attentionLayerWeights.map(_.shape(-1))
      if (sizes.contains(-1)) -1 else sizes.sum
    } else {
      val sizes = attentions.map(_._1.values.shape(-1))
      if (sizes.contains(-1)) -1 else sizes.sum
    }
  }

  /** Returns an initial state for this attention cell wrapper.
    *
    * @param  initialCellState Initial state for the wrapped cell.
    * @return Initial state for this attention cell wrapper.
    */
  def initialState(
      initialCellState: CellState
  ): AttentionWrapperState[T, CellState, AttentionState] = {
    if (initialCellState == null) {
      null
    } else {
      Op.nameScope(s"$name/InitialState") {
        val state = OutputStructure[CellState].outputs(initialCellState).last.asInstanceOf[Output[T]]
        val batchSize = {
          if (state.rank != -1 && state.shape(0) != -1)
            Basic.constant(state.shape(0))
          else
            Basic.shape(state).castTo[Int].slice(0)
        }
        val initialAlignments = attentions.map(_._2.initialAlignment(batchSize))
        AttentionWrapperState(
          cellState = initialCellState,
          time = Basic.zeros[Int](batchSize.expandDims(0)),
          attention = Basic.fill[T, Int](
            Basic.stack[Int](Seq(batchSize, attentionLayersSize))
          )(Tensor.zeros[T](Shape())),
          alignments = initialAlignments,
          alignmentsHistory = {
            if (storeAlignmentsHistory)
              initialAlignments.map(a =>
                TensorArray.create(
                  size = 0,
                  dynamicSize = true,
                  elementShape = a.shape
                )(TF.fromDataType(state.dataType)))
            else
              Seq.empty
          },
          attentionState = attentions.map(a => a._2.initialState(batchSize, a._1)))
      }
    }
  }

  override def outputShape: Shape = {
    if (outputAttention)
      Shape(attentionLayersSize)
    else
      cell.outputShape
  }

  override def stateShape: (CellStateShape, Shape, Shape, Seq[Shape], Seq[Shape], Seq[Attention.StateShape[AttentionStateShape]]) = {
    (cell.stateShape, Shape(), Shape(attentionLayersSize),
        attentions.map(a => {
          Output.constantValueAsShape(a._2.memorySize.expandDims(0))
              .getOrElse(Shape.unknown())
        }),
        attentions.map(a => {
          if (storeAlignmentsHistory) {
            Output.constantValueAsShape(a._2.memorySize.expandDims(0))
                .getOrElse(Shape.unknown())
          } else {
            Shape.scalar()
          }
        }),
        attentions.map(a => {
          val memoryShape = a._1.values.shape(1 ::)
          val keysShape = a._2.keysShape(a._1.values.shape)(1 ::)
          val stateShape = a._2.stateShape
          val memoryLengthsShape = a._1.lengths.map(_ => Shape())
          (keysShape, memoryShape, stateShape, memoryLengthsShape)
        }))
  }

  /** Performs a step using this attention-wrapped RNN cell.
    *
    *  - Step 1: Mix the `inputs` and the previous step's `attention` output via `cellInputFn`.
    *  - Step 2: Call the wrapped `cell` with the mixed input and its previous state.
    *  - Step 3: Score the cell's output with `attentionMechanism`.
    *  - Step 4: Calculate the alignments by passing the score through the `normalizer`.
    *  - Step 5: Calculate the context vector as the inner product between the alignments and the attention mechanism's
    *    values (memory).
    *  - Step 6: Calculate the attention output by concatenating the cell output and context through the attention layer
    * (a linear layer with `attentionLayerWeights.shape(-1)` outputs).
    *
    * @param  input Input tuple to the attention wrapper cell.
    * @return Next tuple.
    */
  override def forward(
      input: Tuple[Output[T], AttentionWrapperState[T, CellState, AttentionState]]
  ): Tuple[Output[T], AttentionWrapperState[T, CellState, AttentionState]] = {
    // Step 1: Calculate the true inputs to the cell based on the previous attention value.
    val cellInput = cellInputFn(input.output, input.state.attention)
    val nextTuple = cell.forward(Tuple(cellInput, input.state.cellState))
    val output = nextTuple.output
    val weights = if (attentionLayerWeights != null) attentionLayerWeights else attentions.map(_ => null)
    val (allAttentions, allAlignments, allStates) = (attentions, input.state.attentionState, weights).zipped.map {
      case (attentionPair, previousState, w) =>
        val (alignments, state) = attentionPair._2.alignment(output, previousState)
        // Reshape from [batchSize, memoryTime] to [batchSize, 1, memoryTime]
        val expandedAlignments = alignments.expandDims(1)
        // Context is the inner product of alignments and values along the memory time dimension.
        // The alignments shape is:       [batchSize, 1, memoryTime]
        // The mechanism values shape is: [batchSize, memoryTime, memorySize]
        // The batched matrix multiplication is over `memoryTime` and so the output shape is: [batchSize, 1, memorySize]
        // We then squeeze out the singleton dimension.
        val context = Math.matmul(expandedAlignments, previousState.values).squeeze(Seq(1))
        val attention = {
          if (w != null)
            Math.matmul(Basic.concatenate(Seq(output, context), 1), w)
          else
            context
        }
        (attention, alignments, state)
    }.unzip3
    val histories = {
      if (storeAlignmentsHistory)
        input.state.alignmentsHistory.zip(allAlignments).map(p => p._1.write(input.state.time, p._2))
      else
        input.state.alignmentsHistory
    }
    val one = Basic.ones[Int](Shape())
    val attention = Basic.concatenate(allAttentions, one)
    val nextState = AttentionWrapperState(
      nextTuple.state, input.state.time + one, attention, allAlignments, histories, allStates)
    if (outputAttention)
      Tuple(attention, nextState)
    else
      Tuple(output, nextState)
  }
}

object AttentionWrapperCell {
  def apply[T: TF : IsNotQuantized, CellState: OutputStructure, AttentionState, CellStateShape, AttentionStateShape](
      cell: RNNCell[Output[T], CellState, Shape, CellStateShape],
      attentions: Seq[(Attention.Memory[T], Attention[T, AttentionState, AttentionStateShape])],
      attentionLayerWeights: Seq[Output[T]] = null,
      cellInputFn: (Output[T], Output[T]) => Output[T] = {
        (input: Output[T], attention: Output[T]) =>
          Basic.concatenate(Seq(input, attention), -1)(TF.fromDataType(input.dataType))
      },
      outputAttention: Boolean = true,
      storeAlignmentsHistory: Boolean = false,
      name: String = "AttentionWrapperCell"
  )(implicit
      evOutputToShapeCellState: OutputToShape.Aux[CellState, CellStateShape],
      evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
  ): AttentionWrapperCell[T, CellState, AttentionState, CellStateShape, AttentionStateShape] = {
    new AttentionWrapperCell[T, CellState, AttentionState, CellStateShape, AttentionStateShape](
      cell, attentions, attentionLayerWeights, cellInputFn,
      outputAttention, storeAlignmentsHistory, name)
  }
}
