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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.types.{DataType, INT32}

/** RNN cell that wraps another RNN cell and adds support for attention to it.
  *
  * @param  cell                   RNN cell being wrapped.
  * @param  attentions             Attention mechanisms to use.
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
class AttentionWrapperCell[S, SS] private[attention] (
    val cell: RNNCell[Output, Shape, S, SS],
    val attentions: Seq[Attention],
    val attentionLayerWeights: Seq[Output] = null,
    val cellInputFn: (Output, Output) => Output = (input, attention) => Basic.concatenate(Seq(input, attention), -1),
    val outputAttention: Boolean = true,
    val storeAlignmentsHistory: Boolean = false,
    val name: String = "AttentionWrapperCell"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS]
) extends RNNCell[Output, Shape, AttentionWrapperState[S, SS], (SS, Shape, Shape, Seq[Shape], Seq[Shape])] {
  private[this] val attentionLayersSize: Int = {
    if (attentionLayerWeights != null) {
      require(attentionLayerWeights.lengthCompare(attentions.size) == 0,
        s"The number of attention layer weights (${attentionLayerWeights.size}) must match the number of " +
            s"attention mechanisms (${attentions.size}).")
      val sizes = attentionLayerWeights.map(_.shape(-1))
      if (sizes.contains(-1)) -1 else sizes.sum
    } else {
      val sizes = attentions.map(_.values.shape(-1))
      if (sizes.contains(-1)) -1 else sizes.sum
    }
  }

  /** Returns an initial state for this attention cell wrapper.
    *
    * @param  initialCellState Initial state for the wrapped cell.
    * @param  dataType         Optional data type which defaults to the data type of the last tensor in
    *                          `initialCellState`.
    * @return Initial state for this attention cell wrapper.
    */
  def initialState(initialCellState: S, dataType: DataType = null): AttentionWrapperState[S, SS] = {
    if (initialCellState == null) {
      null
    } else {
      Op.createWithNameScope(s"$name/InitialState") {
        val state = evS.outputs(initialCellState).last
        val inferredDataType = if (dataType == null) state.dataType else dataType
        val batchSize: Output = if (state.rank != -1 && state.shape(0) != -1) state.shape(0) else Basic.shape(state)(0)
        val checkedCellState = Op.createWith(controlDependencies = attentions.map(a => Checks.assertEqual(
          a.batchSize, batchSize, message =
              s"When calling `initialState` of `AttentionWrapperCell` '$name': Non-matching batch sizes between the " +
                  "memory (encoder output) and the requested batch size.")).toSet) {
          evS.map(initialCellState, {
            case s: TensorArray => s.identity
            case s: OutputLike => Basic.identity(s, "CheckedInitialCellState")
          })
        }
        AttentionWrapperState(
          cellState = checkedCellState,
          time = Basic.zeros(INT32, Shape.scalar()),
          attention = Basic.fill(inferredDataType, Basic.stack(Seq(batchSize, attentionLayersSize)))(0),
          alignments = attentions.map(_.initialAlignment),
          alignmentsHistory = attentions.map(_ => {
            if (storeAlignmentsHistory) TensorArray.create(0, inferredDataType, dynamicSize = true) else null
          }))
      }
    }
  }

  override def outputShape: Shape = if (outputAttention) Shape(attentionLayersSize) else cell.outputShape

  override def stateShape: (SS, Shape, Shape, Seq[Shape], Seq[Shape]) = {
    (cell.stateShape, Shape.scalar(), Shape(attentionLayersSize),
        attentions.map(a => Output.constantValueAsShape(a.alignmentSize).getOrElse(Shape.unknown())),
        attentions.map(_ => Shape.scalar()))
  }

  /** Performs a step using this attention-wrapped RNN cell.
    *
    *  - Step 1: Mix the `inputs` and the previous step's `attention` output via `cellInputFn`.
    *  - Step 2: Call the wrapped `cell` with the mixed input and its previous state.
    *  - Step 3: Score the cell's output with `attentionMechanism`.
    *  - Step 4: Calculate the alignments by passing the score through the `normalizer`.
    *  - Step 5: Calculate the context vector as the inner product between the alignments and the attention mechanism's
    *            values (memory).
    *  - Step 6: Calculate the attention output by concatenating the cell output and context through the attention layer
    *            (a linear layer with `attentionLayerWeights.shape(-1)` outputs).
    *
    * @param  input Input tuple to the attention wrapper cell.
    * @return Next tuple.
    */
  override def forward(
      input: Tuple[Output, AttentionWrapperState[S, SS]]): Tuple[Output, AttentionWrapperState[S, SS]] = {
    // Step 1: Calculate the true inputs to the cell based on the previous attention value.
    val cellInput = cellInputFn(input.output, input.state.attention)
    val nextTuple = cell.forward(Tuple(cellInput, input.state.cellState))
    val output = nextTuple.output
    val batchSize: Output = if (output.rank != -1 && output.shape(0) != -1) output.shape(0) else Basic.shape(output)(0)
    val checkedOutput = Op.createWith(controlDependencies = attentions.map(a => Checks.assertEqual(
      a.batchSize, batchSize, message =
          s"When calling `initialState` of `AttentionWrapperCell` '$name': Non-matching batch sizes between the " +
              "memory (encoder output) and the requested batch size.")).toSet) {
      Basic.identity(output, "CheckedCellOutput")
    }
    val weights = if (attentionLayerWeights != null) attentionLayerWeights else attentions.map(_ => null)
    val (allAttentions, allAlignments) = (attentions, input.state.alignments, weights).zipped.map {
      case (mechanism, previous, w) =>
        val alignments = mechanism.alignment(checkedOutput, previous)
        // Reshape from [batchSize, memoryTime] to [batchSize, 1, memoryTime]
        val expandedAlignments = alignments.expandDims(1)
        // Context is the inner product of alignments and values along the memory time dimension.
        // The alignments shape is:       [batchSize, 1, memoryTime]
        // The mechanism values shape is: [batchSize, memoryTime, memorySize]
        // The batched matrix multiplication is over `memoryTime` and so the output shape is: [batchSize, 1, memorySize]
        // We then squeeze out the singleton dimension.
        val context = Math.matmul(expandedAlignments, mechanism.values).squeeze(Seq(1))
        val attention = {
          if (w != null)
            Math.matmul(Basic.concatenate(Seq(checkedOutput, context), 1), w)
          else
            context
        }
        (attention, alignments)
    }.unzip
    val histories = {
      if (storeAlignmentsHistory)
        input.state.alignmentsHistory.zip(allAlignments).map(p => p._1.write(input.state.time, p._2))
      else
        input.state.alignmentsHistory
    }
    val one = Basic.constant(1)
    val attention = Basic.concatenate(allAttentions, one)
    val nextState = AttentionWrapperState(nextTuple.state, input.state.time + one, attention, allAlignments, histories)
    if (outputAttention)
      Tuple(attention, nextState)
    else
      Tuple(checkedOutput, nextState)
  }
}

object AttentionWrapperCell {
  def apply[S, SS](
      cell: RNNCell[Output, Shape, S, SS],
      attentions: Seq[Attention],
      attentionLayerWeights: Seq[Output] = null,
      cellInputFn: (Output, Output) => Output = (input, attention) => Basic.concatenate(Seq(input, attention), -1),
      outputAttention: Boolean = true,
      storeAlignmentsHistory: Boolean = false,
      name: String = "AttentionWrapperCell"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS]
  ): AttentionWrapperCell[S, SS] = {
    new AttentionWrapperCell[S, SS](
      cell, attentions, attentionLayerWeights, cellInputFn, outputAttention, storeAlignmentsHistory, name)
  }
}
