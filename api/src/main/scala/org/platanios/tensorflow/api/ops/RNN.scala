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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.VariableScope
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32}

import scala.language.postfixOps

/** Contains functions for constructing ops related to recurrent neural networks (RNNs).
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait RNN {
  /** $OpDocRNNDynamicRNN
    *
    * @group RNNOps
    * @param  cell               RNN cell to use.
    * @param  cellOutputSize     Sequence containing the sizes of each output of `cell`.
    * @param  inputs             Inputs to the RNN loop.
    * @param  initialState       Initial state to use for the RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    * @param  timeMajor          Boolean value indicating whether the `inputs` are provided in time-major format (i.e.,
    *                            have shape `[time, batch, depth]`) or in batch-major format (i.e., have shape
    *                            `[batch, time, depth]`).
    * @param  parallelIterations Number of loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the loop.
    * @param  sequenceLengths    Optional `INT32` tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @param  name               Name prefix to use for the created ops.
    * @return RNN cell tuple after the dynamic RNN loop is completed. The `output` of that tuple has a time axis
    *         prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration in the loop.
    *         The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException If the inputs or the provided sequence lengths have invalid or unknown shapes.
    */
  @throws[InvalidShapeException]
  def dynamicRNN(
      cell: RNNCell.Tuple => RNNCell.Tuple, cellOutputSize: Seq[Int], inputs: Seq[Output],
      initialState: Seq[Output], timeMajor: Boolean = false, parallelIterations: Int = 32,
      swapMemory: Boolean = false, sequenceLengths: Output = null, name: String = "RNN"): RNNCell.Tuple = {
    Op.createWithNameScope(name) {
      // By default, `timeMajor` is false and inputs are shaped batch-major: [batch, time, depth]
      // For internal calculations, we transpose to: [time, batch, depth]
      val processedInputs = {
        if (!timeMajor) {
          // [B, T, D] => [T, B, D]
          inputs.map(RNN.transposeBatchTime)
        } else {
          inputs
        }
      }
      var processedSequenceLength = {
        if (sequenceLengths == null) {
          null
        } else {
          if (sequenceLengths.rank != -1 && sequenceLengths.rank != 1)
            throw InvalidShapeException(
              s"'sequenceLength' (rank = ${sequenceLengths.rank}) must be a vector with length equal to the batch size.")
          Math.cast(sequenceLengths, INT32, "SequenceLengthCast")
        }
      }
      // Create a new variable scope in which the caching device is either determined by the parent scope, or is set to
      // place the cached variables using the same device placement as for the rest of the RNN.
      val currentVariableScope = VariableScope.createWithVariableScope(name)(Op.currentVariableScope)
      val cachingDevice = {
        if (currentVariableScope.cachingDevice == null)
          (opSpecification: OpSpecification) => opSpecification.device
        else
          currentVariableScope.cachingDevice
      }
      VariableScope.createWithUpdatedVariableScope(currentVariableScope, cachingDevice = cachingDevice) {
        val batchSize = RNN.bestEffortInputBatchSize(processedInputs)
        // Perform some shape validation
        processedSequenceLength = Op.createWith(
          controlDependencies = Set(RNN.assertHasShape(processedSequenceLength, batchSize.expandDims(0)))) {
          Basic.identity(processedSequenceLength, "SequenceLengthShapeValidation")
        }
        var finalTuple = RNN.dynamicRNNLoop(
          cell, cellOutputSize, processedInputs, initialState, parallelIterations, swapMemory, processedSequenceLength)
        // Outputs of `dynamicRNNLoop` are always shaped [time, batch, depth].
        // If we are performing batch-major calculations, transpose output back to shape [batch, time, depth].
        if (!timeMajor) {
          // [T, B, D] => [B, T, D]
          finalTuple = finalTuple.copy(output = finalTuple.output.map(RNN.transposeBatchTime))
        }
        finalTuple
      }
    }
  }

  /** $OpDocRNNBidirectionalDynamicRNN
    *
    * @group RNNOps
    * @param  cellFw             RNN cell to use for the forward direction.
    * @param  cellFwOutputSize   Sequence containing the sizes of each output of `cellFw`.
    * @param  cellBw             RNN cell to use for the backward direction.
    * @param  cellBwOutputSize   Sequence containing the sizes of each output of `cellBw`.
    * @param  inputs             Inputs to the RNN loop.
    * @param  initialStateFw     Initial state to use for the forward RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    * @param  initialStateBw     Initial state to use for the backward RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    * @param  timeMajor          Boolean value indicating whether the `inputs` are provided in time-major format (i.e.,
    *                            have shape `[time, batch, depth]`) or in batch-major format (i.e., have shape
    *                            `[batch, time, depth]`).
    * @param  parallelIterations Number of loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the loop.
    * @param  sequenceLengths    Optional `INT32` tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @param  name               Name prefix to use for the created ops.
    * @return Tuple containing: (i) the forward RNN cell tuple after the forward dynamic RNN loop is completed, and (ii)
    *         the backward RNN cell tuple after the backward dynamic RNN loop is completed. The `output` of these tuples
    *         has a time axis prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration
    *         in the loop. The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException If the inputs or the provided sequence lengths have invalid or unknown shapes.
    */
  @throws[InvalidShapeException]
  def bidirectionalDynamicRNN(
      cellFw: RNNCell.Tuple => RNNCell.Tuple, cellFwOutputSize: Seq[Int],
      cellBw: RNNCell.Tuple => RNNCell.Tuple, cellBwOutputSize: Seq[Int],
      inputs: Seq[Output], initialStateFw: Seq[Output], initialStateBw: Seq[Output],
      timeMajor: Boolean = false, parallelIterations: Int = 32, swapMemory: Boolean = false,
      sequenceLengths: Output = null, name: String = "RNN"): (RNNCell.Tuple, RNNCell.Tuple) = {
    Op.createWithNameScope(name) {
      VariableScope.createWithVariableScope(name) {
        // Forward direction
        val forwardTuple = VariableScope.createWithVariableScope("Forward") {
          dynamicRNN(
            cellFw, cellFwOutputSize, inputs, initialStateFw, timeMajor, parallelIterations, swapMemory,
            sequenceLengths)
        }

        // Backward direction
        val (timeAxis, batchAxis) = if (timeMajor) (0, 1) else (1, 0)

        def reverse(inputs: Seq[Output]): Seq[Output] = {
          if (sequenceLengths == null)
            inputs.map(input => Basic.reverse(input, Tensor(timeAxis)))
          else
            inputs.map(input => Basic.reverseSequence(input, sequenceLengths, timeAxis, batchAxis))
        }

        val backwardTuple = VariableScope.createWithVariableScope("Backward") {
          val reversedInputs = reverse(inputs)
          dynamicRNN(
            cellBw, cellBwOutputSize, reversedInputs, initialStateBw, timeMajor, parallelIterations, swapMemory,
            sequenceLengths)
        }

        (forwardTuple, backwardTuple.copy(output = reverse(backwardTuple.output)))
      }
    }
  }
}

object RNN extends RNN {
  /** Performs the dynamic RNN loop and returns the RNN cell tuple at the end of the loop.
    *
    * @param  cell               RNN cell to use.
    * @param  cellOutputSize     Sequence containing the sizes of each output of `cell`.
    * @param  inputs             Inputs to the RNN loop.
    * @param  initialState       Initial state to use for the RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    * @param  parallelIterations Number of loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the loop.
    * @param  sequenceLengths    Optional `INT32` tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @return RNN cell tuple after the dynamic RNN loop is completed. The `output` of that tuple had a time axis
    *         prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration in the loop.
    *         The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException If the inputs have invalid or unknown shapes.
    */
  @throws[InvalidShapeException]
  private[RNN] def dynamicRNNLoop(
      cell: RNNCell.Tuple => RNNCell.Tuple, cellOutputSize: Seq[Int], inputs: Seq[Output],
      initialState: Seq[Output], parallelIterations: Int, swapMemory: Boolean,
      sequenceLengths: Output = null): RNNCell.Tuple = {
    // Construct an initial output.
    val inputShape = Basic.shape(inputs.head)
    val timeSteps = inputShape(0)
    val batchSize = bestEffortInputBatchSize(inputs)
    val inputsGotShape = inputs.map(_.shape.withRankAtLeast(3))
    val constantTimeSteps = inputsGotShape(0)(0)
    val constantBatchSize = inputsGotShape(0)(1)
    inputsGotShape.foreach(shape => {
      if (!shape(2 ::).isFullyDefined)
        throw InvalidShapeException("The inputs shape must be known (through shape inference), but it is not.")
      if (constantTimeSteps != shape(0))
        throw InvalidShapeException("The time steps dimension does not have the same size for all inputs.")
      if (constantBatchSize != shape(1))
        throw InvalidShapeException("The batch size is not the same for all inputs.")
    })
    val inferredDataType = inferStateDataType(null, initialState)
    val zeroOutput = cellOutputSize.map(size => {
      Basic.fill(inferredDataType, Basic.stack(Seq(batchSize, size)))(0)
    })
    val (minSequenceLength, maxSequenceLength) = {
      if (sequenceLengths != null)
        (Math.min(sequenceLengths), Math.max(sequenceLengths))
      else
        (null, null)
    }
    val time = Basic.constant(0, INT32, name = "Time")
    val baseName = Op.createWithNameScope("DynamicRNN")(Op.currentNameScope)
    val outputTensorArrays = cellOutputSize.indices.map(index => {
      TensorArray.create(timeSteps, inferredDataType, name = s"$baseName/Output_$index")
    })
    val inputTensorArrays = inputs.zipWithIndex.map({
      case (input, index) =>
        TensorArray.create(timeSteps, input.dataType, name = s"$baseName/Input_$index").unstack(input)
    })

    type LoopVariables = (Output, Seq[TensorArray], Seq[Output])

    /** Takes a time step for the dynamic RNN. */
    def timeStep(loopVariables: LoopVariables): LoopVariables = {
      val time = loopVariables._1
      val state = loopVariables._3
      val inputs = inputTensorArrays.map(_.read(time))
      // Restore some shape information.
      inputs.zip(inputsGotShape).foreach({
        case (input, shape) => input.setShape(shape(1 ::))
      })
      val callCell: () => RNNCell.Tuple = () => cell(RNNCell.Tuple(inputs, state))
      val nextTuple = {
        if (sequenceLengths != null) {
          rnnStep(
            time, sequenceLengths, minSequenceLength, maxSequenceLength, zeroOutput, state, callCell,
            skipConditionals = true)
        } else {
          callCell()
        }
      }
      val nextOutputTensorArrays = loopVariables._2.zip(nextTuple.output).map({
        case (tensorArray, output) => tensorArray.write(time, output)
      })
      (time + 1, nextOutputTensorArrays, nextTuple.state)
    }

    val (_, finalOutputTensorArrays, finalState) = ControlFlow.whileLoop(
      (loopVariables: LoopVariables) => Math.less(loopVariables._1, timeSteps),
      (loopVariables: LoopVariables) => timeStep(loopVariables),
      (time, outputTensorArrays, initialState),
      parallelIterations = parallelIterations,
      swapMemory = swapMemory)

    // Unpack the final output if not using output tuples
    val finalOutputs = finalOutputTensorArrays.map(_.stack())
    // Restore some shape information
    cellOutputSize.map(size => {
      Basic.fill(inferredDataType, Basic.stack(Seq(batchSize, size)))(0)
    })
    finalOutputs.zip(cellOutputSize).foreach({
      case (output, size) => output.setShape(Shape(constantTimeSteps, constantBatchSize, size))
    })
    RNNCell.Tuple(finalOutputs, finalState)
  }

  /** Calculates one step of a dynamic RNN mini-batch.
    *
    * Returns an RNN cell tuple conditioned on the `sequenceLengths`. When `skipConditionals = false`, the pseudocode
    * looks something like:
    * {{{
    *   if (step >= maxSequenceLength) {
    *     RNNCell.Tuple(zeroOutput, state)
    *   } else if (step < minSequenceLength) {
    *     callCell()
    *   } else {
    *     // Selectively output zeros or `output`, and the state or the new state, depending on if we have finished
    *     // calculating each row, by using `sequenceLengths`.
    *     ...
    *   }
    * }}}
    *
    * @param  step              Scalar tensor containing the current step in the RNN loop.
    * @param  sequenceLengths   `INT32` tensor with shape `[batchSize]` containing the sequence lengths.
    * @param  minSequenceLength `INT32` scalar containing the minimum value of `sequenceLengths`.
    * @param  maxSequenceLength `INT32` scalar containing the maximum value of `sequenceLengths`.
    * @param  zeroOutput        Tensor with shape `[outputSize]` containing the "zero output" value.
    * @param  state             Sequence of state tensors with shapes `[batchSize, stateSize(i)]`, where `i` is an index
    *                           over that sequence.
    * @param  callCell          Function returning the next RNN cell tuple.
    * @param  skipConditionals  Boolean value indicating whether to skip using the conditional calculations. This is
    *                           useful for `dynamicRNN`, where the input tensor matches `maxSequenceLength`, and using
    *                           conditionals just slows everything down.
    * @return New RNN cell tuple containing an output tensor with shape `[batchSize, outputSize]` and a sequence of
    *         state tensors with shapes `[batchSize, stateSize(i)]`, where `i` is an index over that sequence.
    */
  private[RNN] def rnnStep(
      step: Output, sequenceLengths: Output, minSequenceLength: Output, maxSequenceLength: Output,
      zeroOutput: Seq[Output], state: Seq[Output], callCell: () => RNNCell.Tuple,
      skipConditionals: Boolean = false): RNNCell.Tuple = {
    def copyOneThrough(output: Output, newOutput: Output): Output = {
      // If the state contains a scalar value we simply pass it through.
      if (output.rank == 0) {
        newOutput
      } else {
        val copyCond = Math.greaterEqual(step, sequenceLengths)
        Op.colocateWith(Set(newOutput.op))(Math.select(copyCond, output, newOutput))
      }
    }

    def copySomeThrough(newOutput: Seq[Output], newState: Seq[Output]): Seq[Output] = {
      // Use broadcasting select to determine which values should get the previous state and zero output, and which
      // values should get a computed state and output.
      val copiedNewOutput = zeroOutput.zip(newOutput).map(o => copyOneThrough(o._1, o._2))
      val copiedNewState = state.zip(newState).map(s => copyOneThrough(s._1, s._2))
      copiedNewOutput ++ copiedNewState
    }

    /** Runs the RNN step and passes through either no or some past state. */
    def maybeCopySomeThrough(): Seq[Output] = {
      val newTuple = callCell()
      ControlFlow.cond(
        Math.less(step, minSequenceLength),
        // If step < minSequenceLength we calculate and return everything
        () => newTuple.output ++ newTuple.state,
        // Else we copy some of it through
        () => copySomeThrough(newTuple.output, newTuple.state))
    }

    val finalOutputAndState = {
      if (skipConditionals) {
        // Instead of using conditionals, perform the selective copy at all time steps. This is faster when
        // `maxSequenceLength` is equal to the number of unrolls (which is typical for `dynamicRNN`).
        val newTuple = callCell()
        copySomeThrough(newTuple.output, newTuple.state)
      } else {
        ControlFlow.cond(
          Math.greaterEqual(step, maxSequenceLength),
          // If step >= maxSequenceLength we copy the whole state through and we output zeros
          () => zeroOutput ++ state,
          // Else we copy some or all of it through
          () => maybeCopySomeThrough())
      }
    }

    val finalOutput = finalOutputAndState.take(zeroOutput.size)
    val finalState = finalOutputAndState.drop(zeroOutput.size)
    finalOutput.zip(zeroOutput).foreach(o => o._1.setShape(o._2.shape))
    finalState.zip(state).foreach(s => s._1.setShape(s._2.shape))

    RNNCell.Tuple(finalOutput, finalState)
  }

  /** Transposes the batch and time dimensions of the input tensor, while retaining as much of the static shape
    * information as possible. */
  @throws[InvalidShapeException]
  private[RNN] def transposeBatchTime(input: Output): Output = {
    val staticShape = input.shape
    if (staticShape.rank != -1 && staticShape.rank < 2)
      throw InvalidShapeException(s"Expected tensor '$input' to have rank at least 2, but saw shape: $staticShape.")
    val rank = Basic.rank(input)
    val transposed = Basic.transpose(input, Basic.concatenate(Seq(Tensor(1, 0), Math.range(2, rank)), axis = 0))
    transposed.setShape(Shape(staticShape(1), staticShape(0)) ++ staticShape(2 ::))
    transposed
  }

  /** Returns the static input batch size if available, with fallback to the dynamic one.
    *
    * @param  inputs Sequence containing time-major input tensors with shape `[maxTime, batchSize, ...]`. All inputs
    *                must have compatible batch sizes.
    * @return Tensor containing the inputs batch size.
    * @throws InvalidArgumentException If the input tensor batch sizes do not match or if any one of them has rank less
    *                                  than 2.
    */
  @throws[InvalidArgumentException]
  private[RNN] def bestEffortInputBatchSize(inputs: Seq[Output]): Output = {
    var batchSize = -1
    inputs.filter(_.rank != -1).foreach(input => {
      if (input.rank < 2)
        throw InvalidArgumentException(s"The input tensor (rank = ${input.rank}) must have rank at least 2.")
      if (input.shape(1) != -1 && batchSize == -1)
        batchSize = input.shape(1)
      if (input.shape(1) != -1 && input.shape(1) != batchSize)
        throw InvalidArgumentException(s"The input tensor batch sizes do not match.")
    })
    if (batchSize != -1) {
      Basic.constant(batchSize)
    } else {
      // Fallback to the dynamic batch size of the first input.
      Basic.shape(inputs.head)(1)
    }
  }

  /** Infers the data type of an RNN's state.
    *
    * @param  dataType If not `null`, then this data type is returned.
    * @param  state    RNN state for which the data type will be inferred.
    * @return
    * @throws InvalidArgumentException If `dataType` is `null` and `state` is empty.
    * @throws InvalidDataTypeException If the tensors in `state` do not all have the same data type.
    */
  @throws[InvalidArgumentException]
  @throws[InvalidDataTypeException]
  private[RNN] def inferStateDataType(dataType: DataType, state: Seq[Output]): DataType = {
    if (dataType != null) {
      dataType
    } else {
      if (state.isEmpty)
        throw InvalidArgumentException("Unable to infer data type from empty state.")
      val inferredDataTypes = state.map(_.dataType)
      if (inferredDataTypes.exists(_ != inferredDataTypes.head))
        throw InvalidDataTypeException("All state tensors must have the same data type.")
      inferredDataTypes.head
    }
  }

  /** Creates an assert op that checks whether the shape of `input` matches `shape`. */
  private[RNN] def assertHasShape(input: Output, shape: Output): Op = {
    val inputShape = Basic.shape(input)
    Checks.assert(
      Math.all(Math.equal(inputShape, shape)),
      Seq(s"Expected shape for tensor ${input.name} is ", shape, " but saw shape ", inputShape))
  }

  /** @define OpDocRNNDynamicRNN
    *   The `dynamicRNN` op creates a recurrent neural network (RNN) specified by the provided RNN cell. The op performs
    *   fully dynamic unrolling of the RNN.
    *
    * @define OpDocRNNBidirectionalDynamicRNN
    *   The `bidirectionalDynamicRNN` op creates a bidirectional recurrent neural network (RNN) specified by the
    *   provided RNN cell. The op performs fully dynamic unrolling of the forward and backward RNNs.
    *
    *   The op takes the inputs and builds independent forward and backward RNNs. The output sizes of the forward and
    *   the backward RNN cells must match. The initial state for both directions can be provided and no intermediate
    *   states are ever returned -- the network is fully unrolled for the provided sequence length(s) of the sequence(s)
    *   or completely unrolled if sequence length(s) are not provided.
    */
  private[ops] trait Documentation
}
