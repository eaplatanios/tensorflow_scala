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

package org.platanios.tensorflow.api.ops.rnn

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.core.types.{IsIntOrLong, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, OpSpecification, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.ops.variables.VariableScope

import scala.language.postfixOps

/** Contains functions for constructing ops related to recurrent neural networks (RNNs).
  *
  * @author Emmanouil Antonios Platanios
  */
trait RNN {
  /** A recurrent neural network (RNN) specified by the provided RNN cell.
    *
    * The op performs fully dynamic unrolling of the RNN.
    *
    * @group RNNOps
    * @param  cell               RNN cell to use.
    * @param  input              Input to the RNN loop.
    * @param  initialState       Initial state to use for the RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    *                            Defaults to a zero state.
    * @param  timeMajor          Boolean value indicating whether the `inputs` are provided in time-major format (i.e.,
    *                            have shape `[time, batch, depth]`) or in batch-major format (i.e., have shape
    *                            `[batch, time, depth]`).
    * @param  parallelIterations Number of RNN loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the RNN loop.
    * @param  sequenceLengths    Optional tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @param  name               Name prefix to use for the created ops.
    * @return RNN cell tuple after the dynamic RNN loop is completed. The `output` of that tuple has a time axis
    *         prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration in the loop.
    *         The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException    If the inputs or the provided sequence lengths have invalid or unknown shapes.
    * @throws InvalidArgumentException If neither `initialState` nor `zeroState` is provided.
    */
  @throws[InvalidShapeException]
  @throws[InvalidArgumentException]
  def dynamicRNN[Out: OutputStructure, State: OutputStructure, OutShape, StateShape](
      cell: RNNCell[Out, State, OutShape, StateShape],
      input: Out,
      initialState: Option[State] = None,
      timeMajor: Boolean = false,
      parallelIterations: Int = 32,
      swapMemory: Boolean = false,
      sequenceLengths: Output[Int] = null,
      name: String = "RNN"
  )(implicit
      evZeroOut: Zero.Aux[Out, OutShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Tuple[Out, State] = {
    Op.nameScope(name) {
      // Create a new variable scope in which the caching device is either determined by the parent scope, or is set to
      // place the cached variables using the same device placement as for the rest of the RNN.
      val currentVariableScope = VariableScope.scope(name)(VariableScope.current)
      val cachingDevice = {
        if (currentVariableScope.cachingDevice == null)
          (opSpecification: OpSpecification) => opSpecification.device
        else
          currentVariableScope.cachingDevice
      }
      VariableScope.updatedScope(currentVariableScope, cachingDevice = cachingDevice) {
        // By default, `timeMajor` is false and inputs are shaped batch-major: [batch, time, depth]
        // For internal calculations, we transpose to: [time, batch, depth]
        var processedInput = OutputStructure[Out].outputs(input)
        processedInput = {
          if (!timeMajor) {
            // [B, T, D] => [T, B, D]
            processedInput.map(i => {
              RNN.transposeBatchTime(i)(TF.fromDataType(i.dataType))
            })
          } else {
            processedInput
          }
        }
        var processedSequenceLength = {
          if (sequenceLengths == null) {
            null
          } else {
            if (sequenceLengths.rank != -1 && sequenceLengths.rank != 1)
              throw InvalidShapeException(
                s"'sequenceLength' (rank = ${sequenceLengths.rank}) must be a vector " +
                    "with length equal to the batch size.")
            sequenceLengths.castTo[Int]
          }
        }
        var finalTuple = RNN.dynamicRNNLoop(
          cell, OutputStructure[Out].decodeOutput(input, processedInput)._1,
          initialState, parallelIterations, swapMemory, processedSequenceLength)
        // Outputs of `dynamicRNNLoop` are always shaped [time, batch, depth].
        // If we are performing batch-major calculations, transpose output back to shape [batch, time, depth].
        val finalTupleOutputs = OutputStructure[Out].outputs(finalTuple.output)
        if (!timeMajor) {
          // [T, B, D] => [B, T, D]
          finalTuple = Tuple(
            output = OutputStructure[Out].decodeOutput(
              input,
              finalTupleOutputs.map(o => {
                RNN.transposeBatchTime(o)(TF.fromDataType(o.dataType))
              }))._1,
            state = finalTuple.state)
        }
        finalTuple
      }
    }
  }

  /** A bidirectional recurrent neural network (RNN) specified by the provided RNN cell. The op performs fully dynamic
    * unrolling of the forward and backward RNNs.
    *
    * The op takes the inputs and builds independent forward and backward RNNs. The output sizes of the forward and
    * the backward RNN cells must match. The initial state for both directions can be provided and no intermediate
    * states are ever returned -- the network is fully unrolled for the provided sequence length(s) of the sequence(s)
    * or completely unrolled if sequence length(s) are not provided.
    *
    * @group RNNOps
    * @param  cellFw             RNN cell to use for the forward direction.
    * @param  cellBw             RNN cell to use for the backward direction.
    * @param  input              Input to the RNN loop.
    * @param  initialStateFw     Initial state to use for the forward RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    *                            Defaults to a zero state.
    * @param  initialStateBw     Initial state to use for the backward RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    *                            Defaults to a zero state.
    * @param  timeMajor          Boolean value indicating whether the `inputs` are provided in time-major format (i.e.,
    *                            have shape `[time, batch, depth]`) or in batch-major format (i.e., have shape
    *                            `[batch, time, depth]`).
    * @param  parallelIterations Number of RNN loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the RNN loop.
    * @param  sequenceLengths    Optional tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @param  name               Name prefix to use for the created ops.
    * @return Tuple containing: (i) the forward RNN cell tuple after the forward dynamic RNN loop is completed, and (ii)
    *         the backward RNN cell tuple after the backward dynamic RNN loop is completed. The `output` of these tuples
    *         has a time axis prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration
    *         in the loop. The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException If the inputs or the provided sequence lengths have invalid or unknown shapes.
    */
  @throws[InvalidShapeException]
  def bidirectionalDynamicRNN[Out: OutputStructure, State: OutputStructure, OutShape, StateShape](
      cellFw: RNNCell[Out, State, OutShape, StateShape],
      cellBw: RNNCell[Out, State, OutShape, StateShape],
      input: Out,
      initialStateFw: Option[State] = None,
      initialStateBw: Option[State] = None,
      timeMajor: Boolean = false,
      parallelIterations: Int = 32,
      swapMemory: Boolean = false,
      sequenceLengths: Output[Int] = null,
      name: String = "RNN"
  )(implicit
      evZeroOut: Zero.Aux[Out, OutShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): (Tuple[Out, State], Tuple[Out, State]) = {
    Op.nameScope(name) {
      VariableScope.scope(name) {
        // Forward direction
        val forwardTuple = VariableScope.scope("Forward") {
          dynamicRNN(
            cellFw, input, initialStateFw, timeMajor,
            parallelIterations, swapMemory, sequenceLengths)
        }

        // Backward direction
        val (timeAxis, batchAxis) = if (timeMajor) (0, 1) else (1, 0)

        def reverse(input: Out): Out = {
          var sequence = OutputStructure[Out].outputs(input)
          if (sequenceLengths == null)
            sequence = sequence.map(input => {
              Basic.reverse(input, timeAxis)(TF.fromDataType(input.dataType), TF[Int], IsIntOrLong[Int])
            })
          else
            sequence = sequence.map(input => {
              Basic.reverseSequence(
                input, sequenceLengths, timeAxis, batchAxis
              )(TF.fromDataType(input.dataType), TF[Int], IsIntOrLong[Int])
            })
          OutputStructure[Out].decodeOutput(input, sequence)._1
        }

        val backwardTuple = VariableScope.scope("Backward") {
          val reversedInput = reverse(input)
          dynamicRNN(
            cellBw, reversedInput, initialStateBw, timeMajor,
            parallelIterations, swapMemory, sequenceLengths)
        }

        (forwardTuple, Tuple(reverse(backwardTuple.output), backwardTuple.state))
      }
    }
  }
}

object RNN extends RNN {
  /** Performs the dynamic RNN loop and returns the RNN cell tuple at the end of the loop.
    *
    * @param  cell               RNN cell to use.
    * @param  input              Input to the RNN loop.
    * @param  initialState       Initial state to use for the RNN, which is a sequence of tensors with shapes
    *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
    * @param  parallelIterations Number of loop iterations allowed to run in parallel.
    * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the loop.
    * @param  sequenceLengths    Optional tensor with shape `[batchSize]` containing the sequence lengths for
    *                            each row in the batch.
    * @return RNN cell tuple after the dynamic RNN loop is completed. The `output` of that tuple had a time axis
    *         prepended to the shape of each tensor and corresponds to the RNN outputs at each iteration in the loop.
    *         The `state` represents the RNN state at the end of the loop.
    * @throws InvalidShapeException If the inputs have invalid or unknown shapes.
    */
  @throws[InvalidShapeException]
  private[RNN] def dynamicRNNLoop[Out: OutputStructure, State: OutputStructure, OutShape, StateShape](
      cell: RNNCell[Out, State, OutShape, StateShape],
      input: Out,
      initialState: Option[State],
      parallelIterations: Int,
      swapMemory: Boolean,
      sequenceLengths: Output[Int] = null
  )(implicit
      evZeroOut: Zero.Aux[Out, OutShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Tuple[Out, State] = {
    // Construct an initial output.
    val inputs = OutputStructure[Out].outputs(input)
    val inputShape = Basic.shape(inputs.head)(TF.fromDataType(inputs.head.dataType)).castTo[Int]
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
    val zeroOutput = cell.zeroOutput(batchSize)
    val zeroOutputs = OutputStructure[Out].outputs(zeroOutput)
    val (minSequenceLength, maxSequenceLength) = {
      if (sequenceLengths != null)
        (Math.min(sequenceLengths), Math.max(sequenceLengths))
      else
        (null, timeSteps)
    }
    val time = Op.nameScope("Time")(Basic.zeros[Int](Shape()))
    val outputDataTypes = OutputStructure[Out].outputs(input).map(_.dataType)
    val outputShapes = cell.evOutputToShapeOut.shapeStructure.shapes(cell.outputShape)
    val outputTensorArrays = outputDataTypes.zip(outputShapes).zipWithIndex.map({
      case ((dataType, shape), index) =>
        TensorArray.create(
          size = timeSteps,
          elementShape = Shape(constantBatchSize) ++ Output.constantValueAsShape(shape).get,
          name = s"Output_$index"
        )(TF.fromDataType(dataType))
    })
    val inputTensorArrays = inputs.zipWithIndex.map({
      case (in, index) => TensorArray.create(
        size = timeSteps,
        elementShape = in.shape(1 ::),
        name = s"Input_$index"
      )(TF.fromDataType(in.dataType)).unstack(in)
    })

    type LoopVariables = (Output[Int], Seq[TensorArray[Any]], State)

    /** Takes a time step for the dynamic RNN. */
    def timeStep(loopVariables: LoopVariables): LoopVariables = {
      val time = loopVariables._1
      val state = loopVariables._3
      val inputs = inputTensorArrays.map(_.read(time))
      // Restore some shape information.
      inputs.zip(inputsGotShape).foreach(i => i._1.setShape(i._2(1 ::)))
      val callCell: () => (Seq[Output[Any]], Seq[Output[Any]]) = () => {
        val newTuple = cell(Tuple(
          output = OutputStructure[Out].decodeOutput(input, inputs)._1,
          state = state))
        (OutputStructure[Out].outputs(newTuple.output), OutputStructure[State].outputs(newTuple.state))
      }
      val states = OutputStructure[State].outputs(state)
      val (nextOutputs, nextStates) = {
        if (sequenceLengths != null) {
          RNN.rnnStep(
            time, sequenceLengths, minSequenceLength, maxSequenceLength, zeroOutputs, states, callCell,
            skipConditionals = true)
        } else {
          callCell()
        }
      }
      val nextState = OutputStructure[State].decodeOutput(state, nextStates)._1
      val nextOutputTensorArrays = loopVariables._2.zip(nextOutputs).map({
        case (tensorArray, output) =>
          tensorArray.write(time, output)
      })
      (time + Output.constant[Int](1), nextOutputTensorArrays, nextState)
    }

    implicit val evOutputToShapeState: OutputToShape.Aux[State, StateShape] = cell.evOutputToShapeState

    // Make sure that we run at least 1 step, if necessary, to ensure that the tensor arrays pick up the dynamic shape.
    val loopBound = Math.minimum(timeSteps, Math.maximum(1, maxSequenceLength))
    val (_, finalOutputTensorArrays, finalState) = ControlFlow.whileLoop(
      (loopVariables: LoopVariables) => Math.less(loopVariables._1, loopBound),
      (loopVariables: LoopVariables) => timeStep(loopVariables),
      (time, outputTensorArrays, initialState.getOrElse(cell.zeroState(batchSize))),
      parallelIterations = parallelIterations,
      swapMemory = swapMemory,
      maximumIterations = timeSteps)(OutputToShape[LoopVariables])

    // Unpack the final output if not using output tuples
    val finalOutputs = finalOutputTensorArrays.map(_.stack())
    // Restore some shape information
    finalOutputs
        .zip(cell.evOutputToShapeOut.shapeStructure.shapes(cell.outputShape))
        .foreach(o => o._1.setShape(Shape(constantTimeSteps, constantBatchSize) ++ o._2))
    Tuple(OutputStructure[Out].decodeOutput(input, finalOutputs)._1, finalState)
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
    * @param  sequenceLengths   Tensor with shape `[batchSize]` containing the sequence lengths.
    * @param  minSequenceLength Scalar containing the minimum value of `sequenceLengths`.
    * @param  maxSequenceLength Scalar containing the maximum value of `sequenceLengths`.
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
      step: Output[Int],
      sequenceLengths: Output[Int],
      minSequenceLength: Output[Int],
      maxSequenceLength: Output[Int],
      zeroOutput: Seq[Output[Any]],
      state: Seq[Output[Any]],
      callCell: () => (Seq[Output[Any]], Seq[Output[Any]]),
      skipConditionals: Boolean = false
  ): (Seq[Output[Any]], Seq[Output[Any]]) = {
    // Describes which batch entries have finished.
    val copyCond = Math.greaterEqual(step, sequenceLengths)

    def copyOneThrough(
        output: Output[Any],
        newOutput: Output[Any]
    ): Output[Any] = {
      // If the state contains a scalar value we simply pass it through.
      if (output.rank == 0) {
        newOutput
      } else {
        Op.colocateWith(Set(newOutput.op), ignoreExisting = true) {
          Math.select(copyCond, output, newOutput)(TF.fromDataType(output.dataType))
        }
      }
    }

    def copySomeThrough(
        newOutput: Seq[Output[Any]],
        newState: Seq[Output[Any]]
    ): Seq[Output[Any]] = {
      // Use broadcasting select to determine which values should get the previous state and zero output, and which
      // values should get a computed state and output.
      val copiedNewOutput = zeroOutput.zip(newOutput).map(o => copyOneThrough(o._1, o._2))
      val copiedNewState = state.zip(newState).map(s => copyOneThrough(s._1, s._2))
      copiedNewOutput ++ copiedNewState
    }

    /** Runs the RNN step and passes through either no or some past state. */
    def maybeCopySomeThrough(): Seq[Output[Any]] = {
      val newTuple = callCell()
      ControlFlow.cond(
        Math.less(step, minSequenceLength),
        // If step < minSequenceLength we calculate and return everything
        () => newTuple._1 ++ newTuple._2,
        // Else we copy some of it through
        () => copySomeThrough(newTuple._1, newTuple._2))
    }

    val finalOutputAndState = {
      if (skipConditionals) {
        // Instead of using conditionals, perform the selective copy at all time steps. This is faster when
        // `maxSequenceLength` is equal to the number of unrolls (which is typical for `dynamicRNN`).
        val newTuple = callCell()
        copySomeThrough(newTuple._1, newTuple._2)
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

    (finalOutput, finalState)
  }

  /** Transposes the batch and time dimensions of the input tensor, while retaining as much of the static shape
    * information as possible. If the input tensor has rank less than `2` it returns the original tensor. */
  @throws[InvalidShapeException]
  def transposeBatchTime[T: TF](input: Output[T]): Output[T] = {
    val staticShape = input.shape
    if (staticShape.rank != -1 && staticShape.rank < 2) {
      input
    } else {
      val rank = Basic.rank(input)
      val transposed = Basic.transpose(
        input,
        Basic.concatenate[Int](Seq(
          Seq(1, 0),
          Math.range(2, rank)
        ), axis = 0))
      val staticTransposedShape = {
        if (staticShape.rank > 2)
          Shape(staticShape(1), staticShape(0)) ++ staticShape(2 ::)
        else
          Shape(staticShape(1), staticShape(0))
      }
      transposed.setShape(staticTransposedShape)
      transposed
    }
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
  private[rnn] def bestEffortInputBatchSize(
      inputs: Seq[Output[Any]]
  ): Output[Int] = {
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
      Basic.shape(inputs.head)(TF.fromDataType(inputs.head.dataType))
          .castTo[Int]
          .slice(1)
    }
  }
}
