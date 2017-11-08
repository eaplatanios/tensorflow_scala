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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, ZerosInitializer}
import org.platanios.tensorflow.api.types.INT32

abstract class RNNCell(override protected val name: String) extends Layer[RNNCell.Tuple, RNNCell.Tuple](name) {
  def stateSize: Seq[Shape]
  def outputSize: Seq[Shape]

  //region Helper Methods for Subclasses

  @inline protected def concatenate(x: Output, y: Output, axis: Output): Output = {
    ops.Basic.concatenate(Seq(x, y), axis = axis)
  }

  @inline protected def splitEvenly(x: Output, numSplits: Int, axis: Output): Seq[Output] = {
    ops.Basic.splitEvenly(x, numSplits, axis = axis)
  }

  @inline protected def sigmoid[T](x: T): T = ops.Math.sigmoid(x)
  @inline protected def add(x: Output, y: Output): Output = ops.Math.add(x, y)
  @inline protected def multiply(x: Output, y: Output): Output = ops.Math.multiply(x, y)
  @inline protected def matmul(x: Output, y: Output): Output = ops.Math.matmul(x, y)
  @inline protected def addBias(x: Output, bias: Output): Output = ops.NN.addBias(x, bias)

  //endregion Helper Methods for Subclasses
}

object RNNCell {
  private[layers] val KERNEL_NAME: String = "weights"
  private[layers] val BIAS_NAME  : String = "bias"

  case class Tuple(output: Output, state: Seq[Output])
}

/** Most basic RNN cell.
  *
  * Defined as: `output = newState = activation(W * input + U * state + b)`.
  *
  * @note Input tensors to this layer must be two-dimensional.
  *
  * @param  numUnits          Number of units in the RNN cell.
  * @param  activation        Activation function used by this RNN cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
  *                           appending a number to it, if it has been used before for another layer).
  */
class BasicRNNCell(
    numUnits: Int,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "BasicRNNCell"
) extends RNNCell(name) {
  override val layerType: String = "BasicRNNCell"

  override def stateSize: Seq[Shape] = Seq(Shape(numUnits))
  override def outputSize: Seq[Shape] = Seq(Shape(numUnits))

  override def forward(input: RNNCell.Tuple, mode: Mode): LayerInstance[RNNCell.Tuple, RNNCell.Tuple] = {
    if (input.output.rank != 2)
      throw InvalidArgumentException(s"Input must be rank-2 (provided rank-${input.output.rank}).")
    if (input.output.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input shape (${input.output.shape}) must be known.")
    if (input.state.length != 1)
      throw InvalidArgumentException(s"The state must consist of one tensor.")
    val kernel = variable(
      RNNCell.KERNEL_NAME, input.output.dataType, Shape(input.output.shape(1) + numUnits, numUnits), kernelInitializer)
    val bias = variable(RNNCell.BIAS_NAME, input.output.dataType, Shape(numUnits), biasInitializer)
    val linear = addBias(matmul(concatenate(input.output, input.state.head, axis = 1), kernel), bias)
    val output = activation(linear)
    LayerInstance(input, RNNCell.Tuple(output, Seq(output)), Set(kernel, bias))
  }
}

/** Gated Recurrent Unit (GRU) cell.
  *
  * For details refer to
  * [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078).
  *
  * @note Input tensors to this layer must be two-dimensional.
  *
  * @param  numUnits          Number of units in the GRU cell.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
  *                           appending a number to it, if it has been used before for another layer).
  */
class GRUCell(
    numUnits: Int,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "GRUCell"
) extends RNNCell(name) {
  override val layerType: String = "GRUCell"

  override def stateSize: Seq[Shape] = Seq(Shape(numUnits))
  override def outputSize: Seq[Shape] = Seq(Shape(numUnits))

  override def forward(input: RNNCell.Tuple, mode: Mode): LayerInstance[RNNCell.Tuple, RNNCell.Tuple] = {
    if (input.output.rank != 2)
      throw InvalidArgumentException(s"Input must be rank-2 (provided rank-${input.output.rank}).")
    if (input.output.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input shape (${input.output.shape}) must be known.")
    if (input.state.length != 1)
      throw InvalidArgumentException(s"The state must consist of one tensor.")
    val gateKernel = variable(
      s"Gate/${RNNCell.KERNEL_NAME}",
      input.output.dataType,
      Shape(input.output.shape(1) + numUnits, 2 * numUnits),
      kernelInitializer)
    val gateBias = variable(
      s"Gate/${RNNCell.BIAS_NAME}",
      input.output.dataType, Shape(2 * numUnits),
      biasInitializer)
    val candidateKernel = variable(
      s"Candidate/${RNNCell.KERNEL_NAME}",
      input.output.dataType,
      Shape(input.output.shape(1) + numUnits, numUnits),
      kernelInitializer)
    val candidateBias = variable(
      s"Candidate/${RNNCell.BIAS_NAME}",
      input.output.dataType,
      Shape(numUnits),
      biasInitializer)
    val gateInputs = addBias(matmul(concatenate(input.output, input.state.head, axis = 1), gateKernel), gateBias)
    val value = splitEvenly(sigmoid(gateInputs), 2, axis = 1)
    val (r, u) = (value(0), value(1))
    val rState = multiply(r, input.state.head)
    val c = addBias(matmul(concatenate(input.output, rState, axis = 1), candidateKernel), candidateBias)
    val newH = add(multiply(u, input.state.head), multiply(1 - u, c))
    LayerInstance(input, RNNCell.Tuple(newH, Seq(newH)), Set(gateKernel, gateBias, candidateKernel, candidateBias))
  }
}

/** Long-Short Term Memory (LSTM) cell.
  *
  * The implementation is based on: [http://arxiv.org/abs/1409.2329](http://arxiv.org/abs/1409.2329).
  *
  * We add `forgetBias` (which defaults to 1) to the biases of the forget gate in order to reduce the scale of
  * forgetting in the beginning of training.
  *
  * This cell does not allow for cell clipping, a projection layer, or for peep-hole connections.
  *
  * @note Input tensors to this layer must be two-dimensional.
  *
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
  *                           appending a number to it, if it has been used before for another layer).
  */
class LSTMCell(
    numUnits: Int,
    forgetBias: Float = 1.0f,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "GRUCell"
) extends RNNCell(name) {
  override val layerType: String = "GRUCell"

  override def stateSize: Seq[Shape] = Seq(Shape(numUnits), Shape(numUnits))
  override def outputSize: Seq[Shape] = Seq(Shape(numUnits))

  override def forward(input: RNNCell.Tuple, mode: Mode): LayerInstance[RNNCell.Tuple, RNNCell.Tuple] = {
    if (input.output.rank != 2)
      throw InvalidArgumentException(s"Input must be rank-2 (provided rank-${input.output.rank}).")
    if (input.output.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input shape (${input.output.shape}) must be known.")
    if (input.state.length != 1)
      throw InvalidArgumentException(s"The state must consist of one tensor.")
    val kernel = variable(
      RNNCell.KERNEL_NAME,
      input.output.dataType,
      Shape(input.output.shape(1) + numUnits, 4 * numUnits),
      kernelInitializer)
    val bias = variable(
      RNNCell.BIAS_NAME,
      input.output.dataType, Shape(4 * numUnits),
      biasInitializer)
    val one = ops.Basic.constant(1, INT32)
    // Parameters of gates are concatenated into one multiply for efficiency.
    val c = input.state(0)
    val h = input.state(1)
    val gateInputs = addBias(matmul(concatenate(input.output, h, axis = 1), kernel), bias)
    // i = input gate, j = new input, f = forget gate, o = output gate
    val gateInputsBlocks = splitEvenly(gateInputs, 4, axis = one)
    val (i, j, f, o) = (gateInputsBlocks(0), gateInputsBlocks(1), gateInputsBlocks(2), gateInputsBlocks(3))
    val forgetBiasTensor = ops.Basic.constant(forgetBias, f.dataType)
    val newC = multiply(c, sigmoid(f + forgetBiasTensor)) + multiply(sigmoid(i), activation(j))
    val newH = multiply(activation(newC), sigmoid(o))
    LayerInstance(input, RNNCell.Tuple(newH, Seq(newC, newH)), Set(kernel, bias))
  }
}
