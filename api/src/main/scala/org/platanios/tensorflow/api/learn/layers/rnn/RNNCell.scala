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

package org.platanios.tensorflow.api.learn.layers.rnn

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.types.INT32

import scala.collection.mutable

abstract class RNNCell(override protected val name: String) extends Layer[RNNCell.Tuple, RNNCell.Tuple](name) {
  def stateSize: Seq[Int]
  def outputSize: Int

  //region Helper Methods for Subclasses

  @inline protected def concatenate(x: Output, y: Output, axis: Output): Output = {
    ops.Basic.concatenate(Seq(x, y), axis = axis)
  }

  @inline protected def splitEvenly(x: Output, numSplits: Int, axis: Output): Seq[Output] = {
    ops.Basic.splitEvenly(x, numSplits, axis = axis)
  }

  @inline protected def sigmoid(x: Output): Output = ops.Math.sigmoid(x)
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

  override def stateSize: Seq[Int] = Seq(numUnits)
  override def outputSize: Int = numUnits

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

  override def stateSize: Seq[Int] = Seq(numUnits)
  override def outputSize: Int = numUnits

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

/** Basic Long-Short Term Memory (LSTM) cell.
  *
  * The implementation is based on: ["Recurrent Neural Network Regularization", Zaremba et al](http://arxiv.org/abs/1409.2329).
  *
  * We add `forgetBias` (which defaults to 1) to the biases of the forget gate in order to reduce the scale of
  * forgetting in the beginning of training.
  *
  * This cell does not allow for cell clipping, a projection layer, or for peep-hole connections. For advanced models,
  * please use the full [[LSTMCell]].
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
class BasicLSTMCell(
    numUnits: Int,
    forgetBias: Float = 1.0f,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "BasicLSTMCell"
) extends RNNCell(name) {
  override val layerType: String = "BasicLSTMCell"

  override def stateSize: Seq[Int] = Seq(numUnits, numUnits)
  override def outputSize: Int = numUnits

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
    val cPrev = input.state(0)
    val mPrev = input.state(1)
    val lstmMatrix = addBias(matmul(concatenate(input.output, mPrev, axis = 1), kernel), bias)
    // i = input gate, j = new input, f = forget gate, o = output gate
    val lstmMatrixBlocks = splitEvenly(lstmMatrix, 4, axis = one)
    val (i, j, f, o) = (lstmMatrixBlocks(0), lstmMatrixBlocks(1), lstmMatrixBlocks(2), lstmMatrixBlocks(3))
    val forgetBiasTensor = ops.Basic.constant(forgetBias, f.dataType)
    val c = multiply(cPrev, sigmoid(f + forgetBiasTensor)) + multiply(sigmoid(i), activation(j))
    val m = multiply(activation(c), sigmoid(o))
    LayerInstance(input, RNNCell.Tuple(m, Seq(c, m)), Set(kernel, bias))
  }
}

/** Long-Short Term Memory (LSTM) cell.
  *
  * This class uses optional peep-hole connections, optional cell clipping, and an optional projection layer.
  *
  * The default non-peephole implementation is based on:
  * ["Long Short-Term Memory", S. Hochreiter and J. Schmidhuber. Neural Computation, 9(8):1735-1780, 1997.](http://www.bioinf.jku.at/publications/older/2604.pdf).
  *
  * The peephole implementation is based on:
  * ["Long short-term memory recurrent neural network architectures for large scale acoustic modeling", Hasim Sak, Andrew Senior, and Francoise Beaufays. INTERSPEECH, 2014](https://research.google.com/pubs/archive/43905.pdf).
  *
  * @note Input tensors to this layer must be two-dimensional.
  *
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  usePeepholes      Boolean value indicating whether or not to use diagonal/peephole connections.
  * @param  cellClip          If different than `-1`, then the cell state is clipped by this value prior to the cell
  *                           output activation.
  * @param  projectionSize    If different than `-1`, then a projection to that size is added at the output.
  * @param  projectionClip    If different than `-1`, then the projected output is clipped by this value.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
  *                           appending a number to it, if it has been used before for another layer).
  */
class LSTMCell(
    numUnits: Int,
    forgetBias: Float = 1.0f,
    usePeepholes: Boolean = false,
    cellClip: Float = -1,
    projectionSize: Int = -1,
    projectionClip: Float = -1,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "BasicLSTMCell"
) extends RNNCell(name) {
  override val layerType: String = "BasicLSTMCell"

  override def stateSize: Seq[Int] = {
    if (projectionSize != -1)
      Seq(numUnits, projectionSize)
    else
      Seq(numUnits, numUnits)
  }

  override def outputSize: Int = {
    if (projectionSize != -1)
      projectionSize
    else
      numUnits
  }

  override def forward(input: RNNCell.Tuple, mode: Mode): LayerInstance[RNNCell.Tuple, RNNCell.Tuple] = {
    if (input.output.rank != 2)
      throw InvalidArgumentException(s"Input must be rank-2 (provided rank-${input.output.rank}).")
    if (input.output.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input shape (${input.output.shape}) must be known.")
    if (input.state.length != 1)
      throw InvalidArgumentException(s"The state must consist of one tensor.")
    val trainableVariables: mutable.Set[Variable] = mutable.Set[Variable]()
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = variable(
      RNNCell.KERNEL_NAME,
      input.output.dataType,
      Shape(input.output.shape(1) + hiddenDepth, 4 * numUnits),
      kernelInitializer)
    val bias = variable(
      RNNCell.BIAS_NAME,
      input.output.dataType, Shape(4 * numUnits),
      biasInitializer)
    trainableVariables += kernel
    trainableVariables += bias
    val one = ops.Basic.constant(1, INT32)
    // Parameters of gates are concatenated into one multiply for efficiency.
    val cPrev = input.state(0)
    val mPrev = input.state(1)
    val lstmMatrix = addBias(matmul(concatenate(input.output, mPrev, axis = 1), kernel), bias)
    // i = input gate, j = new input, f = forget gate, o = output gate
    val lstmMatrixBlocks = splitEvenly(lstmMatrix, 4, axis = one)
    val (i, j, f, o) = (lstmMatrixBlocks(0), lstmMatrixBlocks(1), lstmMatrixBlocks(2), lstmMatrixBlocks(3))
    // Diagonal connections
    val forgetBiasTensor = ops.Basic.constant(forgetBias, f.dataType)
    var c = {
      if (usePeepholes) {
        val wfDiag = variable("Peepholes/ForgetKernelDiag", input.output.dataType, Shape(numUnits), kernelInitializer)
        val wiDiag = variable("Peepholes/InputKernelDiag", input.output.dataType, Shape(numUnits), kernelInitializer)
        trainableVariables += wfDiag
        trainableVariables += wiDiag
        multiply(cPrev, sigmoid(f + forgetBiasTensor + multiply(wfDiag, cPrev))) +
            multiply(sigmoid(i + multiply(wiDiag, cPrev)), activation(j))
      } else {
        multiply(cPrev, sigmoid(f + forgetBiasTensor)) + multiply(sigmoid(i), activation(j))
      }
    }
    if (cellClip != -1) {
      val cellClipTensor = ops.Basic.constant(cellClip)
      c = c.clipByValue(-cellClipTensor, cellClipTensor)
    }
    var m = {
      if (usePeepholes) {
        val woDiag = variable("Peepholes/OutputKernelDiag", input.output.dataType, Shape(numUnits), kernelInitializer)
        trainableVariables += woDiag
        multiply(activation(c), sigmoid(o + multiply(woDiag, c)))
      } else {
        multiply(activation(c), sigmoid(o))
      }
    }
    if (projectionSize != -1) {
      val projectionKernel = variable(
        s"Projection/${RNNCell.KERNEL_NAME}",
        input.output.dataType,
        Shape(numUnits, projectionSize),
        kernelInitializer)
      trainableVariables += projectionKernel
      m = matmul(m, projectionKernel)
      if (projectionClip != -1) {
        val projectionClipTensor = ops.Basic.constant(projectionClip)
        m = m.clipByValue(-projectionClipTensor, projectionClipTensor)
      }
    }
    LayerInstance(input, RNNCell.Tuple(m, Seq(c, c)), trainableVariables.toSet)
  }
}
