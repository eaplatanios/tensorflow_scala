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

abstract class RNNCell(override protected val name: String) extends Layer[(Output, Output), (Output, Output)](name) {
  def stateSize: Int
  def outputSize: Int
}

object RNNCell {
  private[layers] val KERNEL_NAME: String = "weights"
  private[layers] val BIAS_NAME  : String = "bias"
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

  override def stateSize: Int = numUnits
  override def outputSize: Int = numUnits

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), (Output, Output)] = {
    if (input._1.rank != 2)
      throw InvalidArgumentException(s"Input to 'BasicRNNCell' must be rank-2 (provided rank-${input._1.rank}).")
    if (input._1.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input (shape=${input._1.shape}) to 'BasicRNNCell' must be known.")
    val kernel = variable(
      RNNCell.KERNEL_NAME, input._1.dataType, Shape(input._1.shape(1) + numUnits, numUnits), kernelInitializer)
    val bias = variable(RNNCell.BIAS_NAME, input._1.dataType, Shape(numUnits), biasInitializer)
    val linear = ops.NN.addBias(ops.Math.matmul(ops.Basic.concatenate(Seq(input._1, input._2), axis = 1), kernel), bias)
    val output = activation(linear)
    LayerInstance(input, (output, output), Set(kernel, bias))
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

  override def stateSize: Int = numUnits
  override def outputSize: Int = numUnits

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), (Output, Output)] = {
    if (input._1.rank != 2)
      throw InvalidArgumentException(s"Input to 'BasicRNNCell' must be rank-2 (provided rank-${input._1.rank}).")
    if (input._1.shape(1) == -1)
      throw InvalidArgumentException(s"Last axis of input (shape=${input._1.shape}) to 'BasicRNNCell' must be known.")
    val gateKernel = variable(
      s"Gate/${RNNCell.KERNEL_NAME}",
      input._1.dataType,
      Shape(input._1.shape(1) + numUnits, 2 * numUnits),
      kernelInitializer)
    val gateBias = variable(
      s"Gate/${RNNCell.BIAS_NAME}",
      input._1.dataType, Shape(2 * numUnits),
      biasInitializer)
    val candidateKernel = variable(
      s"Candidate/${RNNCell.KERNEL_NAME}",
      input._1.dataType,
      Shape(input._1.shape(1) + numUnits, numUnits),
      kernelInitializer)
    val candidateBias = variable(
      s"Candidate/${RNNCell.BIAS_NAME}",
      input._1.dataType,
      Shape(numUnits),
      biasInitializer)
    val gateInputs = ops.NN.addBias(
      ops.Math.matmul(ops.Basic.concatenate(Seq(input._1, input._2), axis = 1), gateKernel), gateBias)
    val value = ops.Basic.splitEvenly(ops.Math.sigmoid(gateInputs), 2, axis = 1)
    val (r, u) = (value(0), value(1))
    val rState = ops.Math.multiply(r, input._2)
    val c = ops.NN.addBias(
      ops.Math.matmul(ops.Basic.concatenate(Seq(input._1, rState), axis = 1), candidateKernel), candidateBias)
    val newH = ops.Math.multiply(u, input._2) + ops.Math.multiply(1 - u, c)
    LayerInstance(input, (newH, newH), Set(gateKernel, gateBias, candidateKernel, candidateBias))
  }
}
