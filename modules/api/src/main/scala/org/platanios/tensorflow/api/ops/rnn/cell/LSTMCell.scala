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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.OutputToShape
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output}

/** The Long-Short Term Memory (LSTM) cell.
  *
  * The op uses optional peep-hole connections, optional cell clipping, and an optional projection layer.
  *
  * The default non-peephole implementation is based on:
  * ["Long Short-Term Memory", S. Hochreiter and J. Schmidhuber. Neural Computation, 9(8):1735-1780, 1997.](http://www.bioinf.jku.at/publications/older/2604.pdf).
  *
  * The peephole implementation is based on:
  * ["Long short-term memory recurrent neural network architectures for large scale acoustic modeling", Hasim Sak, Andrew Senior, and Francoise Beaufays. INTERSPEECH, 2014](https://research.google.com/pubs/archive/43905.pdf).
  *
  * Input tensors must be two-dimensional.
  *
  * @group RNNCellOps
  * @param  kernel           Kernel matrix to use.
  * @param  bias             Bias vector to use.
  * @param  activation       Activation function to use.
  * @param  cellClip         If different than `-1`, then the cell state is clipped by this value prior to the cell
  *                          output activation.
  * @param  wfDiag           If not `null`, then diagonal peep-hole connections are added from the forget gate to the
  *                          state, using these weights.
  * @param  wiDiag           If not `null`, then diagonal peep-hole connections are added from the input gate to the
  *                          state, using these weights.
  * @param  woDiag           If not `null`, then diagonal peep-hole connections are added from the output gate to the
  *                          state, using these weights.
  * @param  projectionKernel If not `null`, then this matrix is used to project the cell output.
  * @param  projectionClip   If different than `-1` and `projectionKernel` not `null`, then the projected output is
  *                          clipped by this value.
  * @param  forgetBias       Forget bias added to the forget gate.
  * @param  name             Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class LSTMCell[T: TF : IsNotQuantized] protected (
    val kernel: Output[T],
    val bias: Output[T],
    val activation: Output[T] => Output[T],
    val cellClip: Float = -1,
    val wfDiag: Output[T] = null,
    val wiDiag: Output[T] = null,
    val woDiag: Output[T] = null,
    val projectionKernel: Output[T] = null,
    val projectionClip: Float = -1,
    val forgetBias: Float = 1.0f,
    val name: String = "LSTMCell"
) extends RNNCell[Output[T], LSTMState[T], Shape, (Shape, Shape)] {
  private val numUnits = bias.shape(0) / 4

  override def outputShape: Shape = {
    if (projectionKernel != null)
      Shape(projectionKernel.shape(1))
    else
      Shape(numUnits)
  }

  override def stateShape: (Shape, Shape) = {
    if (projectionKernel != null)
      (Shape(numUnits), Shape(projectionKernel.shape(1)))
    else
      (Shape(numUnits), Shape(numUnits))
  }

  @throws[IllegalArgumentException]
  override def forward(input: Tuple[Output[T], LSTMState[T]]): Tuple[Output[T], LSTMState[T]] = {
    Op.nameScope(name) {
      val output = input.output
      if (output.rank != 2)
        throw new IllegalArgumentException(s"Input must be rank-2 (provided rank-${output.rank}).")
      if (output.shape(1) == -1)
        throw new IllegalArgumentException(s"Last axis of input shape (${output.shape}) must be known.")
      val one = Basic.constant(1)
      // Parameters of gates are concatenated into one multiply for efficiency.
      val lstmMatrix = NN.addBias(Math.matmul(Basic.concatenate(Seq(output, input.state.m), axis = 1), kernel), bias)
      // i = input gate, j = new input, f = forget gate, o = output gate
      val lstmMatrixBlocks = Basic.splitEvenly[T](lstmMatrix, 4, axis = one)
      val (i, j, f, o) = (lstmMatrixBlocks(0), lstmMatrixBlocks(1), lstmMatrixBlocks(2), lstmMatrixBlocks(3))
      // Diagonal connections
      val forgetBiasTensor = Basic.constant[Float](forgetBias).castTo[T]
      var firstTerm = f + forgetBiasTensor
      if (wfDiag != null)
        firstTerm = firstTerm + Math.multiply(wfDiag, input.state.c)
      var secondTerm = i
      if (wiDiag != null)
        secondTerm = secondTerm + Math.multiply(wiDiag, input.state.c)
      var c = Math.add(
        Math.multiply(input.state.c, Math.sigmoid(firstTerm)),
        Math.multiply(Math.sigmoid(secondTerm), activation(j)))
      if (cellClip != -1) {
        val cellClipTensor = Basic.constant(cellClip).castTo[T]
        c = c.clipByValue(-cellClipTensor, cellClipTensor)
      }
      var m = {
        if (woDiag != null)
          Math.multiply(activation(c), Math.sigmoid(o + Math.multiply(woDiag, c)))
        else
          Math.multiply(activation(c), Math.sigmoid(o))
      }
      // Projection
      if (projectionKernel != null) {
        m = Math.matmul(m, projectionKernel)
        if (projectionClip != -1) {
          val projectionClipTensor = Basic.constant(projectionClip).castTo[T]
          m = m.clipByValue(-projectionClipTensor, projectionClipTensor)
        }
      }
      LSTMTuple(m, LSTMState(c, m))
    }
  }
}

object LSTMCell {
  def apply[T: TF : IsNotQuantized](
      kernel: Output[T],
      bias: Output[T],
      activation: Output[T] => Output[T],
      cellClip: Float = -1,
      wfDiag: Output[T] = null,
      wiDiag: Output[T] = null,
      woDiag: Output[T] = null,
      projectionKernel: Output[T] = null,
      projectionClip: Float = -1,
      forgetBias: Float = 1.0f,
      name: String = "LSTMCell"
  ): LSTMCell[T] = {
    new LSTMCell[T](
      kernel, bias, activation, cellClip, wfDiag, wiDiag, woDiag,
      projectionKernel, projectionClip, forgetBias, name)
  }
}
