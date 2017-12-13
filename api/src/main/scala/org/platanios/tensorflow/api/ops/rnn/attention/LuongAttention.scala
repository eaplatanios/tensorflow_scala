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
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.variables.{OnesInitializer, Variable}
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Output}

/** Luong-style (multiplicative) attention scoring.
  *
  * This attention has two forms. The first is standard Luong attention, as described in:
  * ["Effective Approaches to Attention-based Neural Machine Translation.", EMNLP 2015](https://arxiv.org/abs/1508.04025).
  *
  * The second is the scaled form inspired partly by the normalized form of Bahdanau attention. To enable the second
  * form, construct the object with `weightsScale` set to the value of a scalar scaling variable.
  *
  * @param  numUnits       Number of units in the attention mechanism.
  * @param  scale          If `true`, the scaled form of Luong attention is used.
  * @param  probabilityFn  Function that takes the scores as inputs and produces probabilities.
  * @param  scoreMaskValue Mask value to use for the score before passing it to `probabilityFn`. Defaults to negative
  *                        infinity. Note that this value is only used if `memorySequenceLengths` is not `null`.
  * @param  name           Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
case class LuongAttention(
    numUnits: Int,
    scale: Boolean = false,
    probabilityFn: (Output) => Output = NN.softmax(_),
    scoreMaskValue: Float = Float.NegativeInfinity,
    name: String = "LuongAttention"
) extends Attention[Output, Shape] {
  override def create(memory: Output, memorySequenceLengths: Output): AttentionMechanism[Output, Shape] = {
    LuongAttention.Mechanism(memory, numUnits, scale, probabilityFn, memorySequenceLengths, scoreMaskValue, name)
  }
}

object LuongAttention {
  class Mechanism(
      override protected val memory: Output,
      val numUnits: Int,
      val scale: Boolean = false,
      protected val scoreProbabilityFn: (Output) => Output = NN.softmax(_),
      override val memorySequenceLengths: Output = null,
      override val scoreMaskValue: Float = Float.NegativeInfinity,
      override val name: String = "LuongAttention"
  ) extends AttentionMechanism[Output, Shape](
    memory = memory,
    probabilityFn = (score: Output, _: Output) => scoreProbabilityFn(score),
    memorySequenceLengths = memorySequenceLengths,
    scoreMaskValue = scoreMaskValue
  ) {
    private val memoryWeights: Variable = {
      Variable.getVariable(s"$name/MemoryWeights", memory.dataType, Shape(memory.shape(-1), numUnits))
    }

    /** Scalar used in weight scaling. */
    private val scaleWeights: Variable = {
      if (scale)
        Variable.getVariable(s"$name/ScaleWeights", memory.dataType, Shape(), OnesInitializer)
      else
        null
    }

    override val trainableVariables: Set[Variable] = if (scale) Set(memoryWeights, scaleWeights) else Set(memoryWeights)

    /** Function to apply on the memory before using it for queries. Defaults to the identity function. */
    override val memoryFn: Output => Output = (m: Output) => NN.linear(m, memoryWeights.value)

    override def call(query: Output, previousAlignments: Output): Output = {
      val depth = query.shape(-1)
      val keyUnits = keys.shape(-1)
      if (depth != keyUnits)
        throw InvalidArgumentException(
          "Incompatible or unknown inner dimensions between query and keys. " +
              s"Query (${query.name}) has $depth units. Keys (${keys.name}) have $keyUnits units. " +
              "Perhaps you need to set the number of units of the attention model to the keys' number of units.")

      // Reshape from `[batchSize, depth]` to `[batchSize, 1, depth]` for `matmul`.
      val reshapedQuery = query.expandDims(1)

      // Inner product along the query units dimension. `matmul` shapes: query is [batchSize, 1, depth] and keys is
      // [batchSize, maxTime, depth]. The inner product is asked to transpose the keys' inner shape to get a batched
      // `matmul` on [batchSize, 1, depth] * [batchSize, depth, maxTime], resulting in an output shape of:
      // [batchTime, 1, maxTime]. We then squeeze out the center singleton dimension.
      var score = Math.matmul(reshapedQuery, keys, transposeB = true)
      score = Basic.squeeze(score, Seq(1))
      if (scale)
        score = scaleWeights.value * score

      probabilityFn(score, previousAlignments)
    }
  }

  object Mechanism {
    def apply(
        memory: Output, numUnits: Int, scale: Boolean = false, probabilityFn: (Output) => Output = NN.softmax(_),
        memorySequenceLengths: Output = null, scoreMaskValue: Float = Float.NegativeInfinity,
        name: String = "LuongAttention"
    ): Mechanism = {
      new Mechanism(memory, numUnits, scale, probabilityFn, memorySequenceLengths, scoreMaskValue, name)
    }
  }
}
