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

import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Output}

/** Luong-style (multiplicative) attention scoring.
  *
  * This attention has two forms. The first is standard Luong attention, as described in:
  * ["Effective Approaches to Attention-based Neural Machine Translation.", EMNLP 2015](https://arxiv.org/abs/1508.04025).
  *
  * The second is the scaled form inspired partly by the normalized form of Bahdanau attention. To enable the second
  * form, construct the object with `weightsScale` set to the value of a scalar scaling variable.
  *
  * @param  memory                Memory to query; usually the output of an RNN encoder. Each tensor in the memory
  *                               should be shaped `[batchSize, maxTime, ...]`.
  * @param  memoryWeights         Weights tensor with which the memory is multiplied to produce the attention keys.
  * @param  memorySequenceLengths Sequence lengths for the batch entries in the memory. If provided, the memory tensor
  *                               rows are masked with zeros for values past the respective sequence lengths.
  * @param  scaleFactor           Scalar tensor with which the scores are multiplied before used to compute attention
  *                               probabilities.
  * @param  scoreMaskValue        Mask value to use for the score before passing it to `probabilityFn`. Defaults to
  *                               negative infinity. Note that this value is only used if `memorySequenceLengths` is not
  *                               `null`.
  * @param  name                  Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class LuongAttention(
    override protected val memory: Output,
    protected val memoryWeights: Output,
    override protected val memorySequenceLengths: Output = null,
    protected val scaleFactor: Output = null,
    override val scoreMaskValue: Output = Float.NegativeInfinity,
    override val name: String = "LuongAttention"
) extends Attention(memory, memorySequenceLengths, checkInnerDimensionsDefined = true, scoreMaskValue, name) {
  override val keys: Output = NN.linear(values, memoryWeights)

  override protected def score(query: Output, previousAlignment: Output): Output = {
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
    if (scaleFactor == null)
      score
    else
      scaleFactor * score
  }
}

object LuongAttention {
  def apply(
      memory: Output,
      memoryWeights: Output,
      memorySequenceLengths: Output = null,
      scaleWeights: Output = null,
      scoreMaskValue: Output = Float.NegativeInfinity,
      name: String = "LuongAttention"
  ): LuongAttention = {
    new LuongAttention(memory, memoryWeights, memorySequenceLengths, scaleWeights, scoreMaskValue, name)
  }
}
