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

import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.INT32

/** Contains functions for constructing ops related to data flow.
  *
  * @author Emmanouil Antonios Platanios
  */
private[ops] trait DataFlow {
  /** Creates an op that partitions `data` into `numberOfPartitions` tensors using indices from `partitions`.
    *
    * For each index tuple `js` of size `partitions.rank`, the slice `data[js, ...]` becomes part of
    * `outputs[partitions[js]]`. The slices with `partitions[js] = i` are placed in `outputs[i]` in lexicographic order
    * of `js`, and the first dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`. In detail:
    * {{{
    *   outputs(i).shape = [sum(partitions == i)] + data.shape(partitions.rank::)
    *   outputs(i) = pack(js.filter(partitions(_) == i).map(data(_, ---))
    * }}}
    *
    * `data.shape` must start with `partitions.shape`.
    *
    * For example:
    * {{{
    *   // Scalar partitions.
    *   val outputs = dynamicPartition(
    *     data = Tensor(10, 20),
    *     partitions = 1,
    *     numberOfPartitions = 2)
    *   outputs(0) ==> []
    *   outputs(1) ==> [[10, 20]]
    *
    *   // Vector partitions.
    *   val outputs = dynamicPartition(
    *     data = Tensor(10, 20, 30, 40, 50),
    *     partitions = [0, 0, 1, 1, 0],
    *     numberOfPartitions = 2)
    *   outputs(0) ==> [10, 20, 50]
    *   outputs(1) ==> [30, 40]
    * }}}
    *
    * See [[dynamicStitch]] for an example on how to merge partitions back together.
    *
    * @param  data               Tensor to partition.
    * @param  partitions         Tensor containing indices in the range `[0, numberOfPartitions]`.
    * @param  numberOfPartitions Number of partitions to output.
    * @param  name               Name for the created op.
    * @return Created op outputs (i.e., partitions).
    */
  def dynamicPartition(
      data: Output, partitions: Output, numberOfPartitions: Int, name: String = "DynamicPartition"): Seq[Output] = {
    Op.Builder(opType = "DynamicPartition", name = name)
        .addInput(data)
        .addInput(partitions)
        .setAttribute("num_partitions", numberOfPartitions)
        .build().outputs.toSeq
  }

  /** Creates an op that interleaves the values from the `data` tensors into a single tensor.
    *
    * The op builds a merged tensor such that:
    * `merged(indices(m)(i, ---, j), ---) = data(m)(i, ---, j, ---)`
    *
    * For example, if each `indices(m)` is scalar or vector, we have:
    * {{{
    *   // Scalar indices.
    *   merged(indices(m), ---) == data(m)(---)
    *
    *   // Vector indices.
    *   merged(indices(m)(i), ---) == data(m)(i, ---)
    * }}}
    *
    * Each `data(i).shape` must start with the corresponding `indices(i).shape`, and the rest of `data(i).shape` must be
    * constant w.r.t. `i`. That is, we must have `data(i).shape = indices(i).shape + constant`. In terms of this
    * `constant`, the output shape is `merged.shape = [max(indices)] + constant`.
    *
    * Values are merged in order, so if an index appears in both `indices(m)(i)` and `indices(n)(j)` for
    * `(m,i) < (n,j)`, the slice `data(n)(j)` will appear in the merged result.
    *
    * For example:
    * {{{
    *   indices(0) = 6
    *   indices(1) = [4, 1]
    *   indices(2) = [[5, 2], [0, 3]]
    *   data(0) = [61, 62]
    *   data(1) = [[41, 42], [11, 12]]
    *   data(2) = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    *   dynamicStitch(indices, data) ==> [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 62]]
    * }}}
    *
    * This method can be used to merge partitions created by [[dynamicPartition]], as shown in the following example:
    * {{{
    *   // Apply a function that increments x_i on elements for which a certain condition applies
    *   // (x_i != -1, in this example).
    *   var x = tf.constant(Tensor(0.1, -1., 5.2, 4.3, -1., 7.4))
    *   val conditionMask = tf.notEqual(x, tf.constant(-1.0))
    *   val partitionedData = tf.dynamicPartition(x, tf.cast(conditionMask, tf.INT32), 2)
    *   partitionedData(1) = partitioned_data(1) + 1.0
    *   val conditionIndices = tf.dynamicPartition(tf.range(tf.shape(x)(0)), tf.cast(conditionMask, tf.INT32), 2)
    *   x = tf.dynamicStitch(conditionIndices, partitionedData)
    *   // Here x = [1.1, -1., 6.2, 5.3, -1, 8.4] (i.e., the -1 values remained unchanged).
    * }}}
    *
    * @param  indices Tensors containing the indices of the tensors to merge.
    * @param  data    Tensors to merge/stitch together.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def dynamicStitch(indices: Seq[Output], data: Seq[Output], name: String = "DynamicStitch"): Output = {
    Op.Builder(opType = "DynamicStitch", name = name)
        .addInputList(indices)
        .addInputList(data)
        .build().outputs(0)
  }
}

private[ops] object DataFlow extends DataFlow {
  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("Stack")
    GradientsRegistry.registerNonDifferentiable("StackPush")
    GradientsRegistry.registerNonDifferentiable("StackPop")
    GradientsRegistry.registerNonDifferentiable("StackClose")

    GradientsRegistry.registerNonDifferentiable("GetSessionHandle")
    GradientsRegistry.registerNonDifferentiable("GetSessionHandleV2")
    GradientsRegistry.registerNonDifferentiable("GetSessionTensor")
    GradientsRegistry.registerNonDifferentiable("DeleteSessionTensor")

    GradientsRegistry.register("DynamicPartition", dynamicPartitionGradient)
    GradientsRegistry.register("DynamicStitch", dynamicStitchGradient)

    private[this] def dynamicPartitionGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val data = op.inputs(0)
      val indices = op.inputs(1)
      val numberOfPartitions = op.longAttribute("num_partitions").asInstanceOf[Int]
      val prefixShape = Basic.shape(indices)
      val originalIndices = Basic.reshape(Math.range(Basic.constant(0), Math.prod(prefixShape)), prefixShape)
      val partitionedIndices = dynamicPartition(originalIndices, indices, numberOfPartitions)
      val reconstructed = dynamicStitch(partitionedIndices, outputGradients.map(_.toOutput))
      Seq(Basic.reshape(reconstructed, Basic.shape(data)), null)
    }

    private[this] def dynamicStitchGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val numberOfValues = op.inputs.length / 2
      val indicesGradient = Seq.fill(numberOfValues)(null)
      val valuesGradient = op.inputs.take(numberOfValues).map(Math.cast(_, INT32)).map(Basic.gather(outputGradient, _))
      indicesGradient ++ valuesGradient
    }
  }
}
