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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.types.{DataType, Resource, TF}
import org.platanios.tensorflow.api.implicits.Implicits._

/** Contains functions for constructing ops related to data flow.
  *
  * @author Emmanouil Antonios Platanios
  */
trait DataFlow {
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
  def dynamicPartition[T: TF](
      data: Output[T],
      partitions: Output[Int],
      numberOfPartitions: Int,
      name: String = "DynamicPartition"
  ): Seq[Output[T]] = {
    Op.Builder[(Output[T], Output[Int]), Seq[Output[T]]](
      opType = "DynamicPartition",
      name = name,
      input = (data, partitions)
    ).setAttribute("num_partitions", numberOfPartitions)
        .setGradientFn(dynamicPartitionGradient(_, _)(TF[T]))
        .build().output
  }

  protected def dynamicPartitionGradient[T: TF](
      op: Op[(Output[T], Output[Int]), Seq[Output[T]]],
      outputGradient: Seq[Output[T]]
  ): (Output[T], Output[Int]) = {
    val data = op.input._1
    val indices = op.input._2
    val numberOfPartitions = op.longAttribute("num_partitions").asInstanceOf[Int]
    val prefixShape = Basic.shape(indices).castTo[Int]
    val originalIndices = Basic.reshape(Math.range(Basic.constant(0), Math.prod(prefixShape)), prefixShape)
    val partitionedIndices = dynamicPartition(originalIndices, indices, numberOfPartitions)
    val reconstructed = dynamicStitch(partitionedIndices, outputGradient)
    (Basic.reshape(reconstructed, Basic.shape(data).castTo[Int]), null)
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
    *   val partitionedData = tf.dynamicPartition(x, conditionMask.toInt, 2)
    *   partitionedData(1) = partitioned_data(1) + 1.0
    *   val conditionIndices = tf.dynamicPartition(tf.range(tf.shape(x)(0)), conditionMask.toInt, 2)
    *   x = tf.dynamicStitch(conditionIndices, partitionedData)
    *   // Here x = [1.1, -1., 6.2, 5.3, -1, 8.4] (i.e., the -1 values remained unchanged).
    * }}}
    *
    * @param  indices Tensors containing the indices of the tensors to merge.
    * @param  data    Tensors to merge/stitch together.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def dynamicStitch[T: TF](
      indices: Seq[Output[Int]],
      data: Seq[Output[T]],
      name: String = "DynamicStitch"
  ): Output[T] = {
    Op.Builder[(Seq[Output[Int]], Seq[Output[T]]), Output[T]](
      opType = "DynamicStitch",
      name = name,
      input = (indices, data)
    ).setGradientFn(dynamicStitchGradient(_, _)(TF[T]))
        .build().output
  }

  protected def dynamicStitchGradient[T: TF](
      op: Op[(Seq[Output[Int]], Seq[Output[T]]), Output[T]],
      outputGradient:  Output[T]
  ): (Seq[Output[Int]], Seq[Output[T]]) = {
    val numberOfValues = op.input._2.size
    val indicesGradient = Seq.fill(numberOfValues)(null)
    val valuesGradient = op.input._1.map(Basic.gather(outputGradient, _, axis = 0))
    (indicesGradient, valuesGradient)
  }

  /** Creates an op that creates a new stack and returns a resource handle to it.
    *
    * A stack produces elements in first-in last-out (FILO) order.
    *
    * @param  maxSize     Maximum size of the stack. If negative, the stack size is unlimited.
    * @param  elementType Data type of the elements in the stack.
    * @param  stackName   Overrides the name used for the temporary stack resource. Defaults to the name of the created
    *                     op, which is guaranteed to be unique.
    * @param  name        Name for the created op.
    * @return Created op output, which is a handle to the new stack resource.
    */
  def newStack(
      maxSize: Output[Int],
      elementType: DataType[Any],
      stackName: String = "",
      name: String = "NewStack"
  ): Output[Resource] = {
    Op.Builder[Output[Int], Output[Resource]](
      opType = "StackV2",
      name = name,
      input = maxSize
    ).setAttribute("elem_type", elementType)
        .setAttribute("stack_name", stackName)
        .build().output
  }

  /** Creates an op that pushes an element into a stack and then returns that same element.
    *
    * @param  stackHandle Handle to a stack resource.
    * @param  element     Element to push into the stack.
    * @param  swapMemory  Boolean value indicating whether to swap the element memory to the CPU.
    * @param  name        Name for the created op.
    * @return Created op output, which has the same value as `element`.
    */
  def stackPush[T: TF](
      stackHandle: Output[Resource],
      element: Output[T],
      swapMemory: Boolean = false,
      name: String = "StackPush"
  ): Output[T] = {
    Op.Builder[(Output[Resource], Output[T]), Output[T]](
      opType = "StackPushV2",
      name = name,
      input = (stackHandle, element)
    ).setAttribute("swap_memory", swapMemory)
        .build().output
  }

  /** Creates an op that pops an element from a stack and then returns it.
    *
    * @param  stackHandle Handle to a stack resource.
    * @param  elementType Data type of the elements in the stack.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def stackPop[T: TF](
      stackHandle: Output[Resource],
      elementType: DataType[T],
      name: String = "StackPop"
  ): Output[T] = {
    Op.Builder[Output[Resource], Output[T]](
      opType = "StackPopV2",
      name = name,
      input = stackHandle
    ).setAttribute("elem_type", elementType)
        .build().output
  }

  /** Creates an op that deletes a stack from its resource container.
    *
    * @param  stackHandle Handle to a stack resource.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  def stackClose(
      stackHandle: Output[Resource],
      name: String = "StackClose"
  ): Op[Output[Resource], Unit] = {
    Op.Builder[Output[Resource], Unit](
      opType = "StackCloseV2",
      name = name,
      input = stackHandle
    ).build()
  }
}

object DataFlow extends DataFlow
