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

package org.platanios.tensorflow.horovod

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
private[horovod] trait Ops {
  /** Creates an op which sums an input tensor over all the Horovod processes.
    *
    * The reduction operation is keyed by the name of the op. The tensor type and shape must be the same on all Horovod
    * processes for a given name. The reduction will not start until all processes are ready to send and receive the
    * tensor.
    *
    * @param  value Tensor to reduce.
    * @param  name  Name for the created op.
    * @return Tensor of the same shape and type as `value` that is summed across all processes.
    */
  def allReduce(value: Output, name: String = "HorovodAllReduce"): Output = {
    Op.Builder("HorovodAllreduce", name)
        .addInput(value)
        .build().outputs(0)
  }

  /** Creates an op which concatenates the input tensor with the same input tensor on all other Horovod processes.
    *
    * The concatenation is done along the first dimension, and so the input tensors on the different processes must
    * have the same rank and shape, except for the first dimension, which is allowed to be different.
    *
    * @param  value Tensor to gather.
    * @param  name  Name for the created op.
    * @return Tensor of the same type as `value`, concatenated along dimension zero across all processes. Its shape is
    *         identical to the input shape, except for the first dimension, which may be greater and is the sum of all
    *         first dimensions of the tensors in the different Horovod processes.
    */
  def allGather(value: Output, name: String = "HorovodAllGather"): Output = {
    Op.Builder("HorovodAllgather", name)
        .addInput(value)
        .build().outputs(0)
  }

  /** Creates an op which broadcasts the input tensor on root rank to the same input tensor on all other Horovod
    * processes.
    *
    * The broadcast operation is keyed by the name of the op. The tensor type and shape must be the same on all Horovod
    * processes for a given name. The broadcast will not start until all processes are ready to send and receive the
    * tensor.
    *
    * @param  value    Tensor to broadcast.
    * @param  rootRank Rank that will send data, other ranks will receive data.
    * @param  name     Name for the created op.
    * @return Tensor of the same shape and type as `value`, with its value broadcasted from root rank.
    */
  def broadcast(value: Output, rootRank: Int, name: String = "HorovodBroadcast"): Output = {
    Op.Builder("HorovodBroadcast", name)
        .addInput(value)
        .setAttribute("root_rank", rootRank)
        .build().outputs(0)
  }
}

private[horovod] object Ops extends Ops {
  private[horovod] object Gradients {
    tf.gradientsRegistry.registerNonDifferentiable("HorovodAllReduce")
    tf.gradientsRegistry.registerNonDifferentiable("HorovodAllGather")
    tf.gradientsRegistry.registerNonDifferentiable("HorovodBroadcast")
  }
}
