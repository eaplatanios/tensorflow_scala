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

package org.platanios.tensorflow

import org.platanios.tensorflow.api._

/** Horovod allows training TensorFlow models in a distributed fashion, using MPI for communication between processes.
  *
  * TensorFlow natively provides inter-device communication through send and receive ops and inter-node communication
  * through distributed TensorFlow, based on the same send and receive abstractions. On HPC clusters where Infiniband or
  * other high-speed node interconnects are available, these can end up being insufficient for synchronous data-parallel
  * training (without asynchronous gradient descent). This module implements a variety of MPI ops which can take
  * advantage of hardware-specific MPI libraries for efficient communication.
  *
  * @author Emmanouil Antonios Platanios
  */
package object horovod {
  object hvd {
    /** Initializes Horovod. */
    def initialize(): Unit = Horovod.init()

    /** Returns the Horovod rank of the calling process.
      *
      * @throws IllegalStateException If Horovod has not been initialized yet.
      */
    @throws[IllegalStateException]
    def rank: Int = {
      val cValue = Horovod.rank()
      if (cValue == -1)
        throw new IllegalStateException("Horovod has not been initialized.")
      cValue
    }

    /** Returns the local Horovod rank of the calling process, within the node that it is running on. For example, if
      * there are seven processes running on a node, their local ranks will be zero through six, inclusive.
      *
      * @throws IllegalStateException If Horovod has not been initialized yet.
      */
    @throws[IllegalStateException]
    def localRank: Int = {
      val cValue = Horovod.localRank()
      if (cValue == -1)
        throw new IllegalStateException("Horovod has not been initialized.")
      cValue
    }

    /** Returns the number of Horovod processes.
      *
      * @throws IllegalStateException If Horovod has not been initialized yet.
      */
    @throws[IllegalStateException]
    def size: Int = {
      val cValue = Horovod.size()
      if (cValue == -1)
        throw new IllegalStateException("Horovod has not been initialized.")
      cValue
    }

    /** Returns the number of Horovod processes within the node the current process is running on.
      *
      * @throws IllegalStateException If Horovod has not been initialized yet.
      */
    @throws[IllegalStateException]
    def localSize: Int = {
      val cValue = Horovod.localSize()
      if (cValue == -1)
        throw new IllegalStateException("Horovod has not been initialized.")
      cValue
    }

    /** Returns a flag indicating whether MPI multi-threading is supported. If MPI multi-threading is supported, users
      * may mix and match Horovod usage with other MPI libraries.
      *
      * @throws IllegalStateException If Horovod has not been initialized yet.
      */
    @throws[IllegalStateException]
    def mpiThreadsSupported: Boolean = {
      val cValue = Horovod.mpiThreadsSupported()
      if (cValue == -1)
        throw new IllegalStateException("Horovod has not been initialized.")
      cValue > 0
    }

    /** Performs an all-reduce operation on `value`.
      *
      * This function performs a bandwidth-optimal ring all-reduce on the input tensor. If the input is indexed slices,
      * then this function instead does an all-gather on the values and the indices, effectively doing an all-reduce on
      * the represented tensor.
      *
      * @param  value        Value to reduce.
      * @param  average      If `true`, the average over all ranks will be computed.
      * @param  deviceDense  Device to use for dense tensor reduce operations. Defaults to a GPU if Horovod was built
      *                      with `HOROVOD_GPU_ALLREDUCE`.
      * @param  deviceSparse Device to use for sparse tensor reduce operations. Defaults to a GPU if Horovod was built
      *                      with `HOROVOD_GPU_ALLGATHER`.
      * @return Reduced tensor value.
      */
    def allReduce[O <: OutputLike](
        value: O,
        average: Boolean = true,
        deviceDense: String = "",
        deviceSparse: String = ""
    ): O = value match {
      case v: OutputIndexedSlices => tf.device(deviceSparse) {
        // For indexed slices we do two all-gathers instead of an all-reduce.
        val horovodSize = tf.constant(size, v.dataType)
        var values = allGatherOp(v.values, name = s"${v.values.name.replace(":", "_")}/AllGather")
        val indices = allGatherOp(v.indices, name = s"${v.indices.name.replace(":", "_")}/AllGather")

        // To convert this operation to an average, we divide all gathered values by the Horovod size.
        values = if (average) tf.divide(values, horovodSize) else values
        OutputIndexedSlices(indices, values, v.denseShape).asInstanceOf[O]
      }
      case v => tf.device(deviceDense) {
        // TODO: [HOROVOD] What about sparse tensors?
        val horovodSize = tf.constant(size, v.dataType)
        val summedValue = allReduceOp(v.toOutput, name = s"${v.name.replace(":", "_")}/AllReduce")
        if (average)
          tf.divide(summedValue, horovodSize).asInstanceOf[O]
        else
          summedValue.asInstanceOf[O]
      }
    }

    /** Broadcasts all global variables from root rank to all other processes.
      *
      * @param  rootRank Rank of the process from which the global variable values will be broadcasted to all other
      *                  processes.
      * @return Created broadcast op.
      */
    def broadcastGlobalVariables(rootRank: Int): Op = {
      tf.group(tf.currentGraph.globalVariables.map(v => {
        Op.Builder(opType = "AssignVariableOp", name = s"${v.name}/Broadcast/Assign")
            .addInput(v.op.outputs.head)
            .addInput(broadcastOp(v.value, rootRank, s"${v.name}/Broadcast"))
            .setAttribute("dtype", v.dataType)
            .build()
      }))
    }

    /** Hooks that will broadcast all global variables from root rank to all other processes during initialization.
      *
      * This is necessary to ensure consistent initialization of all workers when training is started with random
      * weights or restored from a checkpoint.
      *
      * @param  rootRank Rank of the process from which the global variable values will be broadcasted to all other
      *                  processes.
      * @param  device   Device to be used for broadcasting. Defaults to a GPU if Horovod was built with
      *                  `HOROVOD_GPU_BROADCAST`.
      */
    case class BroadcastGlobalVariablesHook(rootRank: Int, device: String = "")
        extends tf.learn.Hook {
      protected var broadcastOp: Option[Op] = None

      override protected def begin(): Unit = {
        if (broadcastOp.isEmpty || broadcastOp.get.graph != tf.currentGraph) {
          tf.device(device) {
            broadcastOp = Some(broadcastGlobalVariables(rootRank))
          }
        }
      }

      override protected def afterSessionCreation(session: Session): Unit = {
        broadcastOp.foreach(op => session.run(targets = op))
      }
    }

    class DistributedOptimizer protected(
        val optimizer: tf.train.Optimizer,
        val name: String = "DistributedOptimizer",
        val deviceDense: String = "",
        val deviceSparse: String = ""
    ) extends tf.train.Optimizer {
      /** Boolean value indicating whether to apply use locks to prevent concurrent updates to variables. */
      override val useLocking: Boolean = optimizer.useLocking

      /** Boolean value indicating whether to ignore duplicate indices during sparse updates. */
      override val ignoreDuplicateSparseIndices: Boolean = optimizer.ignoreDuplicateSparseIndices

      /** Computes the gradients of `loss` with respect to the variables in `variables`, if provided, otherwise with
        * respect to all the trainable variables in the graph where `loss` is defined.
        *
        * @param  loss                       Loss value whose gradients will be computed.
        * @param  lossGradients              Optional gradients to back-propagate for `loss`.
        * @param  variables                  Optional list of variables for which to compute the gradients. Defaults to
        *                                    the set of trainable variables in the graph where `loss` is defined.
        * @param  gradientsGatingMethod      Gating method for the gradients computation.
        * @param  gradientsAggregationMethod Aggregation method used to combine gradient terms.
        * @param  colocateGradientsWithOps   Boolean value indicating whether to colocate the gradient ops with the
        *                                    original ops.
        * @return Sequence of gradient-variable pairs.
        */
      override def computeGradients(
          loss: Output,
          lossGradients: Seq[OutputLike] = null,
          variables: Set[Variable] = null,
          gradientsGatingMethod: tf.gradients.GatingMethod = tf.gradients.OpGating,
          gradientsAggregationMethod: tf.gradients.AggregationMethod = tf.gradients.AddAggregationMethod,
          colocateGradientsWithOps: Boolean = false
      ): Seq[(OutputLike, Variable)] = {
        val gradients = super.computeGradients(
          loss, lossGradients, variables, gradientsGatingMethod, gradientsAggregationMethod, colocateGradientsWithOps)
        if (size <= 1) {
          gradients
        } else {
          tf.createWithNameScope(s"$name/AllReduce") {
            gradients.map(gv => {
              if (gv._1 != null) {
                (allReduce(gv._1, deviceDense = deviceDense, deviceSparse = deviceSparse), gv._2)
              } else {
                gv
              }
            })
          }
        }
      }

      /** Creates an op that applies the provided gradients to the provided variables.
        *
        * @param  gradientsAndVariables Sequence with gradient-variable pairs.
        * @param  iteration             Optional `Variable` to increment by one after the variables have been updated.
        * @param  name                  Name for the created op.
        * @return Created op.
        */
      override def applyGradients(
          gradientsAndVariables: Seq[(OutputLike, Variable)],
          iteration: Option[Variable] = None,
          name: String = this.name
      ): Op = {
        optimizer.applyGradients(gradientsAndVariables, iteration, name)
      }

      /** Supported data types for the loss function, the variables, and the gradients. Subclasses should override this
        * field allow other float types. */
      override val supportedDataTypes: Set[DataType[_]] = {
        optimizer.supportedDataTypes
      }

      /** Create all slots needed by this optimizer. */
      override def createSlots(variables: Seq[Variable]): Unit = {
        optimizer.createSlots(variables)
      }

      /** Creates all necessary tensors before applying the gradients. This function is called from within an op
        * creation context that uses as its name scope the name that users have chosen for the application of
        * gradients. */
      override def prepare(iteration: Option[Variable]): Unit = {
        optimizer.prepare(iteration)
      }

      /** Creates an op that finishes the gradients application. This function is called from within an op creation
        * context that uses as its name scope the name that users have chosen for the application of gradients.
        *
        * @param  updateOps Set of ops needed to apply the gradients and update the variable values.
        * @param  nameScope Name scope to use for all the ops created by this function.
        * @return Created op output.
        */
      override def finish(updateOps: Set[Op], nameScope: String): Op = {
        optimizer.finish(updateOps, nameScope)
      }

      /** Applies the updates corresponding to the provided gradient, to the provided variable.
        *
        * @param  gradient  Gradient tensor.
        * @param  variable  Variable.
        * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
        * @return Created op that applies the provided gradient to the provided variable.
        */
      override def applyDense(
          gradient: Output,
          variable: Variable,
          iteration: Option[Variable]
      ): Op = {
        optimizer.applyDense(gradient, variable, iteration)
      }

      /** Applies the updates corresponding to the provided gradient, to the provided variable.
        *
        * The [[OutputIndexedSlices]] object specified by `gradient` in this function is by default pre-processed in
        * `applySparseDuplicateIndices` to remove duplicate indices (refer to that function's documentation for
        * details). Optimizers which can tolerate or have correct special cases for duplicate sparse indices may
        * override `applySparseDuplicateIndices` instead of this function, avoiding that overhead.
        *
        * @param  gradient  Gradient tensor.
        * @param  variable  Variable.
        * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
        * @return Created op that applies the provided gradient to the provided variable.
        */
      override def applySparse(
          gradient: OutputIndexedSlices,
          variable: Variable,
          iteration: Option[Variable]
      ): Op = {
        optimizer.applySparse(gradient, variable, iteration)
      }

      /** Applies the updates corresponding to the provided gradient (with potentially duplicate indices), to the
        * provided variable.
        *
        * Optimizers which override this method must deal with [[OutputIndexedSlices]] objects such as the following:
        * `OutputIndexedSlices(indices=[0, 0], values=[1, 1], denseShape=[1])`, which contain duplicate indices. The
        * correct interpretation in that case should be: `OutputIndexedSlices(values=[2], indices=[0], denseShape=[1])`.
        *
        * Many optimizers deal incorrectly with repeated indices when updating based on sparse gradients (e.g. summing
        * squares rather than squaring the sum, or applying momentum terms multiple times). Adding first is always the
        * correct behavior, so this is enforced here by reconstructing the [[OutputIndexedSlices]] to have only unique
        * indices, and then calling [[applySparse]].
        *
        * Optimizers which deal correctly with repeated indices may instead override this method to avoid the induced
        * overhead.
        *
        * @param  gradient  Gradient tensor.
        * @param  variable  Variable.
        * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
        * @return Created op that applies the provided gradient to the provided variable.
        */
      override def applySparseDuplicateIndices(
          gradient: OutputIndexedSlices,
          variable: Variable,
          iteration: Option[Variable]
      ): Op = {
        optimizer.applySparseDuplicateIndices(gradient, variable, iteration)
      }
    }

    object DistributedOptimizer {
      def apply(
          optimizer: api.tf.train.Optimizer,
          name: String = "DistributedOptimizer",
          deviceDense: String = "",
          deviceSparse: String = ""
      ): DistributedOptimizer = {
        new DistributedOptimizer(optimizer, name, deviceDense, deviceSparse)
      }
    }
  }

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
  private[horovod] def allReduceOp(value: Output, name: String = "HorovodAllReduce"): Output = {
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
  private[horovod] def allGatherOp(value: Output, name: String = "HorovodAllGather"): Output = {
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
  private[horovod] def broadcastOp(value: Output, rootRank: Int, name: String = "HorovodBroadcast"): Output = {
    Op.Builder("HorovodBroadcast", name)
        .addInput(value)
        .setAttribute("root_rank", rootRank)
        .build().outputs(0)
  }

  tf.gradientsRegistry.registerNonDifferentiable("HorovodAllreduce")
  tf.gradientsRegistry.registerNonDifferentiable("HorovodAllgather")
  tf.gradientsRegistry.registerNonDifferentiable("HorovodBroadcast")
}
