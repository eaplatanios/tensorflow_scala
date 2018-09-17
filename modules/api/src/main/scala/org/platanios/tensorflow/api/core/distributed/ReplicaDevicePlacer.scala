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

package org.platanios.tensorflow.api.core.distributed

import org.platanios.tensorflow.api.config.ClusterConfig
import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.OpSpecification

/** Device placement strategy to use in a replicated training setup.
  *
  * @param  psNumTasks   Number of parameter server tasks.
  * @param  psDevice     Name of the parameter server device. If empty, no parameter server job is used.
  * @param  workerDevice Name of the worker device. If empty, no worker job is used.
  * @param  psOpTypes    Set of strings representing op types that need to be placed on parameter server devices.
  * @param  psStrategy   Function invoked for every parameter server op (i.e., matched by `psOpTypes`), that takes the
  *                      op and returns the parameter server task index to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class ReplicaDevicePlacer private[distributed](
    psNumTasks: Int, psDevice: String, workerDevice: String, psOpTypes: Set[String],
    psStrategy: OpSpecification => Int) {
  def apply(opSpecification: OpSpecification): String = {
    val currentDevice = DeviceSpecification.fromString(opSpecification.device)

    // The `psJob` will be used for the specified ops (`psOps`) whenever it is present and `psNumTasks` is non-zero.
    // However, its task number will only be set (using `psStrategy`) if there is a job field in `psJob` that won't be
    // changed by the job field (if present) `currentDevice`.
    if (psNumTasks > 0 && psDevice != null && psOpTypes.contains(opSpecification.opType)) {
      var psDeviceSpec = DeviceSpecification.fromString(psDevice)
      val currentJob = currentDevice.job
      val psJob = psDeviceSpec.job
      if (psJob != null && (currentJob == null || currentJob == psJob))
        psDeviceSpec = psDeviceSpec.copy(task = psStrategy(opSpecification))
      DeviceSpecification.merge(psDeviceSpec, currentDevice).toString
    } else {
      val workerDeviceSpec = DeviceSpecification.fromString(if (workerDevice != null) workerDevice else "")
      DeviceSpecification.merge(workerDeviceSpec, currentDevice).toString
    }
  }
}

/** Contains helper methods for dealing with [[ReplicaDevicePlacer]]s. */
object ReplicaDevicePlacer {
  /** Return a device function to use when building a graph for replicas.
    *
    * Device functions are used in the `Op.createWith(deviceFunction = ...)` statement to automatically assign ops to
    * devices as they are being constructed. Device constraints are added from the inner-most context first, working
    * outwards. The merging behavior adds constraints to fields that are yet unset by a more general inner context.
    * Currently the fields include `job`, `task`, and `cpu`/`gpu`.
    *
    * If `clusterConfig` is `null`, and `psNumTasks` is 0, the returned function is a no-op. Otherwise, the value of
    * `psNumTasks` is derived from `clusterConfig`.
    *
    * By default, only variable ops are placed on parameter server tasks and the placement strategy is round-robin over
    * all parameter server tasks. A custom `psStrategy` may be used to do more intelligent device placement.
    *
    * For example:
    * {{{
    *   // To build a cluster with two parameter server jobs on hosts `ps0` and `ps1`, and 3 worker jobs on hosts
    *   // `worker0`, `worker1`, and `worker2`.
    *   val clusterConfig = ClusterConfig(Map(
    *     "ps" -> JobConfig.fromAddresses(
    *       "ps0:2222",
    *       "ps1:2222"),
    *     "worker" -> JobConfig.fromAddresses(
    *       "worker0:2222",
    *       "worker1:2222",
    *       "worker2:2222")))
    *   Op.createWith(ReplicaDeviceSetter(clusterConfig = clusterConfig)) {
    *     // Build the graph.
    *     val v1 = tf.variable(...)  // Assigned to device `/job:ps/task:0`
    *     val v2 = tf.variable(...)  // Assigned to device `/job:ps/task:1`
    *     val v3 = tf.variable(...)  // Assigned to device `/job:ps/task:0`
    *   }
    *   // Run computation.
    * }}}
    *
    * @param  psNumTasks    Number of parameter server tasks. Ignored if `clusterConfig` is provided.
    * @param  psDevice      Name of the parameter server device. If empty, no parameter server job is used.
    * @param  workerDevice  Name of the worker device. If empty, no worker job is used.
    * @param  clusterConfig Cluster configuration.
    * @param  psOpTypes     Set of strings representing op types that need to be placed on parameter server devices.
    * @param  psStrategy    Function invoked for every parameter server op (i.e., matched by `psOpTypes`), that takes
    *                       the op and returns the parameter server task index to use. If `null`, defaults to a
    *                       round-robin strategy across all parameter server devices.
    * @return [[ReplicaDevicePlacer]], whose `apply` method can be passed to `Op.createWith(deviceFunction = ...)`.
    */
  def apply(
      psNumTasks: Int = 0, psDevice: String = "/job:ps", workerDevice: String = "/job:worker",
      clusterConfig: ClusterConfig = null, psOpTypes: Set[String] = Set("Variable", "VariableV2", "VarHandleOp"),
      psStrategy: OpSpecification => Int = null): ReplicaDevicePlacer = {
    val numTasks = {
      if (clusterConfig != null) {
        // Get `psJob` from `psDevice` by stripping "/job:".
        val psJob = DeviceSpecification.fromString(psDevice).job
        val psJobTasks = clusterConfig.jobTasks(psJob)
        if (psJobTasks.isEmpty || psJobTasks.get == null) 0 else psJobTasks.get.size
      } else {
        psNumTasks
      }
    }
    if (numTasks == 0) {
      null
    } else {
      // TODO: [DISTRIBUTED] !!! Variables in the LOCAL_VARIABLES collection should not be placed on the parameter server.
      new ReplicaDevicePlacer(
        numTasks, psDevice, workerDevice, psOpTypes,
        if (psStrategy == null) RoundRobinDeviceSetter(numTasks).apply else psStrategy)
    }
  }

  /** Device placement strategy which returns the next parameter server task index for placement in round-robin order.
    *
    * @param  psNumTasks Number of parameter server tasks to cycle among.
    */
  private[distributed] case class RoundRobinDeviceSetter(psNumTasks: Int) {
    private[this] var nextTask: Int = 0

    def apply(opSpecification: OpSpecification): Int = {
      val task = nextTask
      nextTask = (nextTask + 1) % psNumTasks
      task
    }
  }
}
