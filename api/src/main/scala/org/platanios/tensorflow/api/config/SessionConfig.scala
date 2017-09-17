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

package org.platanios.tensorflow.api.config

import org.platanios.tensorflow.api.config.SessionConfig._

import org.tensorflow.framework._

import scala.collection.JavaConverters._

/**
  *
  * @param  deviceCount                       Map from device type name (e.g., `"CPU"` or `"GPU"`) to maximum number of
  *                                           devices of that type to use. If a particular device type is not found in
  *                                           the map, the system picks an appropriate number.
  * @param  intraOpParallelismThreads         The execution of an individual op (for some op types) can be parallelized
  *                                           on a thread pool. This number specifies the size of that thread pool. 0
  *                                           means that the system picks an appropriate number.
  * @param  interOpParallelismThreads         Nodes that perform blocking operations are enqueued on a thread pool of
  *                                           size `interOpParallelismThreads`, available in each process. 0 means that
  *                                           the system picks an appropriate number. Note that the first session
  *                                           created in the process sets the number of threads for all future sessions
  *                                           unless `sessionInterOpThreadPools` is provided.
  * @param  sessionInterOpThreadPools         '''EXPERIMENTAL''' This option configures session thread pools. If this is
  *                                           provided, then `RunOptions` for a `run` call can select the thread pool to
  *                                           use. The intended use is for when some session invocations need to run in
  *                                           a background pool limited to a small number of threads:
  *                                           - For example, a session may be configured to have one large pool (for
  *                                             regular compute) and one small pool (for periodic, low priority work);
  *                                             using the small pool is currently the mechanism for limiting the
  *                                             inter-op parallelism of the low priority work. Note that it does not
  *                                             limit the parallelism of work spawned by a single op kernel
  *                                             implementation.
  *                                           - Using this setting is normally not needed in training, but may help some
  *                                             serving use cases.
  *                                           - It is also generally recommended to provide a global thread pool name
  *                                             for each thread pool, in order to avoid creating multiple large pools.
  *                                             It is typically better to run the non-low-priority work, even across
  *                                             sessions, in a single large thread pool.
  *                                           Each element of the sequence is a tuple containing a global thread pool
  *                                           name and a thread pool size (and naturally, each element corresponds to a
  *                                           separate thread pool). A `0` value for the thread pool size means that the
  *                                           system picks a value based on where this configuration is used. If a the
  *                                           provided global name for a thread pool is empty, then the thread pool is
  *                                           made and used according to the scope it is defined in -- e.g., for a
  *                                           session thread pool, it is used by that session only. If non-empty, then:
  *                                           - A global thread pool associated with this name is looked up or created.
  *                                             This allows, for example, sharing one thread pool across many sessions
  *                                             (e.g., like the default behavior, if `interOpParallelismThreads` is not
  *                                             configured), but still partitioning into a large and a small pool.
  *                                           - If the thread pool for this global name already exists, then an error is
  *                                             thrown if the existing pool was created using a different
  *                                             `threadPoolSize` value, as is specified in this call.
  *                                           - Thread pools created this way are never garbage collected.
  * @param  allowSoftPlacement                Specified whether soft placement is allowed. If `allowSoftPlacement` is
  *                                           `true`, an op will be placed on the CPU if:
  *                                           1. There's no GPU implementation for the op, or
  *                                           2. No GPU devices are known or registered, or
  *                                           3. It needs to be co-located with any reference-type input(s) it may have,
  *                                              which are from the CPU.
  * @param  logDevicePlacement                Specifies whether device placements should be logged.
  * @param  placementPeriod                   Assignment of nodes to devices is recomputed every `placementPeriod` steps
  *                                           until the system warms up (at which point the re-computation typically
  *                                           slows down automatically).
  * @param  deviceFilters                     When any device filters are present sessions will ignore all devices which
  *                                           do not match the filters. Each filter can be partially specified. For
  *                                           example, `/job:ps`, `/job:worker/replica:3`, etc.
  * @param  globalOpTimeoutMillis             Global timeout for all blocking operations in this session, in
  *                                           milliseconds. If non-zero and not overridden on a per-operation basis,
  *                                           this value will be used as the deadline for all blocking operations.
  * @param  optLevel                          Overall optimization level. The actual optimizations applied will be the
  *                                           logical OR of the flags that this level implies and any flags already set.
  * @param  optCommonSubExpressionElimination If `true`, the graph is optimized using common subexpression elimination.
  * @param  optConstantFolding                If `true`, the graph is optimized using constant folding.
  * @param  optFunctionInlining               If `true`, the graph is optimized using function inlining.
  * @param  optGlobalJITLevel                 '''EXPERIMENTAL''' Graph JIT compilation level.
  * @param  graphEnableReceiveScheduling      If `true`, use control flow to schedule the activation of receive nodes.
  * @param  graphCostModelSteps               Number of steps to run before returning a cost model detailing the memory
  *                                           usage and performance of each node in the graph. `0` means no cost model
  *                                           is built.
  * @param  graphCostModelSkipSteps           Number of steps to skip before collecting statistics for the cost model.
  * @param  graphInferShapes                  If `true`, annotate each graph node with op output shape data, to the
  *                                           extent that the shapes can be statically inferred.
  * @param  graphPlacePruned                  If `true`, only place the sub-graphs that are run, rather than the entire
  *                                           graph. This is useful for interactive graph building, where one might
  *                                           produce graphs that cannot be placed during the debugging process. In
  *                                           particular, it allows the client to continue work in a session after
  *                                           adding a node to a graph whose placement constraints are unsatisfiable.
  * @param  graphEnableBFloat16SendReceive    If `true`, transfer floating-point values between processes as `BFLOAT16`.
  * @param  graphTimelineSteps                '''EXPERIMENTAL''' If `> 0`, record a timeline every this many steps.
  *                                           Currently, this option has no effect in `MasterSession`.
  * @param  gpuAllocationStrategy             Type of GPU allocation strategy to use.
  * @param  gpuAllowMemoryGrowth              If `true`, the GPU allocator does not pre-allocate the entire specified
  *                                           GPU memory region, instead starting small and growing as needed.
  * @param  gpuPerProcessMemoryFraction       A value between 0 and 1 that indicates what fraction of the available GPU
  *                                           memory to pre-allocate for each process. 1 means to pre-allocate all of
  *                                           the GPU memory, while 0.5 means to pre-allocate ~50% of the available GPU
  *                                           memory.
  * @param  gpuDeferredDeletionBytes          Delay the deletion of up to this many bytes to reduce the number of
  *                                           interactions with GPU driver code. If 0, the system chooses a reasonable
  *                                           default (several MBs).
  * @param  gpuVisibleDevices                 List of GPU IDs that determines the 'visible' to 'virtual' mapping of GPU
  *                                           devices. For example, if TensorFlow can see 8 GPU devices in the process,
  *                                           and one wanted to map visible GPU devices 5 and 3 as `/device:GPU:0`, and
  *                                           `/device:GPU:1`, then one would specify this field as `Seq(5, 3)`. This
  *                                           field is similar in spirit to the `CUDA_VISIBLE_DEVICES` environment
  *                                           variable, except that it applies to the visible GPU devices in the
  *                                           process.
  *                                           '''NOTE:''' The GPU driver provides the process with the visible GPUs in
  *                                           an order which is not guaranteed to have any correlation to the *physical*
  *                                           GPU ID in the machine. This field is used for remapping "visible" to
  *                                           "virtual", which means this operates only after the process starts. Users
  *                                           are required to use vendor specific mechanisms (e.g.,
  *                                           `CUDA_VISIBLE_DEVICES`) to control the physical to visible device mapping
  *                                           prior to invoking TensorFlow.
  * @param  gpuPollingActiveDelayMicros       In the event polling loop sleep this many microseconds between
  *                                           `PollEvents` calls, when the queue is not empty. If the value is not set
  *                                           or set to 0, it gets set to a non-zero default.
  * @param  gpuPollingInactiveDelayMillis     In the event polling loop sleep this many milliseconds between
  *                                           `PollEvents` calls, when the queue is empty.If the value is not set or set
  *                                           to 0, it gets set to a non-zero default.
  * @param  gpuForceCompatible                If `true`, force all tensors to be GPU-compatible. On a GPU-enabled
  *                                           TensorFlow installation, enabling this option forces all CPU tensors to be
  *                                           allocated with CUDA-pinned memory. Normally, TensorFlow will infer which
  *                                           tensors should be allocated with pinned memory. But in case where the
  *                                           inference is incomplete, this option can significantly speed up the
  *                                           cross-device memory copy performance as long as the tensors all fit in the
  *                                           GPU memory. Note that this option is not something that should be enabled
  *                                           by default for unknown or very large models, since all CUDA-pinned memory
  *                                           is unpageable and having too much pinned memory might negatively impact
  *                                           the overall host system performance.
  * @param  rpcUseInProcess                   If `true`, always use RPC to contact the session target. If `false` (the
  *                                           default option), TensorFlow may use an optimized transport for
  *                                           client-master communication that avoids the RPC stack. This option is
  *                                           primarily used for testing the RPC stack.
  * @param  clusterConfig                     Cluster configuration that contains all workers to use in the session.
  *
  * @author Emmanouil Antonios Platanios
  */
case class SessionConfig(
    deviceCount: Option[Map[String, Int]] = None,
    intraOpParallelismThreads: Option[Int] = None,
    interOpParallelismThreads: Option[Int] = None,
    usePerSessionThreads: Option[Boolean] = None,
    sessionInterOpThreadPools: Seq[(Option[String], Option[Int])] = Seq.empty,
    allowSoftPlacement: Option[Boolean] = None,
    logDevicePlacement: Option[Boolean] = None,
    placementPeriod: Option[Int] = None,
    deviceFilters: Set[String] = Set.empty,
    globalOpTimeoutMillis: Option[Long] = None,
    optLevel: Option[GraphOptimizationLevel] = None,
    optCommonSubExpressionElimination: Option[Boolean] = None,
    optConstantFolding: Option[Boolean] = None,
    optFunctionInlining: Option[Boolean] = None,
    optGlobalJITLevel: Option[GraphOptimizerGlobalJITLevel] = None,
    graphEnableReceiveScheduling: Option[Boolean] = None,
    graphCostModelSteps: Option[Long] = None,
    graphCostModelSkipSteps: Option[Long] = None,
    graphInferShapes: Option[Boolean] = None,
    graphPlacePruned: Option[Boolean] = None,
    graphEnableBFloat16SendReceive: Option[Boolean] = None,
    graphTimelineSteps: Option[Int] = None,
    // TODO: [[CONFIG]] Add support for `RewriterConfig`.
    gpuAllocationStrategy: Option[GPUAllocationStrategy] = None,
    gpuAllowMemoryGrowth: Option[Boolean] = None,
    gpuPerProcessMemoryFraction: Option[Double] = None,
    gpuDeferredDeletionBytes: Option[Long] = None,
    gpuVisibleDevices: Option[Seq[Int]] = None,
    gpuPollingActiveDelayMicros: Option[Int] = None,
    gpuPollingInactiveDelayMillis: Option[Int] = None,
    gpuForceCompatible: Option[Boolean] = None,
    rpcUseInProcess: Option[Boolean] = None,
    clusterConfig: Option[ClusterConfig] = None
) {
  val configProto: ConfigProto = {
    val configProto = ConfigProto.newBuilder()
    deviceCount.foreach(d => configProto.putAllDeviceCount(d.mapValues(c => new Integer(c)).asJava))
    intraOpParallelismThreads.foreach(configProto.setIntraOpParallelismThreads)
    interOpParallelismThreads.foreach(configProto.setInterOpParallelismThreads)
    usePerSessionThreads.foreach(configProto.setUsePerSessionThreads)
    if (sessionInterOpThreadPools.nonEmpty) {
      sessionInterOpThreadPools.foreach(tp => {
        val threadPoolOptions = ThreadPoolOptionProto.newBuilder()
        tp._1.foreach(threadPoolOptions.setGlobalName)
        tp._2.foreach(threadPoolOptions.setNumThreads)
        configProto.addSessionInterOpThreadPool(threadPoolOptions)
      })
    }
    allowSoftPlacement.foreach(configProto.setAllowSoftPlacement)
    placementPeriod.foreach(configProto.setPlacementPeriod)
    logDevicePlacement.foreach(configProto.setLogDevicePlacement)
    if (deviceFilters.nonEmpty) {
      configProto.addAllDeviceFilters(deviceFilters.asJava)
    }
    globalOpTimeoutMillis.foreach(configProto.setOperationTimeoutInMs)
    if (optLevel.isDefined ||
        optCommonSubExpressionElimination.isDefined ||
        optConstantFolding.isDefined ||
        optFunctionInlining.isDefined ||
        graphEnableReceiveScheduling.isDefined ||
        graphCostModelSteps.isDefined ||
        graphCostModelSkipSteps.isDefined ||
        graphInferShapes.isDefined ||
        graphPlacePruned.isDefined ||
        graphEnableBFloat16SendReceive.isDefined ||
        graphTimelineSteps.isDefined) {
      val graphOptions = GraphOptions.newBuilder()
      if (optLevel.isDefined ||
          optCommonSubExpressionElimination.isDefined ||
          optConstantFolding.isDefined ||
          optFunctionInlining.isDefined) {
        val optOptions = OptimizerOptions.newBuilder()
        optLevel.foreach(l => optOptions.setOptLevel(l.level))
        optCommonSubExpressionElimination.foreach(optOptions.setDoCommonSubexpressionElimination)
        optConstantFolding.foreach(optOptions.setDoConstantFolding)
        optFunctionInlining.foreach(optOptions.setDoFunctionInlining)
        graphOptions.setOptimizerOptions(optOptions)
      }
      graphEnableReceiveScheduling.foreach(graphOptions.setEnableRecvScheduling)
      graphCostModelSteps.foreach(graphOptions.setBuildCostModel)
      graphCostModelSkipSteps.foreach(graphOptions.setBuildCostModelAfter)
      graphInferShapes.foreach(graphOptions.setInferShapes)
      graphPlacePruned.foreach(graphOptions.setPlacePrunedGraph)
      graphEnableBFloat16SendReceive.foreach(graphOptions.setEnableBfloat16Sendrecv)
      graphTimelineSteps.foreach(graphOptions.setTimelineStep)
      configProto.setGraphOptions(graphOptions)
    }
    if (gpuAllowMemoryGrowth.isDefined ||
        gpuPerProcessMemoryFraction.isDefined ||
        gpuAllocationStrategy.isDefined ||
        gpuDeferredDeletionBytes.isDefined ||
        gpuVisibleDevices.isDefined ||
        gpuPollingActiveDelayMicros.isDefined ||
        gpuPollingInactiveDelayMillis.isDefined ||
        gpuForceCompatible.isDefined) {
      val gpuOptions = GPUOptions.newBuilder()
      gpuAllowMemoryGrowth.foreach(gpuOptions.setAllowGrowth)
      gpuPerProcessMemoryFraction.foreach(gpuOptions.setPerProcessGpuMemoryFraction)
      gpuAllocationStrategy.foreach(s => gpuOptions.setAllocatorType(s.name))
      gpuDeferredDeletionBytes.foreach(gpuOptions.setDeferredDeletionBytes)
      gpuVisibleDevices.foreach(d => gpuOptions.setVisibleDeviceList(d.mkString(",")))
      gpuPollingActiveDelayMicros.foreach(gpuOptions.setPollingActiveDelayUsecs)
      gpuPollingInactiveDelayMillis.foreach(gpuOptions.setPollingInactiveDelayMsecs)
      gpuForceCompatible.foreach(gpuOptions.setForceGpuCompatible)
      configProto.setGpuOptions(gpuOptions)
    }
    if (rpcUseInProcess.isDefined) {
      val rpcOptions = RPCOptions.newBuilder()
      rpcUseInProcess.foreach(rpcOptions.setUseRpcForInprocessMaster)
      configProto.setRpcOptions(rpcOptions)
    }
    clusterConfig.foreach(c => configProto.setClusterDef(c.toClusterDef))
    configProto.build()
  }
}

object SessionConfig {
  /** Graph optimization level. */
  sealed trait GraphOptimizationLevel {
    def level: OptimizerOptions.Level
  }

  /** No graph optimization performed. */
  case object NoGraphOptimizations extends GraphOptimizationLevel {
    override def level: OptimizerOptions.Level = OptimizerOptions.Level.L0
  }

  /** Level-1 graph optimizations performed, which include common subexpression elimination and constant folding. */
  case object L1GraphOptimizations extends GraphOptimizationLevel {
    override def level: OptimizerOptions.Level = OptimizerOptions.Level.L1
  }

  /** '''EXPERIMENTAL''' Graph JIT compilation level. */
  sealed trait GraphOptimizerGlobalJITLevel {
    def level: OptimizerOptions.GlobalJitLevel
  }

  /** Default graph JIT compilation level, which is currently set to JIT not being used. */
  case object DefaultGraphOptimizerGlobalJIT extends GraphOptimizerGlobalJITLevel {
    override def level: OptimizerOptions.GlobalJitLevel = OptimizerOptions.GlobalJitLevel.DEFAULT
  }

  /** No JIT compilation performed. */
  case object NoGraphOptimizerGlobalJIT extends GraphOptimizerGlobalJITLevel {
    override def level: OptimizerOptions.GlobalJitLevel = OptimizerOptions.GlobalJitLevel.OFF
  }

  /** Level-1 JIT compilation performed. Higher JIT levels are more aggressive. Higher levels may reduce opportunities
    * for parallelism and may use more memory (at present, there is no distinction between L1 and L2 but this is
    * expected to change). */
  case object L1GraphOptimizerGlobalJIT extends GraphOptimizerGlobalJITLevel {
    override def level: OptimizerOptions.GlobalJitLevel = OptimizerOptions.GlobalJitLevel.ON_1
  }

  /** Level-2 JIT compilation performed. Higher JIT levels are more aggressive. Higher levels may reduce opportunities
    * for parallelism and may use more memory (at present, there is no distinction between L1 and L2 but this is
    * expected to change). */
  case object L2GraphOptimizerGlobalJIT extends GraphOptimizerGlobalJITLevel {
    override def level: OptimizerOptions.GlobalJitLevel = OptimizerOptions.GlobalJitLevel.ON_2
  }

  /** GPU allocation strategy. */
  sealed trait GPUAllocationStrategy {
    def name: String
  }

  /** System-chosen default GPU allocation strategy which may change over time. */
  case object DefaultGPUAllocation extends GPUAllocationStrategy {
    override def name: String = ""
    override def toString: String = "DefaultGPUAllocation"
  }

  /** A "best-fit with coalescing" GPU allocation algorithm, simplified from a version of `dlmalloc`. */
  case object BestFitWithCoalescingGPUAllocation extends GPUAllocationStrategy {
    override def name: String = "BFC"
    override def toString: String = "BestFitWithCoalescingGPUAllocation"
  }
}
