/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.config.{ClusterConfig, JobConfig}
import org.platanios.tensorflow.api.core.client.{Session, SessionConfig}
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.jni.{Server => NativeServer}

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.distruntime.ServerDef

/** In-process TensorFlow server, for use in distributed training.
  *
  * A [[Server]] instance encapsulates a set of devices along with a [[Session]] target, that can participate in
  * distributed training. A server belongs to a cluster (specified by a [[ClusterConfig]] when creating the server), and
  * corresponds to a particular task in a named job. The server can communicate with any other server in the same
  * cluster.
  *
  * @param  serverDef           [[ServerDef]] containing all the configuration options for this server.
  * @param  startImmediately    Boolean indicator, specifying whether to start the server immediately after creating it.
  * @param  nativeHandleWrapper Wrapper around the pointer to the native server object.
  * @param  closeFn             Function used to delete the native server object (i.e., free relevant memory).
  *
  * @author Emmanouil Antonios Platanios
  */
class Server private[distributed] (
    val serverDef: ServerDef,
    val startImmediately: Boolean = true,
    private[this] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends ProtoSerializable with Closeable {
  /** Lock for the native handle. */
  private[Server] def NativeHandleLock = nativeHandleWrapper.Lock

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = nativeHandleWrapper.handle

  if (startImmediately)
    start()

  /** Starts this server. */
  def start(): Unit = NativeHandleLock.synchronized {
    NativeServer.startServer(nativeHandle)
  }

  /** Stops this server. */
  def stop(): Unit = NativeHandleLock.synchronized {
    NativeServer.stopServer(nativeHandle)
  }

  /** Blocks until the server has shut down. This method currently blocks forever. */
  def join(): Unit = {
    var handle = 0L
    NativeHandleLock.synchronized {
      handle = nativeHandle
      if (handle != 0)
        nativeHandleWrapper.referenceCount += 1
    }
    try {
      NativeServer.joinServer(nativeHandle)
    } finally {
      NativeHandleLock.synchronized {
        if (handle != 0)
          nativeHandleWrapper.referenceCount -= 1
        NativeHandleLock.notifyAll()
      }
    }
  }

  /** Returns the target for a [[Session]] to connect to this server. */
  def target: String = NativeHandleLock.synchronized {
    NativeServer.target(nativeHandle)
  }

  /** Constructs and returns a [[ServerDef]] object that represents this session.
    *
    * @return Constructed [[ServerDef]].
    */
  def toServerDef: ServerDef = serverDef

  /** Constructs and returns a [[ServerDef]] object that represents this session.
    *
    * @return Constructed [[ServerDef]].
    */
  override def toProto: GeneratedMessageV3 = toServerDef
}

/** Contains helper methods for creating [[Server]]s. */
object Server {
  /** Creates a new server.
    *
    * @param  clusterConfig    Cluster configuration to use for the new server.
    * @param  job              Name of the job of which the new server is a member. If not provided and `clusterConfig`
    *                          contains only one job, that job is used.
    * @param  task             Task index of the server in its job. If not provided and `clusterConfig` contains only
    *                          one index for `job`, that index is used.
    * @param  protocol         Communication protocol to be used by the new server.
    * @param  sessionConfig    Default session configuration for all sessions that run on the created server.
    * @param  startImmediately Boolean indicator, specifying whether to start the server immediately after creating it.
    * @return Created server.
    */
  def apply(
      clusterConfig: ClusterConfig,
      job: String = null,
      task: Int = -1,
      protocol: Protocol = GRPC,
      sessionConfig: SessionConfig = null,
      startImmediately: Boolean = true
  ): Server = {
    val serverDef: ServerDef = {
      val _job = {
        val jobs = clusterConfig.jobs
        if (job == null && jobs.size == 1) jobs.head
        else if (job == null) throw InvalidArgumentException("A job name must be provided.")
        else job
      }
      val _task = {
        val tasks = clusterConfig.taskIndices(_job)
        if (task == -1 && tasks.isDefined && tasks.get.size == 1) tasks.get.head
        else if (task == -1) throw InvalidArgumentException("A task index must be provided.")
        else task
      }
      val serverDefBuilder =
        ServerDef.newBuilder()
            .setCluster(clusterConfig.toClusterDef)
            .setJobName(_job)
            .setTaskIndex(_task)
            .setProtocol(protocol.name)
      if (sessionConfig != null)
        serverDefBuilder.setDefaultSessionConfig(sessionConfig.toConfigProto)
      serverDefBuilder.build()
    }
    val nativeHandle = NativeServer.newServer(serverDef.toByteArray)
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      nativeHandleWrapper.Lock.synchronized {
        NativeServer.stopServer(nativeHandleWrapper.handle)
        while (nativeHandleWrapper.referenceCount > 0)
          nativeHandleWrapper.Lock.wait()
        NativeServer.deleteServer(nativeHandleWrapper.handle)
        nativeHandleWrapper.handle = 0
      }
    }
    val server = new Server(serverDef, startImmediately, nativeHandleWrapper, closeFn)
    // Keep track of references in the Scala side and notify the native library when the server is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(server, closeFn)
    server
  }

  /** Creates a new single-process server running on the local host.
    *
    * This method is a convenience wrapper for creating a [[Server]] that uses a single-process cluster containing a
    * single task in a job called `"local"`.
    *
    * @param  sessionConfig    Default session configuration for all sessions that run on the created server.
    * @param  startImmediately Boolean indicator, specifying whether to start the server immediately after creating it.
    * @return Created server.
    */
  def local(sessionConfig: SessionConfig = null, startImmediately: Boolean = true): Server = {
    // Specifying port 0 means that the OS will choose a free port for the server.
    Server(
      clusterConfig = ClusterConfig(Map("local" -> JobConfig.from("localhost:0"))),
      protocol = GRPC,
      sessionConfig = sessionConfig,
      startImmediately = startImmediately)
  }
}
