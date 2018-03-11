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

package org.platanios.tensorflow.api.config

import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.distruntime.{ClusterDef, JobDef}

import scala.collection.JavaConverters._
import scala.collection.immutable.TreeMap

/** Represents a cluster as a set of "tasks", organized into "jobs".
  *
  * A [[ClusterConfig]] represents the set of processes that participate in a distributed TensorFlow computation.
  * Every TensorFlow server is constructed in a particular cluster.
  *
  * To create a cluster with two jobs and five tasks, you specify the mapping from job names to lists of network
  * addresses (typically hostname-port pairs).
  *
  * For example:
  * {{{
  *   val clusterConfig = ClusterConfig(Map(
  *     "worker" -> JobConfig.from(
  *       "worker0.example.com:2222",
  *       "worker1.example.com:2222",
  *       "worker2.example.com:2222"),
  *     "ps" -> JobConfig.from(
  *       "ps0.example.com:2222",
  *       "ps1.example.com:2222")))
  * }}}
  *
  * Each job may also be specified as a sparse mapping from task indices to network addresses. This enables a server to
  * be configured without needing to know the identity of (for example) all other worker tasks:
  *
  * For example:
  * {{{
  *    val clusterConfig = ClusterConfig(Map(
  *     "worker" -> JobConfig(1 -> "worker1.example.com:2222"),
  *     "ps" -> JobConfig.from(
  *       "ps0.example.com:2222",
  *       "ps1.example.com:2222")))
  * }}}
  *
  * @param  jobSpecs Map mapping one or more job names to job configurations.
  * @author Emmanouil Antonios Platanios
  */
case class ClusterConfig(jobSpecs: Map[String, JobConfig]) extends ProtoSerializable {
  val clusterDef: ClusterDef = {
    val clusterDef = ClusterDef.newBuilder()
    // We sort by job name in order to produce deterministic Proto messages.
    jobSpecs.toSeq.sortBy(_._1).foreach(j => {
      val jobDef = JobDef.newBuilder()
      jobDef.setName(j._1)
      j._2.tasks.foreach(task => {
        jobDef.putTasks(task._1, task._2)
      })
      clusterDef.addJob(jobDef)
    })
    clusterDef.build()
  }

  /** Returns the set of jobs defined in this cluster configuration. */
  def jobs: Set[String] = jobSpecs.keySet

  /** Returns a map from task index to network address for the tasks included in the provided job.
    *
    * @param  job Job name.
    * @return Option containing a map from task index to network address for the tasks included in the provided job.
    *         `None`, if the job cannot be found in this cluster configuration.
    */
  def jobTasks(job: String): Option[Map[Int, String]] = jobSpecs.get(job).map(_.tasks)

  /** Returns the number of tasks defined in the provided job.
    *
    * @param  job Job name.
    * @return Option containing the number of tasks defined in the provided job. `None`, if the job cannot be found in
    *         this cluster configuration.
    */
  def numTasks(job: String): Option[Int] = jobSpecs.get(job).map(_.tasks.size)

  /** Returns a sequence of valid task indices for the provided job.
    *
    * @param  job Job name.
    * @return Option containing the sorted sequence of task indices defined in the provided job. `None`, if the job
    *         cannot be found in this cluster configuration.
    */
  def taskIndices(job: String): Option[Seq[Int]] = jobSpecs.get(job).map(_.tasks.keys.toSeq)

  /** Returns the address of the task with index `taskIndex` in the provided job.
    *
    * @param  job       Job name.
    * @param  taskIndex Task index.
    * @return Option containing the network address of the the specified task. `None`, if the provided job cannot be
    *         found in this cluster configuration, or if the provided task index cannot be found for that job.
    */
  def taskAddress(job: String, taskIndex: Int): Option[String] = jobSpecs.get(job).flatMap(_.tasks.get(taskIndex))

  /** Constructs and returns a [[ClusterDef]] object that represents this cluster configuration.
    *
    * @return Constructed [[ClusterDef]].
    */
  def toClusterDef: ClusterDef = clusterDef

  /** Constructs and returns a [[ClusterDef]] object that represents this cluster configuration.
    *
    * @return Constructed [[ClusterDef]].
    */
  override def toProto: GeneratedMessageV3 = toClusterDef
}

/** Contains helper methods for dealing with [[ClusterConfig]]s. */
object ClusterConfig {
  /** Constructs a [[ClusterConfig]] from the provided [[ClusterDef]] serialized representation.
    *
    * @param  clusterDef Protobuf-serialized representation of a [[ClusterConfig]].
    * @return Constructed [[ClusterConfig]]
    */
  def fromClusterDef(clusterDef: ClusterDef): ClusterConfig = {
    ClusterConfig(clusterDef.getJobList.asScala.map(jobDef => {
      (jobDef.getName, JobConfig(jobDef.getTasksMap.asScala.toSeq.map(t => (t._1.intValue(), t._2)): _*))
    }).toMap)
  }
}

/** Job configuration (excluding the job name).
  *
  * @param  tasks Mapping from task index to the corresponding task network address.
  */
case class JobConfig(tasks: TreeMap[Int, String])

/** Contains helper methods for dealing with [[JobConfig]]s. */
object JobConfig {
  /** Constructs a [[JobConfig]] using the provided sequence of task index and network address pairs. */
  def apply(tasks: (Int, String)*): JobConfig = JobConfig(TreeMap(tasks: _*))

  /** Constructs a [[JobConfig]] by treating the provided sequence of strings as a dense list of network addresses. */
  def from(tasks: String*): JobConfig = JobConfig(TreeMap(tasks.indices.zip(tasks): _*))

  /** Constructs a [[JobConfig]] by treating the provided sequence of strings as a dense list of network addresses. */
  def fromSeq(tasks: Seq[String]): JobConfig = JobConfig(TreeMap(tasks.indices.zip(tasks): _*))

  /** Constructs a [[JobConfig]] using the provided mapping from task indices to network addresses. */
  def fromMap(tasks: Map[Int, String]): JobConfig = JobConfig(TreeMap(tasks.toSeq: _*))
}
