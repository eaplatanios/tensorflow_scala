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

package org.platanios.tensorflow.api.core.distributed

import org.platanios.tensorflow.api.ProtoSerializable

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.distruntime.{ClusterDef, JobDef}

import scala.collection.JavaConverters._
import scala.collection.immutable.TreeMap

/** Represents a cluster as a set of "tasks", organized into "jobs".
  *
  * A [[ClusterSpec]] represents the set of processes that participate in a distributed TensorFlow computation.
  * Every TensorFlow server is constructed in a particular cluster.
  *
  * To create a cluster with two jobs and five tasks, you specify the mapping from job names to lists of network
  * addresses (typically hostname-port pairs).
  *
  * For example:
  * {{{
  *   val cluster = ClusterSpec(Map(
  *     WORKER -> JobSpec(
  *       "worker0.example.com:2222",
  *       "worker1.example.com:2222",
  *       "worker2.example.com:2222"),
  *     PARAMETER_SERVER -> JobSpec(
  *       "ps0.example.com:2222",
  *       "ps1.example.com:2222")))
  * }}}
  *
  * Each job may also be specified as a sparse mapping from task indices to network addresses. This enables a server to
  * be configured without needing to know the identity of (for example) all other worker tasks:
  *
  * For example:
  * {{{
  *    val cluster = ClusterSpec(Map(
  *     WORKER -> JobSpec(1 -> "worker1.example.com:2222"),
  *     PARAMETER_SERVER -> JobSpec(
  *       "ps0.example.com:2222",
  *       "ps1.example.com:2222")))
  * }}}
  *
  * @param  jobSpecs Map mapping one or more job names to job specifications.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ClusterSpec(jobSpecs: Map[String, JobSpec]) extends ProtoSerializable {
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

  /** Returns the set of jobs defined in this cluster specification. */
  def jobs: Set[String] = jobSpecs.keySet

  /** Returns a map from task index to network address for the tasks included in the provided job.
    *
    * @param  job Job name.
    * @return Option containing a map from task index to network address for the tasks included in the provided job.
    *         `None`, if the job cannot be found in this cluster specification.
    */
  def jobTasks(job: String): Option[Map[Int, String]] = jobSpecs.get(job).map(_.tasks)

  /** Returns the number of tasks defined in the provided job.
    *
    * @param  job Job name.
    * @return Option containing the number of tasks defined in the provided job. `None`, if the job cannot be found in
    *         this cluster specification.
    */
  def numTasks(job: String): Option[Int] = jobSpecs.get(job).map(_.tasks.size)

  /** Returns a sequence of valid task indices for the provided job.
    *
    * @param  job Job name.
    * @return Option containing the sorted sequence of task indices defined in the provided job. `None`, if the job
    *         cannot be found in this cluster specification.
    */
  def taskIndices(job: String): Option[Seq[Int]] = jobSpecs.get(job).map(_.tasks.keys.toSeq)

  /** Returns the address of the task with index `taskIndex` in the provided job.
    *
    * @param  job       Job name.
    * @param  taskIndex Task index.
    * @return Option containing the network address of the the specified task. `None`, if the provided job cannot be
    *         found in this cluster specification, or if the provided task index cannot be found for that job.
    */
  def taskAddress(job: String, taskIndex: Int): Option[String] = jobSpecs.get(job).flatMap(_.tasks.get(taskIndex))

  /** Constructs and returns a [[ClusterDef]] object that represents this cluster specification.
    *
    * @return Constructed [[ClusterDef]].
    */
  def toClusterDef: ClusterDef = clusterDef

  /** Constructs and returns a [[ClusterDef]] object that represents this cluster specification.
    *
    * @return Constructed [[ClusterDef]].
    */
  override def toProto: GeneratedMessageV3 = toClusterDef
}

/** Contains helper methods for dealing with [[ClusterSpec]]s. */
object ClusterSpec {
  /** Constructs a [[ClusterSpec]] from the provided [[ClusterDef]] serialized representation.
    *
    * @param  clusterDef Protobuf-serialized representation of a [[ClusterSpec]].
    * @return Constructed [[ClusterSpec]]
    */
  def fromClusterDef(clusterDef: ClusterDef): ClusterSpec = {
    ClusterSpec(clusterDef.getJobList.asScala.map(jobDef => {
      (jobDef.getName, JobSpec(jobDef.getTasksMap.asScala.toSeq.map(t => (t._1.intValue(), t._2)): _*))
    }).toMap)
  }
}

/** Job specification (excluding the job name).
  *
  * @param  tasks Mapping from task index to the corresponding task network address.
  */
case class JobSpec(tasks: TreeMap[Int, String])

/** Contains helper methods for dealing with [[JobSpec]]s. */
object JobSpec {
  /** Constructs a [[JobSpec]] by treating the provided sequence of strings as a dense list of network addresses. */
  def apply(tasks: String*): JobSpec = JobSpec(TreeMap(tasks.indices.zip(tasks): _*))

  /** Constructs a [[JobSpec]] using the provided sequence of task index and network address pairs. */
  def apply(tasks: (Int, String)*): JobSpec = JobSpec(TreeMap(tasks: _*))

  /** Constructs a [[JobSpec]] using the provided mapping from task indices to network addresses. */
  def apply(tasks: Map[Int, String]): JobSpec = JobSpec(TreeMap(tasks.toSeq: _*))
}
