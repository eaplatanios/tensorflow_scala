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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.config._
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.learn.Configuration._
import io.circe._
import io.circe.parser._
import java.nio.file.Path

import org.platanios.tensorflow.api.learn.hooks.{HookTrigger, StepHookTrigger}

/** Configuration for models in the learn API, to be used by estimators.
  *
  * If `clusterConfig` is not provided, then all distributed training related properties are set based on the
  * `TF_CONFIG` environment variable, if the pertinent information is present. The `TF_CONFIG` environment variable is a
  * JSON object with attributes: `cluster` and `task`.
  *
  * `cluster` is a JSON serialized version of [[ClusterConfig]], mapping task types (usually one of the instances of
  * [[TaskType]]) to a list of task addresses.
  *
  * `task` has two attributes: `type` and `index`, where `type` can be any of the task types in `cluster`. When
  * `TF_CONFIG` contains said information, the following properties are set on this class:
  *
  *   - `clusterConfig` is parsed from `TF_CONFIG['cluster']`. Defaults to `None`. If present, it must have one and only
  *     one node for the `chief` job (i.e., `CHIEF` task type).
  *   - `taskType` is set to `TF_CONFIG['task']['type']`. Must be set if `clusterConfig` is present; must be `worker`
  *     (the default value), if it is not.
  *   - `taskIndex` is set to `TF_CONFIG['task']['index']`. Must be set if `clusterConfig` is present; must be 0 (the
  *     default value), if it is not.
  *   - `master` is determined by looking up `taskType` and `taskIndex` in the `clusterConfig`. Defaults to `""`.
  *   - `numParameterServers` is set by counting the number of nodes listed in the `ps` job (i.e., `PARAMETER_SERVER`
  *     task type) of `clusterConfig`. Defaults to 0.
  *   - `numWorkers` is set by counting the number of nodes listed in the `worker` and `chief` jobs (i.e., `WORKER` and
  *     `CHIEF` task types) of `clusterConfig`. Defaults to 1.
  *   - `isChief` is determined based on `taskType` and `TF_CONFIG['cluster']`.
  *
  * There is a special node with `taskType` set as `EVALUATOR`, which is not part of the (training) `clusterConfig`. It
  * handles the distributed evaluation job.
  *
  * Example for a non-chief node:
  * {{{
  *   // The TF_CONFIG environment variable contains:
  *   // {
  *   //   "cluster": {
  *   //     "chief": ["host0:2222"],
  *   //     "ps": ["host1:2222", "host2:2222"],
  *   //     "worker": ["host3:2222", "host4:2222", "host5:2222"]}
  *   //   "task": {
  *   //     "type": "worker",
  *   //     "index": 1}}
  *   // }
  *   val config = Configuration()
  *   assert(config.clusterConfig == Some(ClusterConfig(Map(
  *     "chief" -> JobConfig.fromAddresses("host0:2222"),
  *     "ps" -> JobConfig.fromAddresses("host1:2222", "host2:2222"),
  *     "worker" -> JobConfig.fromAddresses("host3:2222", "host4:2222", "host5:2222")))))
  *   assert(config.taskType == "worker")
  *   assert(config.taskIndex == 1)
  *   assert(config.master == "host4:2222")
  *   assert(config.numParameterServers == 2)
  *   assert(config.numWorkers == 4)
  *   assert(!config.isChief)
  * }}}
  *
  * Example for a chief node:
  * {{{
  *   // The TF_CONFIG environment variable contains:
  *   // {
  *   //   "cluster": {
  *   //     "chief": ["host0:2222"],
  *   //     "ps": ["host1:2222", "host2:2222"],
  *   //     "worker": ["host3:2222", "host4:2222", "host5:2222"]}
  *   //   "task": {
  *   //     "type": "chief",
  *   //     "index": 0}}
  *   // }
  *   val config = Configuration()
  *   assert(config.clusterConfig == Some(ClusterConfig(Map(
  *     "chief" -> JobConfig.fromAddresses("host0:2222"),
  *     "ps" -> JobConfig.fromAddresses("host1:2222", "host2:2222"),
  *     "worker" -> JobConfig.fromAddresses("host3:2222", "host4:2222", "host5:2222")))))
  *   assert(config.taskType == "chief")
  *   assert(config.taskIndex == 0)
  *   assert(config.master == "host0:2222")
  *   assert(config.numParameterServers == 2)
  *   assert(config.numWorkers == 4)
  *   assert(config.isChief)
  * }}}
  *
  * Example for an evaluator node (an evaluator is not part of the training cluster):
  * {{{
  *   // The TF_CONFIG environment variable contains:
  *   // {
  *   //   "cluster": {
  *   //     "chief": ["host0:2222"],
  *   //     "ps": ["host1:2222", "host2:2222"],
  *   //     "worker": ["host3:2222", "host4:2222", "host5:2222"]}
  *   //   "task": {
  *   //     "type": "evaluator",
  *   //     "index": 0}}
  *   // }
  *   val config = Configuration()
  *   assert(config.clusterConfig == None)
  *   assert(config.taskType == "evaluator")
  *   assert(config.taskIndex == 0)
  *   assert(config.master == "")
  *   assert(config.numParameterServers == 0)
  *   assert(config.numWorkers == 0)
  *   assert(!config.isChief)
  * }}}
  *
  * '''NOTE:''' If a `checkpointConfig` is set, `maxCheckpointsToKeep` might need to be adjusted accordingly, especially
  * in distributed training. For example, using `TimeBasedCheckpoints(60)` without adjusting `maxCheckpointsToKeep`
  * (which defaults to 5) leads to a situation that checkpoints would be garbage collected after 5 minutes. In
  * distributed training, the evaluation job starts asynchronously and might fail to load or find the checkpoints due to
  * a race condition.
  *
  * @param  workingDir       Directory used to save model parameters, graph, etc. It can also be used to load
  *                          checkpoints for a previously saved model. If `null`, a temporary directory will be used.
  * @param  sessionConfig    Configuration to use for the created sessions.
  * @param  checkpointConfig Configuration specifying when to save checkpoints.
  * @param  summaryConfig    Configuration specifying when to save summaries.
  * @param  randomSeed       Random seed value to be used by the TensorFlow initializers. Setting this value allows
  *                          consistency between re-runs.
  * @author Emmanouil Antonios Platanios
  */
case class Configuration(
    workingDir: Option[Path] = None,
    logTrigger: HookTrigger = StepHookTrigger(100),
    // TODO: [LEARN] Allow a cluster configuration to be directly provided here.
    sessionConfig: Option[SessionConfig] = None,
    checkpointConfig: CheckpointConfig = TimeBasedCheckpoints(600, 5, 10000),
    summaryConfig: SummaryConfig = StepBasedSummaries(100),
    randomSeed: Int = 1
) {
  val (clusterConfig, taskType, taskIndex, master, numParameterServers, numWorkers, isChief): (
      Option[ClusterConfig], String, Int, String, Int, Int, Boolean) = {
    val tfConfigJson = System.getenv(TF_CONFIG_ENV)
    val tfConfigJsonParsed = parse(tfConfigJson)

    // Parse the cluster configuration.
    val tfClusterConfig: Either[Exception, ClusterConfig] = tfConfigJsonParsed.right.flatMap(parsed => {
      val clusterConfigJson = parsed.findAllByKey(CLUSTER_KEY)
      if (clusterConfigJson.isEmpty) {
        Left(null)
      } else if (clusterConfigJson.length > 1) {
        throw InvalidArgumentException(
          s"Only a single 'cluster' configuration field should be provided in $TF_CONFIG_ENV.")
      } else {
          clusterConfigJson.head.as[Map[String, Json]].right.map(_.toSeq.flatMap(p => {
          p._2.as[Seq[String]] match {
            case Left(_) => p._2.as[Map[Int, String]] match {
              case Left(_) => throw InvalidArgumentException(
                s"Could not parse the cluster configuration in $TF_CONFIG_ENV.")
              case Right(tasks) => Map(p._1 -> JobConfig.fromMap(tasks))
            }
            case Right(tasks) => Map(p._1 -> JobConfig.fromSeq(tasks))
          }
        })).right.map(m => ClusterConfig(Map(m: _*)))
      }
    })

    // Parse the task configuration.
    val tfTaskConfig: Either[Exception, (String, Int)] = tfConfigJsonParsed.right.flatMap(parsed => {
      val cursor = parsed.hcursor
      val taskType = cursor.downField(TASK_ENV_KEY).get[String](TASK_TYPE_KEY)
      val taskIndex = cursor.downField(TASK_ENV_KEY).get[Int](TASK_ID_KEY)
      for (t <- taskType.right; i <- taskIndex.right) yield (t, i)
    })

    if (tfClusterConfig.isRight) {
      // Distributed mode.
      val config = tfClusterConfig.right.get
      config.jobTasks(CHIEF.name) match {
        case None => throw InvalidArgumentException(
          s"If 'cluster' is set in $TF_CONFIG_ENV, it must have one 'chief' node.")
        case Some(_) =>
          if (config.jobTasks(CHIEF.name).get.size > 1)
            throw InvalidArgumentException(s"The 'cluster' in $TF_CONFIG_ENV must have only one 'chief' node.")

          val (taskType, taskIndex) = tfTaskConfig match {
            case Left(exception) => throw InvalidArgumentException(
              s"If 'cluster' is set in $TF_CONFIG_ENV, task type and index must be set too.", exception)
            case Right(taskConfig) => taskConfig
          }

          // Check the task index bounds. An upper bound is not necessary as:
          // - for evaluator tasks there is no upper bound.
          // - for non-evaluator tasks, the task index is upper bounded by the number of jobs in the cluster
          //   configuration, which will be checked later (while retrieving the `master`).
          if (taskIndex < 0)
            throw InvalidArgumentException("The task index must be a non-negative number.")

          taskType match {
            case EVALUATOR.name =>
              // Evaluator is not part of the training cluster.
              val clusterConfig = None
              val master = LOCAL_MASTER
              val numParameterServers = 0
              val numWorkers = 0
              val isChief = false
              (clusterConfig, taskType, taskIndex, master, numParameterServers, numWorkers, isChief)
            case _ =>
              val clusterConfig = Some(config)
              val master = getNetworkAddress(config, taskType, taskIndex)
              val numParameterServers = countParameterServers(config)
              val numWorkers = countWorkers(config)
              val isChief = taskType == CHIEF.name
              (clusterConfig, taskType, taskIndex, master, numParameterServers, numWorkers, isChief)
          }
      }
    } else {
      // Local mode.
      val (taskType, taskIndex) = tfTaskConfig match {
        case Left(_) => (WORKER.name, 0)
        case Right(taskConfig) => taskConfig
      }

      if (taskType != WORKER.name)
        throw InvalidArgumentException(s"If 'cluster' is not set in $TF_CONFIG_ENV, task type must be ${WORKER.name}.")
      if (taskIndex != 0)
        throw InvalidArgumentException(s"If 'cluster' is not set in $TF_CONFIG_ENV, task index must be 0.")

      val clusterConfig = None
      val master = ""
      val numParameterServers = 0
      val numWorkers = 1
      val isChief = true
      (clusterConfig, taskType, taskIndex, master, numParameterServers, numWorkers, isChief)
    }
  }

  val evaluationMaster: String = ""
}

/** Contains helper methods for dealing with [[Configuration]]s. */
object Configuration {
  private[learn] val TF_CONFIG_ENV: String = "TF_CONFIG"
  private[learn] val TASK_ENV_KEY : String = "task"
  private[learn] val TASK_TYPE_KEY: String = "type"
  private[learn] val TASK_ID_KEY  : String = "index"
  private[learn] val CLUSTER_KEY  : String = "cluster"
  private[learn] val LOCAL_MASTER : String = ""
  private[learn] val GRPC_SCHEME  : String = "grpc://"

  /** Returns the appropriate network address for the specified task.
    *
    * @param  clusterConfig Cluster configuration.
    * @param  taskType      Task type.
    * @param  taskIndex     Task index.
    * @return Network address for the specified task.
    * @throws InvalidArgumentException If the provided task type or index cannot be found in the cluster configuration.
    */
  @throws[InvalidArgumentException]
  private[Configuration] def getNetworkAddress(
      clusterConfig: ClusterConfig, taskType: String, taskIndex: Int): String = {
    require(
      clusterConfig.jobs.contains(taskType),
      s"'$taskType' is not a valid job name in the provided cluster configuration: $clusterConfig\n\n" +
          s"Note that these values may be coming from the $TF_CONFIG_ENV environment variable.")
    val jobTasks = clusterConfig.jobTasks(taskType).get
    require(
      jobTasks.contains(taskIndex),
      s"'$taskIndex' is not a valid task index for job '$taskType' " +
          s"in the provided cluster configuration: $clusterConfig\n\n" +
          s"Note that these values may be coming from the $TF_CONFIG_ENV environment variable.")
    GRPC_SCHEME + jobTasks(taskIndex)
  }

  /** Counts the number of parameter servers in the provided cluster specification. */
  private[Configuration] def countParameterServers(clusterSpec: ClusterConfig): Int = {
    clusterSpec.jobs.count(_ == PARAMETER_SERVER.name)
  }

  /** Counts the number of workers (including the chief) in the provided cluster specification. */
  private[Configuration] def countWorkers(clusterSpec: ClusterConfig): Int = {
    clusterSpec.jobs.count(j => j == WORKER.name || j == CHIEF.name)
  }

  sealed trait TaskType {
    val name: String
  }

  object TaskType {
    @throws[InvalidArgumentException]
    def fromName(name: String): TaskType = name match {
      case MASTER.name => MASTER
      case PARAMETER_SERVER.name => PARAMETER_SERVER
      case WORKER.name => WORKER
      case CHIEF.name => CHIEF
      case EVALUATOR.name => EVALUATOR
      case _ => throw InvalidArgumentException(s"Unsupported task type '$name' provided.")
    }
  }

  case object MASTER extends TaskType {
    override val name: String = "master"
  }

  case object PARAMETER_SERVER extends TaskType {
    override val name: String = "ps"
  }

  case object WORKER extends TaskType {
    override val name: String = "worker"
  }

  /** Chiefs are identical to workers, except that they also handle saving checkpoints. */
  case object CHIEF extends TaskType {
    override val name: String = "chief"
  }

  case object EVALUATOR extends TaskType {
    override val name: String = "evaluator"
  }
}
