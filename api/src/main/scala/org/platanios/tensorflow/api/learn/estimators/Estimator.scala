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

package org.platanios.tensorflow.api.learn.estimators

import org.platanios.tensorflow.api.config._
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.core.distributed.ReplicaDevicePlacer
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.io.{Data, Dataset}
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

import scala.collection.mutable

/** Estimator class to train and evaluate TensorFlow models.
  *
  * The [[Estimator]] class wraps a model which is specified by a `modelFunction`, which, given inputs and a number of
  * other parameters, creates the ops necessary to perform training, evaluation, or predictions, and provides an
  * interface for doing so.
  *
  * All outputs (checkpoints, event files, etc.) are written to a working directory, provided by `configurationBase`, or
  * a subdirectory thereof. If a working directory is not set in `configurationBase`, a temporary directory is used.
  *
  * The `configurationBase` argument can be passed a [[Configuration]] object containing information about the execution
  * environment. It is passed on to the `modelFunction`, if the `modelFunction` has an argument with [[Configuration]]
  * type (and input functions in the same manner). If the `configurationBase` argument is not passed, it is instantiated
  * by the [[Estimator]]. Not passing a configuration means that defaults useful for local execution are used. The
  * [[Estimator]] class makes the configuration available to the model (for instance, to allow specialization based on
  * the number of workers available), and also uses some of its fields to control internals, especially regarding saving
  * checkpoints while training.
  *
  * For models that have hyper-parameters it is recommended to incorporate them in `modelFunction` before instantiating
  * an estimator. This is in contrast to the TensorFlow Python API, but the reason behind the divergence is that the
  * estimator class never uses the provided hyper-parameters. The recommended way to deal with hyper-parameters in the
  * Scala API is to create a model function with two parameter lists, the first one being the hyper-parameters and the
  * second one being those supported by the model-generating function (i.e., optionally a [[Mode]] and a
  * [[Configuration]]).
  *
  * None of the [[Estimator]] class's methods can be overridden in subclasses. Subclasses should use `modelFunction` to
  * configure the base class, and may add methods implementing specialized functionality.
  *
  * @param  configurationBase Configuration base for this estimator. This allows for setting up distributed training
  *                           environments, for example. Note that this is a *base* for a configuration because the
  *                           estimator might modify it and set some missing fields to appropriate default values, in
  *                           order to obtain its final configuration that can be obtain through its `configuration`
  *                           field.
  *
  * @author Emmanouil Antonios Platanios
  */
class Estimator[IT, IO, ID, IS, I] private[learn] (
    private[this] val configurationBase: Configuration = null) {
  type TrainDataset

  /** Run configuration used for this estimator. */
  val configuration: Configuration = {
    // Process provided run configuration.
    val configuration = {
      if (configurationBase == null) {
        Estimator.logger.info("Using the default run configuration.")
        Configuration()
      } else {
        configurationBase.copy()
      }
    }

    // Process working directory.
    val configurationWithWorkingDir = {
      if (configuration.workingDir == null) {
        val workingDir = Files.createTempDirectory("estimator_working_dir")
        Estimator.logger.info(s"Using a temporary folder as working directory: $workingDir")
        configuration.copy(workingDir = Some(workingDir))
      } else {
        configuration
      }
    }

    // Process session configuration.
    val configurationWithSession = {
      if (configuration.sessionConfig == null) {
        Estimator.logger.info("Using the default session configuration with allowed soft placements.")
        configurationWithWorkingDir.copy(sessionConfig = Some(SessionConfig(allowSoftPlacement = Some(true))))
      } else {
        configurationWithWorkingDir
      }
    }

    configurationWithSession
  }

  /** Device function used by this estimator for managing replica device placement when using distributed training. */
  val deviceFunction: Option[(OpSpecification) => String] = Estimator.getReplicaDeviceSetter(configuration).map(_.apply)

  /** Working directory used by this estimator, used to save model parameters, graph, etc. It can also be used to load
    * checkpoints for a previously saved model. */
  def workingDir: Option[Path] = configuration.workingDir

  /** Session configuration used by this estimator. */
  def sessionConfig: Option[SessionConfig] = configuration.sessionConfig

  /** Checkpoint configuration used by this estimator. */
  def checkpointConfig: CheckpointConfig = configuration.checkpointConfig

  /** Summary configuration used by this estimator. */
  def summaryConfig: SummaryConfig = configuration.summaryConfig

  /** Frequency, in number of steps, that this estimator will log the global step / sec rate during training. */
  def globalStepRateLoggingFrequency: Int = configuration.globalStepRateLoggingFrequency

  /** Random seed value to be used by the TensorFlow initializers in this estimator. */
  def randomSeed: Int = configuration.randomSeed

  /** Gets an existing saver from the current graph, or creates a new one if none exists. */
  protected def getOrCreateSaver(): Saver = {
    val graph = Op.currentGraph
    val savers = graph.getCollection(Graph.Keys.SAVERS)
    if (savers.isEmpty) {
      val saver = Saver(
        sharded = true,
        maxToKeep = configuration.checkpointConfig.maxCheckpointsToKeep,
        keepCheckpointEveryNHours = configuration.checkpointConfig.keepCheckpointEveryNHours,
        saveRelativePaths = true)
      graph.addToCollection(saver, Graph.Keys.SAVERS)
      saver
    } else {
      if (savers.size > 1)
        throw InvalidArgumentException("The graph should only contain one saver in the 'SAVERS' collection.")
      savers.head
    }
  }
}

object Estimator {
  private[estimators] val logger = Logger(LoggerFactory.getLogger("Learn / Estimator"))

  def apply[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      modelFunction: SupervisedEstimator.ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T],
      configurationBase: Configuration = null): SupervisedEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SupervisedEstimator(modelFunction, configurationBase)
  }

  /** Creates a replica device setter, if required, to be used as a default device function.
    *
    * Estimators use a [[ReplicaDevicePlacer]] as a default device placer. It sets the distributed related arguments
    * such as the number of parameter server replicas based on the provided run configuration.
    *
    * @param  configuration Configuration.
    * @return Constructed replica device placer.
    */
  def getReplicaDeviceSetter(configuration: Configuration): Option[ReplicaDevicePlacer] = {
    if (configuration.numParameterServers > 0) {
      Some(ReplicaDevicePlacer(
        psNumTasks = configuration.numParameterServers,
        workerDevice = s"/job:${configuration.taskType}/task:${configuration.taskIndex}",
        clusterConfig = configuration.clusterConfig.orNull,
        psOpTypes = Set(
          "Variable", "VariableV2", "AutoReloadVariable", "MutableHashTable", "MutableHashTableV2",
          "MutableHashTableOfTensors", "MutableHashTableOfTensorsV2", "MutableDenseHashTable",
          "MutableDenseHashTableV2")))
    } else {
      None
    }
  }

  /** Creates a [[MonitoredSession]] to be used for training.
    *
    * For a chief, this utility sets proper session initializers, savers, and restorers. It also creates hooks related
    * to checkpoint and summary saving. For workers, this utility method sets the proper session creator which waits for
    * the chief to initialize or restore the session. Please refer to [[MonitoredSession]] for more information.
    *
    * '''NOTE:''' If you provide any summary saver or checkpoint saver hooks in `hooks` or `chiefOnlyHooks`, then the
    * checkpoint configuration in `configuration` will be ignored for the chief and those hooks will be used instead.
    *
    * @param  configuration   Configuration to use for this session. Contains information related to the session
    *                         configuration, the cluster configuration, etc.
    * @param  hooks           Hooks to use while training.
    * @param  chiefOnlyHooks  Hooks to use for the chief. These will only be used if `isChief` is `true`.
    * @param  sessionScaffold Session scaffold used for gathering and/or building supportive ops. If not specified, a
    *                         default one is created. The session scaffold is used to finalize the graph.
    * @return Created monitored session.
    */
  def monitoredTrainingSession(
      configuration: Configuration = Configuration(),
      hooks: Seq[Hook] = Seq.empty,
      chiefOnlyHooks: Seq[Hook] = Seq.empty,
      sessionScaffold: SessionScaffold = SessionScaffold()): MonitoredSession = {
    if (!configuration.isChief) {
      val sessionCreator = WorkerSessionCreator(configuration.master, sessionScaffold, configuration.sessionConfig)
      MonitoredSession(sessionCreator, hooks)
    } else {
      val sessionCreator = ChiefSessionCreator(
        configuration.master, sessionScaffold, configuration.sessionConfig, configuration.workingDir)
      val chiefHooks = mutable.ListBuffer(hooks ++ chiefOnlyHooks: _*)
      configuration.workingDir.foreach(workingDir => {
        chiefHooks += StepRateHook(log = false, summaryDirectory = workingDir)
        if (!chiefHooks.exists(_.isInstanceOf[SummarySaverHook])) {
          configuration.summaryConfig match {
            case NoSummaries => ()
            case StepBasedSummaries(steps) => chiefHooks += SummarySaverHook(workingDir, StepHookTrigger(steps))
            case TimeBasedSummaries(seconds) => chiefHooks += SummarySaverHook(workingDir, TimeHookTrigger(seconds))
          }
        }
        if (!chiefHooks.exists(_.isInstanceOf[CheckpointSaverHook])) {
          configuration.checkpointConfig match {
            case NoCheckpoints => ()
            case StepBasedCheckpoints(steps, _, _) =>
              chiefHooks += CheckpointSaverHook(workingDir, StepHookTrigger(steps))
            case TimeBasedCheckpoints(seconds, _, _) =>
              chiefHooks += CheckpointSaverHook(workingDir, TimeHookTrigger(seconds))
          }
        }
      })
      MonitoredSession(sessionCreator, chiefHooks)
    }
  }

  trait SupportedInferInput[InferInput, InferOutput, T, O, D, S, ModelInferenceOutput] {
    def toDataset(value: InferInput): Dataset[T, O, D, S]
    def convertFetched(iterator: Iterator[(T, ModelInferenceOutput)]): InferOutput
  }

  object SupportedInferInput {
    implicit def datasetInferInput[T, O, D, S, I](implicit
        ev: Data.Aux[T, O, D, S]
    ): SupportedInferInput[Dataset[T, O, D, S], Iterator[(T, I)], T, O, D, S, I] = {
      new SupportedInferInput[Dataset[T, O, D, S], Iterator[(T, I)], T, O, D, S, I] {
        override def toDataset(value: Dataset[T, O, D, S]): Dataset[T, O, D, S] = value
        override def convertFetched(iterator: Iterator[(T, I)]): Iterator[(T, I)] = iterator
      }
    }

    implicit def singleValueInferInput[T, O, D, S, I](implicit
        ev: Data.Aux[T, O, D, S]
    ): SupportedInferInput[T, I, T, O, D, S, I] = new SupportedInferInput[T, I, T, O, D, S, I] {
      override def toDataset(value: T): Dataset[T, O, D, S] = Dataset.from[T, O, D, S](value)
      override def convertFetched(iterator: Iterator[(T, I)]): I = iterator.next()._2
    }
  }
}
