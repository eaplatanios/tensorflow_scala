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

import org.platanios.tensorflow.api.config.{CheckpointConfig, SummaryConfig}
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.learn.Estimator.ModelFunction
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.learn.utilities.ReplicaDevicePlacer
import org.platanios.tensorflow.api.ops.{ControlFlow, Op, OpSpecification}
import org.platanios.tensorflow.api.ops.io.Dataset
import org.platanios.tensorflow.api.ops.variables.Saver

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

import scala.collection.immutable.TreeMap
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
  * second one being those supported by [[ModelFunction]] (i.e., optionally a [[Mode]] and a [[Configuration]]).
  *
  * None of the [[Estimator]] class's methods can be overridden in subclasses. Subclasses should use `modelFunction` to
  * configure the base class, and may add methods implementing specialized functionality.
  *
  * @param  modelFunction     Model-generating function that can optionally have a [[Configuration]] argument which will
  *                           be used to pass the estimator's configuration to the model and allows customizing the
  *                           model based on the execution environment.
  * @param  configurationBase Configuration base for this estimator. This allows for setting up distributed training
  *                           environments, for example. Note that this is a *base* for a configuration because the
  *                           estimator might modify it and set some missing fields to appropriate default values, in
  *                           order to obtain its final configuration that can be obtain through its `configuration`
  *                           field.
  *
  * @author Emmanouil Antonios Platanios
  */
class Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
    val modelFunction: Estimator.ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T],
    private[this] val configurationBase: Configuration = null) {
  /** Run configuration used for this estimator. */
  val configuration: Configuration = {
    // Process provided run configuration.
    val configuration = {
      if (configurationBase == null) {
        Estimator.logger.info("Using default run configuration.")
        Configuration()
      } else {
        configurationBase.copy()
      }
    }

    // Process working directory.
    val configurationWithWorkingDir = {
      if (configuration.workingDir == null) {
        val workingDir = Files.createTempDirectory("estimator_working_dir")
        Estimator.logger.info(s"Using temporary folder as working directory: $workingDir")
        configuration.copy(workingDir = Some(workingDir))
      } else {
        configuration
      }
    }

    // Process session configuration.
    val configurationWithSession = {
      if (configuration.sessionConfig == null) {
        Estimator.logger.info("Using default session configuration with allowed soft placements.")
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

  /** Trains the model managed by this estimator.
    *
    * @param  data                Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                             supervision labels).
    * @param  terminationCriteria Termination criteria to use for stopping the training iteration. For the default
    *                             criteria please refer to the documentation of [[StopCriteria]].
    * @param  hooks               Hooks to use while training (e.g., logging for the loss function value, etc.).
    */
  def train(
      data: Dataset[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
      terminationCriteria: StopCriteria = StopCriteria(),
      hooks: Seq[Hook] = Seq.empty): Unit = {
    // TODO: !!! Load global step from a checkpoint and skip training if appropriate.
    val allHooks = mutable.ListBuffer(hooks: _*)
    allHooks += StopHook(terminationCriteria)
    val model = modelFunction(configuration)
    val graph = Graph()
    Op.createWith(graph = graph, deviceFunction = deviceFunction.getOrElse(_.device)) {
      graph.setRandomSeed(randomSeed)
      Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, graph)
      val step = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, graph)
      val trainingOps = model.buildTrainOps()
      graph.addToCollection(trainingOps.loss, Graph.Keys.LOSSES)
      allHooks += TensorNaNHook(Set(trainingOps.loss.name))
      allHooks += TensorLoggingHook(TreeMap(
        "Step" -> step.value.name,
        "Loss" -> trainingOps.loss.name
      ), StepHookTrigger(100))
      if (graph.getCollection(Graph.Keys.SAVERS).isEmpty) {
        graph.addToCollection(Saver(
          sharded = true,
          maxToKeep = configuration.checkpointConfig.maxCheckpointsToKeep,
          keepCheckpointEveryNHours = configuration.checkpointConfig.keepCheckpointEveryNHours,
          saveRelativePaths = true
        ), Graph.Keys.SAVERS)
      }
      // TODO: !!! [HOOKS] [CHECKPOINTS] Add checkpoint saver hook for the chief.
      val session = Estimator.monitoredTrainingSession(
        configuration = configuration,
        hooks = allHooks,
        chiefOnlyHooks = Seq.empty,
        sessionScaffold = SessionScaffold(
          initOp = Some(ControlFlow.group(Set(
            trainingOps.input.createInitializer(data),
            graph.globalVariablesInitializer())
          ))))
      try {
        while (!session.shouldStop)
          session.run(targets = trainingOps.trainOp)
      } catch {
        case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) => session.close()
        case e: Throwable =>
          session.closeWithoutHookEnd()
          throw e
      }
    }
  }
}

object Estimator {
  private[Estimator] val logger = Logger(LoggerFactory.getLogger("Learn / Estimator"))

  case class ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      function: (Configuration) => TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]) {
    def apply(configuration: Configuration): TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      function(configuration)
    }
  }

  trait Implicits {
    implicit def modelToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
        model: TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    ): ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      ModelFunction((_: Configuration) => model)
    }

    implicit def unitFunctionToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
        function: () => TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    ): ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      ModelFunction((_: Configuration) => function())
    }

    implicit def unaryRunConfigFunctionToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
        function: (Configuration) => TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    ): ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      ModelFunction(function)
    }
  }

  /** Creates a replica device setter, if required, to be used as a default device function.
    *
    * Estimators use a [[ReplicaDevicePlacer]] as a default device placer. It sets the distributed related arguments
    * such as the number of parameter server replicas based on the provided run configuration.
    *
    * @param  configuration Configuration.
    * @return Constructed replica device placer.
    */
  private[Estimator] def getReplicaDeviceSetter(configuration: Configuration): Option[ReplicaDevicePlacer] = {
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
    * @param  configuration   Configuration to use for this session. Contains information related to the session
    *                         configuration, the cluster configuration, etc.
    * @param  hooks           Hooks to use while training.
    * @param  chiefOnlyHooks  Hooks to use for the chief. These will only be used if `isChief` is `true`.
    * @param  sessionScaffold Session scaffold used for gathering and/or building supportive ops. If not specified, a
    *                         default one is created. The session scaffold is used to finalize the graph.
    * @return Created monitored session.
    */
  private[Estimator] def monitoredTrainingSession(
      configuration: Configuration = Configuration(),
      hooks: Seq[Hook] = Seq.empty,
      chiefOnlyHooks: Seq[Hook] = Seq.empty,
      sessionScaffold: SessionScaffold = SessionScaffold()): MonitoredSession = {
    if (!configuration.isChief) {
      val sessionCreator = WorkerSessionCreator(configuration.master, sessionScaffold, configuration.sessionConfig)
      MonitoredSession(sessionCreator, hooks, stopGracePeriodSeconds = configuration.stopGracePeriodSeconds)
    } else {
      val sessionCreator = ChiefSessionCreator(
        configuration.master, sessionScaffold, configuration.sessionConfig, configuration.workingDir)
      var chiefHooks = hooks ++ chiefOnlyHooks
      if (configuration.workingDir.isDefined) {
        // TODO: !!! [HOOKS] Add step counter hook.
        // TODO: !!! [HOOKS] Add summary hook.
        // TODO: !!! [HOOKS] Add checkpoint hook.
      }
      MonitoredSession(sessionCreator, chiefHooks, stopGracePeriodSeconds = configuration.stopGracePeriodSeconds)
    }
  }
}
