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
import org.platanios.tensorflow.api.core.client.{Fetchable, SessionConfig}
import org.platanios.tensorflow.api.core.distributed.ReplicaDevicePlacer
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.implicits.helpers.OutputToTensor
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset, TensorDataset}
import org.platanios.tensorflow.api.ops.{Function, Op, OpSpecification, Output}
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.io.events.SummaryFileWriterCache
import org.platanios.tensorflow.api.types.FLOAT32

import com.typesafe.scalalogging.Logger
import org.tensorflow.framework.Summary
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

import scala.collection.mutable

/** Abstract class for estimators which are used to train, use, and evaluate TensorFlow models.
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
abstract class Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] private[estimators] (
    protected val modelFunction: Estimator.ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, EI],
    protected val configurationBase: Configuration = null
) {
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

  /** Random seed value to be used by the TensorFlow initializers in this estimator. */
  def randomSeed: Option[Int] = configuration.randomSeed

  /** Gets an existing saver from the current graph, or creates a new one if none exists. */
  protected def getOrCreateSaver(): Option[Saver] = {
    val graph = Op.currentGraph
    val savers = graph.getCollection(Graph.Keys.SAVERS)
    if (savers.isEmpty) {
      val saver = Saver(
        sharded = true,
        maxToKeep = configuration.checkpointConfig.maxCheckpointsToKeep,
        keepCheckpointEveryNHours = configuration.checkpointConfig.keepCheckpointEveryNHours,
        saveRelativePaths = true)
      graph.addToCollection(saver, Graph.Keys.SAVERS)
      Some(saver)
    } else {
      if (savers.size > 1)
        throw InvalidArgumentException("The graph should only contain one saver in the 'SAVERS' collection.")
      savers.headOption
    }
  }

  /** Trains the model managed by this estimator.
    *
    * @param  data         Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                      supervision labels).
    * @param  stopCriteria Stop criteria to use for stopping the training iteration. For the default criteria please
    *                      refer to the documentation of [[StopCriteria]].
    */
  def train(data: () => Dataset[TT, TO, TD, TS], stopCriteria: StopCriteria = StopCriteria()): Unit

  /** Infers output (i.e., computes predictions) for `input` using the model managed by this estimator.
    *
    * `input` can be of one of the following types:
    *
    *   - A [[Dataset]], in which case this method returns an iterator over `(input, output)` tuples corresponding to
    *     each element in the dataset. Note that the predictions are computed lazily in this case, whenever an element
    *     is requested from the returned iterator.
    *   - A single input of type `IT`, in which case this method returns a prediction of type `I`.
    *
    * Note that, `ModelInferenceOutput` refers to the tensor type that corresponds to the symbolic type `I`. For
    * example, if `I` is `(Output, Output)`, then `ModelInferenceOutput` will be `(Tensor, Tensor)`.
    *
    * @param  input Input for the predictions.
    * @return Either an iterator over `(IT, ModelInferenceOutput)` tuples, or a single element of type `I`, depending on
    *         the type of `input`.
    */
  def infer[InferInput, InferOutput, ModelInferenceOutput](
      input: () => InferInput
  )(implicit
      evFetchableIO: Fetchable.Aux[IO, IT],
      evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
      evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
      ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): InferOutput

  /** Evaluates the model managed by this estimator given the provided evaluation data, `data`.
    *
    * The evaluation process is iterative. In each step, a data batch is obtained from `data` and internal metric value
    * accumulators are updated. The number of steps to perform is controlled through the `maxSteps` argument. If set to
    * `-1`, then all batches from `data` will be processed.
    *
    * @param  data           Evaluation dataset. Each element is a tuple over input and training inputs (i.e.,
    *                        supervision labels).
    * @param  metrics        Evaluation metrics to use.
    * @param  maxSteps       Maximum number of evaluation steps to perform. If `-1`, the evaluation process will run
    *                        until `data` is exhausted.
    * @param  saveSummaries  Boolean indicator specifying whether to save the evaluation results as summaries in the
    *                        working directory of this estimator.
    * @param  name           Name for this evaluation. If provided, it will be used to generate an appropriate directory
    *                        name for the resulting summaries. If `saveSummaries` is `false`, this argument has no
    *                        effect. This is useful if the user needs to run multiple evaluations on different data
    *                        sets, such as on training data vs test data. Metrics for different evaluations are saved in
    *                        separate folders, and appear separately in TensorBoard.
    * @return Evaluation metric values at the end of the evaluation process. The return sequence matches the ordering of
    *         `metrics`.
    * @throws InvalidArgumentException If `saveSummaries` is `true`, but the estimator has no working directory
    *                                  specified.
    */
  @throws[InvalidArgumentException]
  def evaluate(
      data: () => Dataset[TT, TO, TD, TS],
      metrics: Seq[Metric[EI, Output]],
      maxSteps: Long = -1L,
      saveSummaries: Boolean = true,
      name: String = null): Seq[Tensor]

  protected def saveEvaluationSummaries(
      step: Long, metrics: Seq[Metric[EI, Output]], metricValues: Seq[Tensor], name: String = null): Unit = {
    // Setup the output directory.
    val evaluationDir = workingDir.map(_.resolve(if (name != null) s"eval_$name" else "eval"))
    if (evaluationDir.isEmpty)
      throw InvalidArgumentException(
        "No working directory is provided and thus evaluation summaries cannot be saved.")
    val summaryProto = Summary.newBuilder()
    metrics.zip(metricValues).foreach {
      case (metric, metricValue) =>
        if (metricValue.shape.rank == 0 &&
            (metricValue.dataType.isFloatingPoint || metricValue.dataType.isInteger)) {
          val castedValue = metricValue.cast(FLOAT32).scalar.asInstanceOf[Float]
          val value = Summary.Value.newBuilder()
          value.setTag(metric.name)
          value.setSimpleValue(castedValue)
          summaryProto.addValue(value)
        } else {
          Estimator.logger.warn(
            s"Skipping summary for non-scalar and/or non-floating-point/non-integer metric '$metric'.")
        }
    }
    evaluationDir.map(SummaryFileWriterCache.get(_)).foreach(writer => {
      writer.writeSummary(summaryProto.build(), step)
      writer.flush()
    })
  }
}

object Estimator {
  private[estimators] val logger = Logger(LoggerFactory.getLogger("Learn / Estimator"))

  class ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, EI](
    val function: (Configuration) => TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]) {
    def apply(configuration: Configuration): TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = {
      function(configuration)
    }
  }

  case class UnsupervisedModelFunction[IT, IO, ID, IS, I](
      override val function: (Configuration) => UnsupervisedTrainableModel[IT, IO, ID, IS, I]
  ) extends ModelFunction[IT, IO, ID, IS, I, IT, IO, ID, IS, I](function) {
    override def apply(configuration: Configuration): UnsupervisedTrainableModel[IT, IO, ID, IS, I] = {
      function(configuration)
    }
  }

  case class SupervisedModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      override val function: (Configuration) => SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
  ) extends ModelFunction[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, T)](function) {
    override def apply(configuration: Configuration): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      function(configuration)
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
      hooks: Set[Hook] = Set.empty,
      chiefOnlyHooks: Set[Hook] = Set.empty,
      sessionScaffold: SessionScaffold = SessionScaffold()): MonitoredSession = {
    if (!configuration.isChief) {
      val sessionCreator = WorkerSessionCreator(configuration.master, sessionScaffold, configuration.sessionConfig)
      MonitoredSession(sessionCreator, hooks)
    } else {
      val sessionCreator = ChiefSessionCreator(
        configuration.master, sessionScaffold, configuration.sessionConfig, configuration.workingDir)
      val chiefHooks = mutable.Set((hooks ++ chiefOnlyHooks).toSeq: _*)
      configuration.workingDir.foreach(workingDir => {
        if (!chiefHooks.exists(_.isInstanceOf[CheckpointSaver])) {
          configuration.checkpointConfig match {
            case NoCheckpoints => ()
            case StepBasedCheckpoints(steps, _, _) =>
              chiefHooks += CheckpointSaver(workingDir, StepHookTrigger(steps))
            case TimeBasedCheckpoints(seconds, _, _) =>
              chiefHooks += CheckpointSaver(workingDir, TimeHookTrigger(seconds))
          }
        }
      })
      MonitoredSession(sessionCreator, chiefHooks.toSet)
    }
  }

  trait SupportedInferInput[InferInput, InferOutput, T, O, D, S, ModelInferenceOutput] {
    def toDataset(value: InferInput): Dataset[T, O, D, S]
    def convertFetched(iterator: Iterator[(T, ModelInferenceOutput)]): InferOutput
  }

  object SupportedInferInput {
    implicit def datasetInferInput[T, O, D, S, I](implicit
        evOToT: OutputToTensor.Aux[O, T],
        ev: Data.Aux[T, O, D, S],
        evFunctionInput: Function.ArgType[O]
    ): SupportedInferInput[Dataset[T, O, D, S], Iterator[(T, I)], T, O, D, S, I] = {
      new SupportedInferInput[Dataset[T, O, D, S], Iterator[(T, I)], T, O, D, S, I] {
        override def toDataset(value: Dataset[T, O, D, S]): Dataset[T, O, D, S] = value
        override def convertFetched(iterator: Iterator[(T, I)]): Iterator[(T, I)] = iterator
      }
    }

    implicit def singleValueInferInput[T, O, D, S, I](implicit
        evOToT: OutputToTensor.Aux[O, T],
        ev: Data.Aux[T, O, D, S],
        evFunctionInput: Function.ArgType[O]
    ): SupportedInferInput[T, I, T, O, D, S, I] = new SupportedInferInput[T, I, T, O, D, S, I] {
      override def toDataset(value: T): Dataset[T, O, D, S] = TensorDataset[T, O, D, S](value)
      override def convertFetched(iterator: Iterator[(T, I)]): I = iterator.next()._2
    }
  }
}
