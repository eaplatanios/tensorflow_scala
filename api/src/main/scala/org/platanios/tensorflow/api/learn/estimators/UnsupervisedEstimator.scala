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
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.core.exception.{CheckpointNotFoundException, InvalidArgumentException}
import org.platanios.tensorflow.api.io.{CheckpointReader, SummaryFileWriterCache}
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.io.Dataset
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.Summary

import java.nio.file.Path

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
  * second one being those supported by the model-generating function (i.e., optionally a [[Mode]] and a
  * [[Configuration]]).
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
class UnsupervisedEstimator[IT, IO, ID, IS, I] private[learn] (
    val modelFunction: UnsupervisedEstimator.UnsupervisedModelFunction[IT, IO, ID, IS, I],
    private[this] val configurationBase: Configuration = null
) extends Estimator[IT, IO, ID, IS, I](configurationBase) {
  /** Trains the model managed by this estimator.
    *
    * '''NOTE:''' If you provide any summary saver or checkpoint saver hooks in `hooks` or `chiefOnlyHooks`, then the
    * checkpoint configuration in this estimator's `configuration` will be ignored for the chief and those hooks will be
    * used instead.
    *
    * @param  data                Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                             supervision labels).
    * @param  terminationCriteria Termination criteria to use for stopping the training iteration. For the default
    *                             criteria please refer to the documentation of [[StopCriteria]].
    * @param  hooks               Hooks to use while training (e.g., logging for the loss function value, etc.).
    * @param  chiefOnlyHooks      Hooks to use while training for the chief node only. This argument is only useful for
    *                             a distributed training setting.
    * @param  tensorBoardConfig   If provided, a TensorBoard server is launched using the provided configuration. In
    *                             that case, it is required that TensorBoard is installed for the default Python
    *                             environment in the system. If training in a distributed setting, the TensorBoard
    *                             server is launched on the chief node.
    */
  @throws[InvalidArgumentException]
  def train(
      data: Dataset[IT, IO, ID, IS],
      terminationCriteria: StopCriteria = StopCriteria(),
      hooks: Seq[Hook] = Seq.empty,
      chiefOnlyHooks: Seq[Hook] = Seq.empty,
      tensorBoardConfig: TensorBoardConfig = null): Unit = {
    val needsToTrain = {
      if (!terminationCriteria.restartCounting) {
        workingDir.flatMap(dir => Saver.latestCheckpoint(dir).flatMap(latestPath => {
          CheckpointReader(latestPath).getTensor(Graph.Keys.GLOBAL_STEP.name)
        })).map(_.scalar.asInstanceOf[Long]).flatMap(s => terminationCriteria.maxSteps.map(_ <= s)).getOrElse(true)
      } else {
        true
      }
    }
    if (!needsToTrain) {
      UnsupervisedEstimator.logger.info(
        "Skipping training because no restarting is allowed in the termination criteria and the maximum number of " +
            "steps have already been executed in the past (i.e., saved checkpoint).")
    } else {
      val allHooks = mutable.ListBuffer(hooks: _*)
      val allChiefOnlyHooks = mutable.ListBuffer(chiefOnlyHooks: _*)
      allHooks += StopHook(terminationCriteria)
      val model = modelFunction(configuration)
      val graph = Graph()
      Op.createWith(graph = graph, deviceFunction = deviceFunction.getOrElse(_.device)) {
        graph.setRandomSeed(randomSeed)
        // TODO: [LEARN] !!! Do we ever update the global epoch?
        Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
        val step = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
        val trainingOps = Op.createWithNameScope("Model")(model.buildTrainOps())
        val inputInitializer = trainingOps.inputIterator.createInitializer(data)
        graph.addToCollection(trainingOps.loss, Graph.Keys.LOSSES)
        allHooks += TensorNaNHook(Set(trainingOps.loss.name))
        allHooks += TensorLoggingHook(TreeMap(
          "Step" -> step.value.name,
          "Loss" -> trainingOps.loss.name
        ), StepHookTrigger(100))
        if (tensorBoardConfig != null)
          allChiefOnlyHooks += TensorBoardHook(tensorBoardConfig)
        val saver = getOrCreateSaver()
        val session = Estimator.monitoredTrainingSession(
          configuration = configuration,
          hooks = allHooks,
          chiefOnlyHooks = allChiefOnlyHooks,
          sessionScaffold = SessionScaffold(
            initOp = Some(graph.globalVariablesInitializer()),
            localInitOp = Some(ControlFlow.group(Set(inputInitializer, graph.localVariablesInitializer()))),
            saver = Some(saver)))
        try {
          while (!session.shouldStop)
            session.run(targets = trainingOps.trainOp)
        } catch {
          case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) => session.close()
          case e: Throwable =>
            session.closeWithoutHookEnd()
            throw e
        } finally {
          if (!session.closed)
            session.close()
        }
      }
    }
  }

  /** Infers output (i.e., computes predictions) for `input` using the model managed by this estimator.
    *
    * This method requires that a checkpoint can be found in either `checkpointPath`, if provided, or in this
    * estimator's working directory. It first loads the trained parameter values from the checkpoint specified by
    * `checkpointPath` or from the latest checkpoint found in the working directory, and it then computes predictions
    * for `input`.
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
    * @param  input          Input for the predictions.
    * @param  hooks          Hooks to use while making predictions.
    * @param  checkpointPath Path to a checkpoint file to use. If `null`, then the latest checkpoint found in this
    *                        estimator's working directory will be used.
    * @return Either an iterator over `(IT, ModelInferenceOutput)` tuples, or a single element of type `I`, depending on
    *         the type of `input`.
    * @throws CheckpointNotFoundException If no checkpoint could be found. This can happen if `checkpointPath` is `null`
    *                                     and no checkpoint could be found in this estimator's working directory.
    */
  // TODO: !!! [ESTIMATORS] Add an "infer" method that doesn't need to load a checkpoint (i.e., in-memory).
  @throws[CheckpointNotFoundException]
  def infer[InferInput, InferOutput, ModelInferenceOutput](
      input: InferInput,
      hooks: Seq[Hook] = Seq.empty,
      checkpointPath: Path = null)(implicit
      evFetchableIO: Fetchable.Aux[IO, IT],
      evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
      evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
      ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): InferOutput = {
    // Check that the model has been trained.
    val _checkpointPath = Option(checkpointPath).orElse(workingDir.flatMap(Saver.latestCheckpoint(_)))
    if (_checkpointPath.isEmpty)
      throw CheckpointNotFoundException(
        "No checkpoint was found. Please provide a valid 'workingDir' the estimator configuration, or a path to a " +
            "valid checkpoint file through the 'checkpointPath' argument.")
    val model = modelFunction(configuration)
    val graph = Graph()
    Op.createWith(graph) {
      graph.setRandomSeed(randomSeed)
      Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
      Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      val inferenceOps = Op.createWithNameScope("Model")(model.buildInferenceOps())
      val inputInitializer = inferenceOps.inputIterator.createInitializer(ev.toDataset(input))
      val saver = getOrCreateSaver()
      val session = MonitoredSession(
        ChiefSessionCreator(
          sessionScaffold = SessionScaffold(
            initOp = Some(graph.globalVariablesInitializer()),
            localInitOp = Some(ControlFlow.group(Set(inputInitializer, graph.localVariablesInitializer()))),
            saver = Some(saver)),
          sessionConfig = configuration.sessionConfig,
          checkpointPath = workingDir),
        hooks, shouldRecover = true)
      val output = ev.convertFetched(new Iterator[(IT, ModelInferenceOutput)] {
        override def hasNext: Boolean = session.shouldStop
        override def next(): (IT, ModelInferenceOutput) = {
          try {
            session.run(fetches = (inferenceOps.input, inferenceOps.output))
          } catch {
            case e: Throwable =>
              session.closeWithoutHookEnd()
              throw e
          }
        }
      })
      if (!session.closed)
        session.close()
      output
    }
  }

  /** Evaluates the model managed by this estimator given the provided evaluation data, `data`.
    *
    * This method requires that a checkpoint can be found in either `checkpointPath`, if provided, or in this
    * estimator's working directory. It first loads the trained parameter values from the checkpoint specified by
    * `checkpointPath` or from the latest checkpoint found in the working directory, and it then computes predictions
    * for `input`.
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
    * @param  hooks          Hooks to use while evaluating.
    * @param  checkpointPath Path to a checkpoint file to use. If `null`, then the latest checkpoint found in this
    *                        estimator's working directory will be used.
    * @param  saveSummaries  Boolean indicator specifying whether to save the evaluation results as summaries in the
    *                        working directory of this estimator.
    * @param  name           Name for this evaluation. If provided, it will be used to generate an appropriate directory
    *                        name for the resulting summaries. If `saveSummaries` is `false`, this argument has no
    *                        effect. This is useful if the user needs to run multiple evaluations on different data
    *                        sets, such as on training data vs test data. Metrics for different evaluations are saved in
    *                        separate folders, and appear separately in TensorBoard.
    * @return                Evaluation metric values at the end of the evaluation process. The return sequence matches
    *                        the ordering of `metrics`.
    * @throws InvalidArgumentException If `saveSummaries` is `true`, but the estimator has no working directory
    *                                  specified.
    */
  @throws[InvalidArgumentException]
  def evaluate(
      data: Dataset[IT, IO, ID, IS],
      metrics: Seq[Metric[I, Output]],
      maxSteps: Long = -1L,
      hooks: Seq[Hook] = Seq.empty,
      checkpointPath: Path = null,
      saveSummaries: Boolean = true,
      name: String = null): Seq[Tensor] = {
    // Check that the model has been trained.
    val _checkpointPath = Option(checkpointPath).orElse(workingDir.flatMap(Saver.latestCheckpoint(_)))
    if (_checkpointPath.isEmpty)
      throw CheckpointNotFoundException(
        "No checkpoint was found. Please provide a valid 'workingDir' the estimator configuration, or a path to a " +
            "valid checkpoint file through the 'checkpointPath' argument.")
    val model = modelFunction(configuration)
    val graph = Graph()
    Op.createWith(graph) {
      graph.setRandomSeed(randomSeed)
      val evaluationOps = Op.createWithNameScope("Model")(model.buildEvaluationOps(graph, metrics))
      val inputInitializer = evaluationOps.inputIterator.createInitializer(data)
      Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
      val globalStep = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      val evalStep = Counter.getOrCreate(Graph.Keys.EVAL_STEP, local = true)
      val evalStepUpdate = evalStep.assignAdd(1)
      val evalUpdateOps = ControlFlow.group(evaluationOps.metricUpdates.map(_.op).toSet + evalStepUpdate.op)
      val allHooks = mutable.ListBuffer(hooks: _*)
      allHooks += StopEvaluationHook(maxSteps)
      val saver = getOrCreateSaver()
      val session = MonitoredSession(
        ChiefSessionCreator(
          master = configuration.evaluationMaster,
          sessionScaffold = SessionScaffold(
            initOp = Some(graph.globalVariablesInitializer()),
            localInitOp = Some(ControlFlow.group(Set(inputInitializer, graph.localVariablesInitializer()))),
            saver = Some(saver)),
          sessionConfig = configuration.sessionConfig,
          checkpointPath = workingDir),
        hooks, shouldRecover = true)
      UnsupervisedEstimator.logger.info("Starting evaluation.")
      val (step, metricValues) = {
        try {
          val step = session.run(fetches = globalStep.value).scalar.asInstanceOf[Long]
          while (!session.shouldStop)
            session.run(targets = evalUpdateOps)
          (step, session.run(fetches = evaluationOps.metricValues))
        } catch {
          case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
            session.close()
            (-1L, Seq.empty[Tensor])
          case e: Throwable =>
            session.closeWithoutHookEnd()
            throw e
        }
      }
      if (!session.closed)
        session.close()
      UnsupervisedEstimator.logger.info("Finished evaluation.")
      UnsupervisedEstimator.logger.info("Saving evaluation results.")
      if (saveSummaries) {
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
              UnsupervisedEstimator.logger.warn(
                s"Skipping summary for non-scalar and/or non-floating-point/non-integer metric '$metric'.")
            }
        }
        evaluationDir.map(SummaryFileWriterCache.get).foreach(writer => {
          writer.writeSummary(summaryProto.build(), step)
          writer.flush()
        })
      }
      metricValues
    }
  }
}

object UnsupervisedEstimator {
  private[UnsupervisedEstimator] val logger = Logger(LoggerFactory.getLogger("Learn / Unsupervised Estimator"))

  def apply[IT, IO, ID, IS, I](
      modelFunction: UnsupervisedEstimator.UnsupervisedModelFunction[IT, IO, ID, IS, I],
      configurationBase: Configuration = null): UnsupervisedEstimator[IT, IO, ID, IS, I] = {
    new UnsupervisedEstimator(modelFunction, configurationBase)
  }

  case class UnsupervisedModelFunction[IT, IO, ID, IS, I](
      function: (Configuration) => UnsupervisedTrainableModel[IT, IO, ID, IS, I]) {
    def apply(configuration: Configuration): UnsupervisedTrainableModel[IT, IO, ID, IS, I] = {
      function(configuration)
    }
  }

  trait Implicits {
    implicit def modelToUnsupervisedModelFunction[IT, IO, ID, IS, I](
        model: UnsupervisedTrainableModel[IT, IO, ID, IS, I]
    ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
      UnsupervisedModelFunction((_: Configuration) => model)
    }

    implicit def unitFunctionToUnsupervisedModelFunction[IT, IO, ID, IS, I](
        function: () => UnsupervisedTrainableModel[IT, IO, ID, IS, I]
    ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
      UnsupervisedModelFunction((_: Configuration) => function())
    }

    implicit def unaryRunConfigFunctionToUnsupervisedModelFunction[IT, IO, ID, IS, I](
        function: (Configuration) => UnsupervisedTrainableModel[IT, IO, ID, IS, I]
    ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
      UnsupervisedModelFunction(function)
    }
  }
}
