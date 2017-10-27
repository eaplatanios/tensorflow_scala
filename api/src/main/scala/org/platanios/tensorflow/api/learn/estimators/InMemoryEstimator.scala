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

import org.platanios.tensorflow.api.config.TensorBoardConfig
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.Fetchable
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.io.Dataset
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.immutable.TreeMap
import scala.collection.mutable

/** In-memory estimator which is used to train, use, and evaluate TensorFlow models, and uses an underlying TensorFlow
  * session that it keeps alive throughout its lifetime. This means that checkpoint files do not need to be written
  * after every call to `train()` and do not need to be loaded on every call to `infer()` or `evaluate()`, since the
  * underlying TensorFlow session used for all these calls stays alive in the background.
  *
  * @param  modelFunction       Model-generating function that can optionally have a [[Configuration]] argument which
  *                             will be used to pass the estimator's configuration to the model and allows customizing
  *                             the model based on the execution environment.
  * @param  configurationBase   Configuration base for this estimator. This allows for setting up distributed training
  *                             environments, for example. Note that this is a *base* for a configuration because the
  *                             estimator might modify it and set some missing fields to appropriate default values, in
  *                             order to obtain its final configuration that can be obtain through its `configuration`
  *                             field.
  * @param  trainHooks          Hooks to use while training (e.g., logging for the loss function value, etc.).
  * @param  trainChiefOnlyHooks Hooks to use while training for the chief node only. This argument is only useful for a
  *                             distributed training setting.
  * @param  inferHooks          Hooks to use while inferring.
  * @param  evaluateHooks       Hooks to use while evaluating.
  * @param  tensorBoardConfig   TensorBoard configuration to use while training. If provided, a TensorBoard server is
  *                             launched while training, using the provided configuration. In that case, it is required
  *                             that TensorBoard is installed for the default Python environment in the system. If
  *                             training in a distributed setting, the TensorBoard server is launched on the chief node.
  * @param  evaluationMetrics   Evaluation metrics to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] private[estimators] (
    override protected val modelFunction: Estimator.ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, EI],
    override protected val configurationBase: Configuration = null,
    val stopCriteria: StopCriteria = StopCriteria(),
    val trainHooks: Set[Hook] = Set.empty,
    val trainChiefOnlyHooks: Set[Hook] = Set.empty,
    val inferHooks: Set[Hook] = Set.empty,
    val evaluateHooks: Set[Hook] = Set.empty,
    val tensorBoardConfig: TensorBoardConfig = null,
    val evaluationMetrics: Seq[Metric[EI, Output]] = Seq.empty
) extends Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI](modelFunction, configurationBase) {
  private[this] val graph: Graph                                                 = Graph()
  private[this] val model: TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = modelFunction(configuration)

  private[this] val stopHook              : StopHook          = StopHook(stopCriteria)
  private[this] var allTrainHooks         : mutable.Set[Hook] = mutable.Set(trainHooks.toSeq: _*) + stopHook
  private[this] var allTrainChiefOnlyHooks: mutable.Set[Hook] = mutable.Set(trainChiefOnlyHooks.toSeq: _*)

  private[this] val (globalStep, trainingOps, inferenceOps, evaluationOps, evaluationUpdateOps) = {
    Op.createWith(graph = graph, deviceFunction = deviceFunction.getOrElse(_.device)) {
      graph.setRandomSeed(randomSeed)
      // TODO: [LEARN] !!! Do we ever update the global epoch?
      Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
      val globalStep = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      Op.createWithNameScope("Model") {
        val trainingOps = model.buildTrainingOps()
        val inferenceOps = model.buildInferenceOps()
        val evaluationOps = model.buildEvaluationOps(evaluationMetrics)
        val evalStep = Counter.getOrCreate(Graph.Keys.EVAL_STEP, local = true)
        val evalStepUpdate = evalStep.assignAdd(1L)
        val evalUpdateOps = ControlFlow.group(evaluationOps.metricUpdates.map(_.op).toSet + evalStepUpdate.op)
        (globalStep, trainingOps, inferenceOps, evaluationOps, evalUpdateOps)
      }
    }
  }

  /** The underlying session that is kept alive throughout this estimator's lifetime. */
  private[this] val session: MonitoredSession = {
    Op.createWith(graph = graph, deviceFunction = deviceFunction.getOrElse(_.device)) {
      val globalStep = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      graph.addToCollection(trainingOps.loss, Graph.Keys.LOSSES)
      allTrainHooks += TensorNaNHook(Set(trainingOps.loss.name))
      allTrainHooks += TensorLoggingHook(TreeMap(
        "Step" -> globalStep.value.name,
        "Loss" -> trainingOps.loss.name
      ), StepHookTrigger(100))
      if (tensorBoardConfig != null)
        allTrainChiefOnlyHooks += TensorBoardHook(tensorBoardConfig)
      val saver = getOrCreateSaver()
      Estimator.monitoredTrainingSession(
        configuration = configuration,
        hooks = allTrainHooks.toSet ++ inferHooks ++ evaluateHooks,
        chiefOnlyHooks = allTrainChiefOnlyHooks.toSet,
        sessionScaffold = SessionScaffold(
          initOp = Some(graph.globalVariablesInitializer()),
          localInitOp = Some(ControlFlow.group(Set(graph.localVariablesInitializer()))),
          saver = Some(saver)))
    }
  }

  /** Trains the model managed by this estimator.
    *
    * @param  data         Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                      supervision labels).
    * @param  stopCriteria Stop criteria to use for stopping the training iteration. For the default criteria please
    *                      refer to the documentation of [[StopCriteria]].
    */
  override def train(data: Dataset[TT, TO, TD, TS], stopCriteria: StopCriteria = this.stopCriteria): Unit = {
    session.resetShouldStop()
    session.removeHooks(inferHooks ++ evaluateHooks)
    Op.createWith(graph) {
      val frozen = graph.isFrozen
      if (frozen)
        graph.unFreeze()
      val initializer = trainingOps.inputIterator.createInitializer(data)
      if (frozen)
        graph.freeze()
      session.disableHooks()
      session.run(targets = initializer)
      session.enableHooks()
      try {
        stopHook.updateCriteria(stopCriteria)
        stopHook.reset(session)
        while (!session.shouldStop)
          session.run(targets = trainingOps.trainOp)
      } catch {
        case t: Throwable if !RECOVERABLE_EXCEPTIONS.contains(t.getClass) =>
          stopHook.updateCriteria(this.stopCriteria)
          session.closeWithoutHookEnd()
          throw t
      }
    }
    session.addHooks(inferHooks ++ evaluateHooks)
  }

  /** Infers output (i.e., computes predictions) for `input` using the model managed by this estimator.
    *
    * `input` can be of one of the following types:
    *
    *   - A [[Dataset]], in which case this method returns an iterator over `(input, output)` tuples corresponding to
    * each element in the dataset. Note that the predictions are computed lazily in this case, whenever an element
    * is requested from the returned iterator.
    *   - A single input of type `IT`, in which case this method returns a prediction of type `I`.
    *
    * Note that, `ModelInferenceOutput` refers to the tensor type that corresponds to the symbolic type `I`. For
    * example, if `I` is `(Output, Output)`, then `ModelInferenceOutput` will be `(Tensor, Tensor)`.
    *
    * @param  input Input for the predictions.
    * @return Either an iterator over `(IT, ModelInferenceOutput)` tuples, or a single element of type `I`, depending on
    *         the type of `input`.
    */
  override def infer[InferInput, InferOutput, ModelInferenceOutput](
      input: InferInput
  )(implicit
      evFetchableIO: Fetchable.Aux[IO, IT],
      evFetchableI: Fetchable.Aux[I, ModelInferenceOutput],
      evFetchableIIO: Fetchable.Aux[(IO, I), (IT, ModelInferenceOutput)],
      ev: Estimator.SupportedInferInput[InferInput, InferOutput, IT, IO, ID, IS, ModelInferenceOutput]
  ): InferOutput = {
    session.resetShouldStop()
    session.removeHooks(currentTrainHooks ++ evaluateHooks)
    val output = Op.createWith(graph) {
      val frozen = graph.isFrozen
      if (frozen)
        graph.unFreeze()
      val initializer = inferenceOps.inputIterator.createInitializer(ev.toDataset(input))
      if (frozen)
        graph.freeze()
      session.disableHooks()
      session.run(targets = initializer)
      session.enableHooks()
      try {
        stopHook.updateCriteria(StopCriteria.none)
        stopHook.reset(session)
        ev.convertFetched(new Iterator[(IT, ModelInferenceOutput)] {
          override def hasNext: Boolean = session.shouldStop
          override def next(): (IT, ModelInferenceOutput) = {
            try {
              session.run(fetches = (inferenceOps.input, inferenceOps.output))
            } catch {
              case t: Throwable =>
                stopHook.updateCriteria(stopCriteria)
                session.closeWithoutHookEnd()
                throw t
            }
          }
        })
      } catch {
        case t: Throwable =>
          stopHook.updateCriteria(stopCriteria)
          session.closeWithoutHookEnd()
          throw t
      }
    }
    session.addHooks(currentTrainHooks ++ evaluateHooks)
    output
  }

  /** Evaluates the model managed by this estimator given the provided evaluation data, `data`.
    *
    * The evaluation process is iterative. In each step, a data batch is obtained from `data` and internal metric value
    * accumulators are updated. The number of steps to perform is controlled through the `maxSteps` argument. If set to
    * `-1`, then all batches from `data` will be processed.
    *
    * If `metrics` is provided, it overrides the value provided in the constructor of this estimator.
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
  override def evaluate(
      data: Dataset[TT, TO, TD, TS],
      metrics: Seq[Metric[EI, Output]],
      maxSteps: Long = -1L,
      saveSummaries: Boolean = true,
      name: String = null): Seq[Tensor] = {
    session.resetShouldStop()
    session.removeHooks(currentTrainHooks ++ inferHooks)
    val values = Op.createWith(graph) {
      val frozen = graph.isFrozen
      if (frozen)
        graph.unFreeze()
      val initializer = evaluationOps.inputIterator.createInitializer(data)
      if (frozen)
        graph.freeze()
      session.disableHooks()
      session.run(targets = initializer)
      session.enableHooks()
      try {
        stopHook.updateCriteria(if (maxSteps != -1L) StopCriteria.steps(maxSteps) else StopCriteria.none)
        stopHook.reset(session)
        InMemoryEstimator.logger.info("Starting evaluation.")
        val (step, metricValues) = {
          try {
            val step = session.run(fetches = globalStep.value).scalar.asInstanceOf[Long]
            while (!session.shouldStop)
              session.run(targets = evaluationUpdateOps)
            (step, session.run(fetches = evaluationOps.metricValues))
          } catch {
            case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
              session.close()
              (-1L, Seq.empty[Tensor])
            case t: Throwable =>
              stopHook.updateCriteria(this.stopCriteria)
              session.closeWithoutHookEnd()
              throw t
          }
        }
        InMemoryEstimator.logger.info("Finished evaluation.")
        InMemoryEstimator.logger.info("Saving evaluation results.")
        if (saveSummaries)
          saveEvaluationSummaries(step, metrics, metricValues, name)
        metricValues
      } catch {
        case t: Throwable =>
          stopHook.updateCriteria(this.stopCriteria)
          session.closeWithoutHookEnd()
          throw t
      }
    }
    session.addHooks(currentTrainHooks ++ inferHooks)
    values
  }

  /** Returns the train hooks being used by this estimator, except for the [[StopHook]] being used. */
  private[this] def currentTrainHooks: Set[Hook] = {
    if (configuration.isChief)
      (allTrainHooks ++ allTrainChiefOnlyHooks).toSet - stopHook
    else
      allTrainHooks.toSet - stopHook
  }
}

object InMemoryEstimator {
  private[estimators] val logger = Logger(LoggerFactory.getLogger("Learn / In-Memory Estimator"))

  def apply[IT, IO, ID, IS, I, TT, TO, TD, TS, EI](
      modelFunction: Estimator.ModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, EI],
      configurationBase: Configuration = null,
      stopCriteria: StopCriteria = StopCriteria(),
      trainHooks: Set[Hook] = Set.empty,
      trainChiefOnlyHooks: Set[Hook] = Set.empty,
      inferHooks: Set[Hook] = Set.empty,
      evaluateHooks: Set[Hook] = Set.empty,
      tensorBoardConfig: TensorBoardConfig = null,
      evaluationMetrics: Seq[Metric[EI, Output]] = Seq.empty
  ): InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = {
    new InMemoryEstimator(
      modelFunction, configurationBase, stopCriteria, trainHooks, trainChiefOnlyHooks, inferHooks, evaluateHooks,
      tensorBoardConfig, evaluationMetrics)
  }
}
