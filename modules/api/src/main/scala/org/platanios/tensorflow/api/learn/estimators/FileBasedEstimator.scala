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

package org.platanios.tensorflow.api.learn.estimators

import org.platanios.tensorflow.api.config._
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.io.CheckpointReader
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

import scala.collection.mutable

// TODO: Issue warning if using this estimator with no checkpoints specified in the provided configuration.

/** File-based estimator which is used to train, use, and evaluate TensorFlow models, and uses checkpoint files for
  * storing and retrieving its state. This means that checkpoint files are written after every call to `train()` and are
  * loaded on every call to `infer()` or `evaluate()`.
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
class FileBasedEstimator[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble, EvalIn] private[estimators] (
    override protected val modelFunction: Estimator.ModelFunction[In, TrainIn, Out, TrainOut, Loss, EvalIn],
    override protected val configurationBase: Configuration = null,
    val stopCriteria: StopCriteria = StopCriteria(),
    val trainHooks: Set[Hook] = Set.empty,
    val trainChiefOnlyHooks: Set[Hook] = Set.empty,
    val inferHooks: Set[Hook] = Set.empty,
    val evaluateHooks: Set[Hook] = Set.empty,
    val tensorBoardConfig: TensorBoardConfig = null,
    val evaluationMetrics: Seq[Metric[EvalIn, Output[Float]]] = Seq.empty
)(implicit
    evOutputStructureIn: OutputStructure[In],
    evOutputStructureTrainIn: OutputStructure[TrainIn],
    evOutputStructureOut: OutputStructure[Out],
    // This implicit helps the Scala 2.11 compiler.
    evOutputStructureInOut: OutputStructure[(In, Out)]
) extends Estimator[In, TrainIn, Out, TrainOut, Loss, EvalIn](modelFunction, configurationBase) {
  /** Trains the model managed by this estimator.
    *
    * @param  data         Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                      supervision labels).
    * @param  stopCriteria Stop criteria to use for stopping the training iteration. For the default criteria please
    *                      refer to the documentation of [[StopCriteria]].
    */
  override def train[TrainInD, TrainInS](
      data: () => Dataset[TrainIn],
      stopCriteria: StopCriteria = this.stopCriteria
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
      evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
  ): Unit = {
    trainWithHooks(data, stopCriteria)
  }

  /** Trains the model managed by this estimator.
    *
    * '''NOTE:''' If you provide any summary saver or checkpoint saver hooks in `hooks` or `chiefOnlyHooks`, then the
    * checkpoint configuration in this estimator's `configuration` will be ignored for the chief and those hooks will be
    * used instead.
    *
    * If any of `hooks`, `chiefOnlyHooks`, or `tensorBoardConfig` are provided, they override the values provided in the
    * constructor of this estimator.
    *
    * @param  data              Training dataset. Each element is a tuple over input and training inputs (i.e.,
    *                           supervision labels).
    * @param  stopCriteria      Stop criteria to use for stopping the training iteration. For the default criteria
    *                           please refer to the documentation of [[StopCriteria]].
    * @param  hooks             Hooks to use while training (e.g., logging for the loss function value, etc.).
    * @param  chiefOnlyHooks    Hooks to use while training for the chief node only. This argument is only useful for
    *                           a distributed training setting.
    * @param  tensorBoardConfig If provided, a TensorBoard server is launched using the provided configuration. In
    *                           that case, it is required that TensorBoard is installed for the default Python
    *                           environment in the system. If training in a distributed setting, the TensorBoard
    *                           server is launched on the chief node.
    */
  def trainWithHooks[TrainInD, TrainInS](
      data: () => Dataset[TrainIn],
      stopCriteria: StopCriteria = this.stopCriteria,
      hooks: Set[Hook] = trainHooks,
      chiefOnlyHooks: Set[Hook] = trainChiefOnlyHooks,
      tensorBoardConfig: TensorBoardConfig = this.tensorBoardConfig
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
      evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
  ): Unit = {
    Op.nameScope("Estimator/Train") {
      if (hooks.exists(_.isInstanceOf[Stopper]) || chiefOnlyHooks.exists(_.isInstanceOf[Stopper]))
        Estimator.logger.warn("The provided stopper hook will be ignored. Please use 'stopCriteria' instead.")
      val needsToTrain = {
        if (!stopCriteria.restartCounting) {
          workingDir.flatMap(dir => Saver.latestCheckpoint(dir).flatMap(latestPath => {
            CheckpointReader(latestPath).getTensor(Graph.Keys.GLOBAL_STEP.name)
          })).map(_.scalar.asInstanceOf[Long]).flatMap(s => stopCriteria.maxSteps.map(_ <= s)).getOrElse(true)
        } else {
          true
        }
      }
      if (!needsToTrain) {
        FileBasedEstimator.logger.debug(
          "Skipping training because no restarting is allowed in the termination criteria and the maximum number of " +
              "steps have already been executed in the past (i.e., saved checkpoint).")
      } else {
        val allHooks = mutable.Set(hooks.filter(!_.isInstanceOf[Stopper]).toSeq: _*)
        val allChiefOnlyHooks = mutable.Set(chiefOnlyHooks.filter(!_.isInstanceOf[Stopper]).toSeq: _*)
        allHooks += Stopper(stopCriteria)
        val model = modelFunction(configuration)
        val graph = Graph()
        Op.createWith(
          graph = graph,
          deviceFunction = deviceFunction
        ) {
          randomSeed.foreach(graph.setRandomSeed)
          // TODO: [LEARN] !!! Do we ever update the global epoch?
          Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
          Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
          val trainOps = Op.nameScope("Model")(model.buildTrainOps())
          graph.addToCollection(Graph.Keys.LOSSES)(trainOps.loss)
          allHooks += NaNChecker(Set(trainOps.loss.name))
          val modelInstance = ModelInstance[In, TrainIn, Out, TrainOut, Loss, EvalIn](
            model = model,
            configuration = configuration,
            trainInputIterator = Some(trainOps.inputIterator),
            trainInput = Some(trainOps.input),
            trainOutput = Some(trainOps.output),
            loss = Some(trainOps.loss),
            gradientsAndVariables = Some(trainOps.gradientsAndVariables),
            trainOp = Some(trainOps.trainOp))
          allHooks.foreach {
            case hook: ModelDependentHook[In, TrainIn, Out, TrainOut, Loss, EvalIn] =>
              hook.setModelInstance(modelInstance)
            case _ => ()
          }
          if (tensorBoardConfig != null)
            allChiefOnlyHooks += TensorBoardHook(tensorBoardConfig)
          val saver = getOrCreateSaver()
          val dataInitializer = trainOps.inputIterator.createInitializer(data())
          val session = Estimator.monitoredTrainingSession(
            configuration = configuration,
            hooks = allHooks.toSet,
            chiefOnlyHooks = allChiefOnlyHooks.toSet,
            sessionScaffold = SessionScaffold(
              localInitFunction = Some((session, _) => session.run(targets = Set(dataInitializer))),
              saver = saver))
          try {
            while (!session.shouldStop)
              session.run(targets = Set(trainOps.trainOp))
          } catch {
            case _: OutOfRangeException => session.setShouldStop(true)
            case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) => session.close()
            case t: Throwable =>
              session.closeWithoutHookEnd()
              throw t
          } finally {
            if (!session.closed)
              session.close()
          }
        }
      }
    }
  }

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
  override def infer[InV, InD, InS, OutV, OutD, OutS, InferIn, InferOut](
      input: () => InferIn
  )(implicit
      evOutputToDataTypeIn: OutputToDataType.Aux[In, InD],
      evOutputToDataTypeOut: OutputToDataType.Aux[Out, OutD],
      evOutputToShapeIn: OutputToShape.Aux[In, InS],
      evOutputToShapeOut: OutputToShape.Aux[Out, OutS],
      evOutputToTensorIn: OutputToTensor.Aux[In, InV],
      evOutputToTensorOut: OutputToTensor.Aux[Out, OutV],
      ev: Estimator.SupportedInferInput[In, InV, OutV, InferIn, InferOut],
      // This implicit helps the Scala 2.11 compiler.
      evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (InV, OutV)]
  ): InferOut = {
    inferWithHooks(input)
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
    * If `hooks` is provided, it overrides the value provided in the constructor of this estimator.
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
  @throws[CheckpointNotFoundException]
  def inferWithHooks[InV, InD, InS, OutV, OutD, OutS, InferIn, InferOut](
      input: () => InferIn,
      hooks: Set[Hook] = inferHooks,
      checkpointPath: Path = null
  )(implicit
      evOutputToDataTypeIn: OutputToDataType.Aux[In, InD],
      evOutputToDataTypeOut: OutputToDataType.Aux[Out, OutD],
      evOutputToShapeIn: OutputToShape.Aux[In, InS],
      evOutputToShapeOut: OutputToShape.Aux[Out, OutS],
      evOutputToTensorIn: OutputToTensor.Aux[In, InV],
      evOutputToTensorOut: OutputToTensor.Aux[Out, OutV],
      ev: Estimator.SupportedInferInput[In, InV, OutV, InferIn, InferOut],
      // This implicit helps the Scala 2.11 compiler.
      evOutputToTensorInOut: OutputToTensor.Aux[(In, Out), (InV, OutV)]
  ): InferOut = {
    Op.nameScope("Estimator/Infer") {
      if (hooks.exists(_.isInstanceOf[Stopper]))
        Estimator.logger.warn("The provided stopper hook will be ignored. Please use 'stopCriteria' instead.")
      // Check that the model has been trained.
      val _checkpointPath = Option(checkpointPath).orElse(workingDir.flatMap(Saver.latestCheckpoint(_)))
      if (_checkpointPath.isEmpty)
        throw CheckpointNotFoundException(
          "No checkpoint was found. Please provide a valid 'workingDir' the estimator configuration, or a path to a " +
              "valid checkpoint file through the 'checkpointPath' argument.")
      val model = modelFunction(configuration)
      val graph = Graph()
      Op.createWith(graph) {
        randomSeed.foreach(graph.setRandomSeed)
        Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
        Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
        val inferOps = Op.nameScope("Model")(model.buildInferOps())
        val modelInstance = ModelInstance[In, TrainIn, Out, TrainOut, Loss, EvalIn](
          model = model,
          configuration = configuration,
          output = Some(inferOps.output))
        hooks.foreach {
          case hook: ModelDependentHook[In, TrainIn, Out, TrainOut, Loss, EvalIn] =>
            hook.setModelInstance(modelInstance)
          case _ => ()
        }
        val saver = getOrCreateSaver()
        val dataInitializer = inferOps.inputIterator.createInitializer(ev.toDataset(input()))
        val session = MonitoredSession(
          ChiefSessionCreator(
            sessionScaffold = SessionScaffold(
              localInitFunction = Some((session, _) => session.run(targets = Set(dataInitializer))),
              saver = saver),
            sessionConfig = configuration.sessionConfig,
            checkpointPath = workingDir),
          hooks.filter(!_.isInstanceOf[Stopper]), shouldRecover = true)
        val output = ev.convertFetched(new Iterator[(InV, OutV)] {
          override def hasNext: Boolean = !session.shouldStop
          override def next(): (InV, OutV) = {
            try {
              session.run(fetches = (inferOps.input, inferOps.output))
            } catch {
              case _: OutOfRangeException =>
                session.setShouldStop(true)
                // TODO: !!! Do something to avoid this null pair.
                (null.asInstanceOf[InV], null.asInstanceOf[OutV])
              case t: Throwable =>
                session.closeWithoutHookEnd()
                throw t
            }
          }
        })
        if (!session.closed)
          session.close()
        output
      }
    }
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
  override def evaluate[TrainInD, TrainInS](
      data: () => Dataset[TrainIn],
      metrics: Seq[Metric[EvalIn, Output[Float]]] = this.evaluationMetrics,
      maxSteps: Long = -1L,
      saveSummaries: Boolean = true,
      name: String = null
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
      evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
  ): Seq[Tensor[Float]] = {
    evaluateWithHooks(data, metrics, maxSteps, saveSummaries = saveSummaries, name = name)
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
    * If `hooks` or `metrics` are provided, they override the values provided in the constructor of this estimator.
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
    * @throws CheckpointNotFoundException If no checkpoint could be found. This can happen if `checkpointPath` is `null`
    *                                     and no checkpoint could be found in this estimator's working directory.
    * @throws InvalidArgumentException    If `saveSummaries` is `true`, but the estimator has no working directory
    *                                     specified.
    */
  @throws[CheckpointNotFoundException]
  @throws[InvalidArgumentException]
  def evaluateWithHooks[TrainInD, TrainInS](
      data: () => Dataset[TrainIn],
      metrics: Seq[Metric[EvalIn, Output[Float]]] = this.evaluationMetrics,
      maxSteps: Long = -1L,
      hooks: Set[Hook] = evaluateHooks,
      checkpointPath: Path = null,
      saveSummaries: Boolean = true,
      name: String = null
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
      evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
  ): Seq[Tensor[Float]] = {
    Op.nameScope("Estimator/Evaluate") {
      if (hooks.exists(_.isInstanceOf[Stopper]))
        Estimator.logger.warn("The provided stopper hook will be ignored. Please use 'stopCriteria' instead.")
      // Check that the model has been trained.
      val _checkpointPath = Option(checkpointPath).orElse(workingDir.flatMap(Saver.latestCheckpoint(_)))
      if (_checkpointPath.isEmpty)
        throw CheckpointNotFoundException(
          "No checkpoint was found. Please provide a valid 'workingDir' the estimator configuration, or a path to a " +
              "valid checkpoint file through the 'checkpointPath' argument.")
      val model = modelFunction(configuration)
      val graph = Graph()
      Op.createWith(graph) {
        randomSeed.foreach(graph.setRandomSeed)
        val evaluateOps = Op.nameScope("Model")(model.buildEvalOps(metrics))
        Counter.getOrCreate(Graph.Keys.GLOBAL_EPOCH, local = false)
        val globalStep = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false).value
        val evalStep = Counter.getOrCreate(Graph.Keys.EVAL_STEP, local = true)
        val evalStepUpdate = evalStep.assignAdd(1L)
        val evalUpdateOps = ControlFlow.group(evaluateOps.metricUpdates.map(_.op).toSet + evalStepUpdate.op)
        val allHooks = mutable.Set(hooks.filter(!_.isInstanceOf[Stopper]).toSeq: _*)
        allHooks += Stopper(StopCriteria(maxSteps = Some(maxSteps)))
        val modelInstance = ModelInstance(
          model, configuration, Some(evaluateOps.inputIterator), Some(evaluateOps.input), Some(evaluateOps.output),
          None, None, None)
        allHooks.foreach {
          case hook: ModelDependentHook[In, TrainIn, Out, TrainOut, Loss, EvalIn] =>
            hook.setModelInstance(modelInstance)
          case _ => ()
        }
        val saver = getOrCreateSaver()
        val dataInitializer = evaluateOps.inputIterator.createInitializer(data())
        val session = MonitoredSession(
          ChiefSessionCreator(
            master = configuration.evaluationMaster,
            sessionScaffold = SessionScaffold(
              localInitFunction = Some((session, _) => session.run(targets = Set(dataInitializer))),
              saver = saver),
            sessionConfig = configuration.sessionConfig,
            checkpointPath = configuration.workingDir),
          allHooks.toSet, shouldRecover = true)
        FileBasedEstimator.logger.debug("Starting evaluation.")
        val (step, metricValues) = {
          try {
            val step = session.run(fetches = globalStep).scalar
            while (!session.shouldStop)
              try {
                session.run(targets = Set(evalUpdateOps))
              } catch {
                case _: OutOfRangeException => session.setShouldStop(true)
              }
            (step, session.run(fetches = evaluateOps.metricValues))
          } catch {
            case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
              session.close()
              (-1L, Seq.empty[Tensor[Float]])
            case t: Throwable =>
              session.closeWithoutHookEnd()
              throw t
          } finally {
            if (!session.closed)
              session.close()
          }
        }
        FileBasedEstimator.logger.debug("Finished evaluation.")
        FileBasedEstimator.logger.debug("Saving evaluation results.")
        if (saveSummaries)
          saveEvaluationSummaries(step, metrics, metricValues, name)
        metricValues
      }
    }
  }
}

object FileBasedEstimator {
  private[estimators] val logger = Logger(LoggerFactory.getLogger("Learn / File-based Estimator"))

  def apply[In: OutputStructure, TrainIn: OutputStructure, Out: OutputStructure, TrainOut, Loss: TF : IsFloatOrDouble, EvalIn](
      modelFunction: Estimator.ModelFunction[In, TrainIn, Out, TrainOut, Loss, EvalIn],
      configurationBase: Configuration = null,
      stopCriteria: StopCriteria = StopCriteria(),
      trainHooks: Set[Hook] = Set.empty,
      trainChiefOnlyHooks: Set[Hook] = Set.empty,
      inferHooks: Set[Hook] = Set.empty,
      evaluateHooks: Set[Hook] = Set.empty,
      tensorBoardConfig: TensorBoardConfig = null,
      evaluationMetrics: Seq[Metric[EvalIn, Output[Float]]] = Seq.empty
  ): FileBasedEstimator[In, TrainIn, Out, TrainOut, Loss, EvalIn] = {
    new FileBasedEstimator(
      modelFunction, configurationBase, stopCriteria, trainHooks, trainChiefOnlyHooks, inferHooks, evaluateHooks,
      tensorBoardConfig, evaluationMetrics)
  }
}
