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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.OutOfRangeException
import org.platanios.tensorflow.api.implicits.helpers.{OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.io.events.SummaryFileWriterCache
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.{Op, Output, UntypedOp}
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.Summary

import java.nio.file.Path

// TODO: [LEARN] Add support for dataset groups and mean over datasets.

/** Hooks that can be used to evaluate the performance of an estimator for a separate dataset, while training. This hook
  * creates a new session whenever invoked that loads the latest saved checkpoint and evaluates performance using the
  * provided set of evaluation metrics.
  *
  * @param  log              If `true`, the step rate is logged using the current logging configuration.
  * @param  summaryDir       If provided, summaries for the step rate will be saved in this directory. This is useful
  *                          for visualization using TensorBoard, for example.
  * @param  datasets         Datasets over which to evaluate and which produce elements of the same type as the train
  *                          dataset elements.
  * @param  metrics          Evaluation metrics to use.
  * @param  trigger          Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only
  *                          want to trigger this hook at the end of a run and not during, then you should set `trigger`
  *                          to [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd     If `true`, the hook will be triggered at the end of the run. Note that if this flag is set
  *                          to `true`, then the global step must be computable without using a feed map for the
  *                          [[Session.run()]] call (which should always be the case by default).
  * @param  numDecimalPoints Number of decimal points to use when logging floating point values.
  * @param  randomSeed       Random number generator seed to use.
  * @param  name             Name to use for the evaluation hook when logging and saving metric values. This must follow
  *                          the same formatting guidelines as the name scopes used when constructing graphs.
  *
  * @author Emmanouil Antonios Platanios
  */
class Evaluator[In, TrainIn, Out, TrainOut, Loss, InEval, TrainInD, TrainInS] protected (
    val log: Boolean = true,
    val summaryDir: Path = null,
    val datasets: Seq[(String, () => Dataset[TrainIn])],
    val metrics: Seq[Metric[InEval, Output[Float]]],
    val trigger: HookTrigger = StepHookTrigger(100),
    val triggerAtEnd: Boolean = true,
    val numDecimalPoints: Int = 4,
    val randomSeed: Option[Int] = None,
    val name: String = "Evaluator"
)(implicit
    evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
    evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
) extends TriggeredHook(trigger, triggerAtEnd)
    with ModelDependentHook[In, TrainIn, Out, TrainOut, Loss, InEval] {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")
  require(Op.checkNameScope(name), "Invalid evaluator name.")

  override private[learn] val priority: Int = -1000

  protected var graph              : Graph                       = _
  protected var sessionCreator     : SessionCreator              = _
  protected var datasetInitializers: Seq[(String, UntypedOp)]    = _
  protected var evaluateOps        : Model.EvalOps[TrainIn, Out] = _

  override protected def fetches: Seq[Output[Any]] = Seq.empty
  override protected def targets: Set[UntypedOp] = Set.empty

  override protected def begin(): Unit = {
    graph = Graph()
    Op.createWith(graph, nameScope = name) {
      randomSeed.foreach(graph.setRandomSeed)
      evaluateOps = Op.nameScope("Model")(modelInstance.model.buildEvalOps(metrics))
      datasetInitializers = datasets.map(d => {
        val dataset = d._2()
        (d._1, evaluateOps.inputIterator.createInitializer(dataset).asUntyped)
      })
      this.sessionCreator = ChiefSessionCreator(
        master = modelInstance.configuration.evaluationMaster,
        sessionScaffold = SessionScaffold(),
        sessionConfig = modelInstance.configuration.sessionConfig,
        checkpointPath = modelInstance.configuration.workingDir)
    }
  }

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]],
      session: Session
  ): Unit = Op.createWith(graph, nameScope = name) {
    Evaluator.logger.debug(s"Computing $name.")
    val session = MonitoredSession(sessionCreator, shouldRecover = true)
    val rowNames = datasetInitializers.map(_._1)
    val firstColWidth = rowNames.map(_.length).max
    val colWidth = math.max(metrics.map(_.name.length).max, numDecimalPoints + 6)
    var skippedMetricSummaries = Seq.empty[String]
    val summaryProto = {
      if (summaryDir != null) {
        Evaluator.logger.info(s"Saving $name results at '$summaryDir'.")
        Some(Summary.newBuilder())
      } else {
        None
      }
    }
    if (log) {
      Evaluator.logger.info(s"Step $step $name:")
      Evaluator.logger.info(s"╔═${"═" * firstColWidth}═╤${metrics.map(_ => "═" * (colWidth + 2)).mkString("╤")}╗")
      Evaluator.logger.info(s"║ ${" " * firstColWidth} │${metrics.map(s" %${colWidth}s ".format(_)).mkString("│")}║")
      Evaluator.logger.info(s"╟─${"─" * firstColWidth}─┼${metrics.map(_ => "─" * (colWidth + 2)).mkString("┼")}╢")
    }
    try {
      datasetInitializers.foreach {
        case (datasetName, datasetInitializer) =>
          sessionCreator.addLocalInitOp(datasetInitializer)
          session.run(targets = evaluateOps.metricResets)
          session.run(targets = Set(datasetInitializer))
          var shouldStop = false
          while (!shouldStop) {
            try {
              session.run(targets = evaluateOps.metricUpdates.map(_.op).toSet)
            } catch {
              case _: OutOfRangeException => shouldStop = true
            }
          }
          val value = session.run(fetches = evaluateOps.metricValues)
          sessionCreator.removeLocalInitOp(datasetInitializer)
          summaryProto.foreach(sp => value.zip(metrics.map(_.name)).foreach(m => {
            if (m._1.shape.rank == 0 && (m._1.dataType.isFloatingPoint || m._1.dataType.isInteger)) {
              val castedValue = m._1.toFloat.scalar
              val value = Summary.Value.newBuilder()
              value.setTag(s"$name/$datasetName/${m._2}")
              value.setSimpleValue(castedValue)
              sp.addValue(value)
            } else {
              skippedMetricSummaries :+= s"'${m._2}'"
            }
          }))
          if (log) {
            val line = s"║ %${firstColWidth}s │".format(datasetName) + value.map(metricValue => {
              if (metricValue.shape.rank == 0 && metricValue.dataType.isFloatingPoint) {
                val castedValue = metricValue.toFloat.scalar
                s" %${colWidth}.${numDecimalPoints}f ".format(castedValue)
              } else if (metricValue.shape.rank == 0 && metricValue.dataType.isInteger) {
                val castedValue = metricValue.toLong.scalar
                s" %${colWidth}d ".format(castedValue)
              } else {
                s" %${colWidth}s ".format("Not Scalar")
              }
            }).mkString("│") + "║"
            Evaluator.logger.info(line)
          }
      }
      session.setShouldStop(true)
    } catch {
      case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
        session.close()
      case t: Throwable =>
        session.closeWithoutHookEnd()
        throw t
    } finally {
      if (!session.closed)
        session.close()
    }
    if (log) {
      Evaluator.logger.info(s"╚═${"═" * firstColWidth}═╧${metrics.map(_ => "═" * (colWidth + 2)).mkString("╧")}╝")
    }
    summaryProto.foreach(sp => {
      if (skippedMetricSummaries.nonEmpty)
        Evaluator.logger.warn(
          s"Skipped summaries for non-scalar and/or non-numeric metrics: ${skippedMetricSummaries.mkString(", ")}.")
      val writer = SummaryFileWriterCache.get(summaryDir)
      writer.writeSummary(sp.build(), step)
      writer.flush()
    })
  }
}

object Evaluator {
  private[Evaluator] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Evaluation"))

  def apply[In, TrainIn, Out, TrainOut, Loss, InEval, TrainInD, TrainInS](
      log: Boolean = true,
      summaryDir: Path = null,
      datasets: Seq[(String, () => Dataset[TrainIn])],
      metrics: Seq[Metric[InEval, Output[Float]]],
      trigger: HookTrigger = StepHookTrigger(100),
      triggerAtEnd: Boolean = true,
      numDecimalPoints: Int = 4,
      randomSeed: Option[Int] = None,
      name: String = "Evaluator"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[TrainIn, TrainInD],
      evOutputToShape: OutputToShape.Aux[TrainIn, TrainInS]
  ): Evaluator[In, TrainIn, Out, TrainOut, Loss, InEval, TrainInD, TrainInS] = {
    new Evaluator(log, summaryDir, datasets, metrics, trigger, triggerAtEnd, numDecimalPoints, randomSeed, name)
  }
}
