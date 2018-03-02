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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.OutOfRangeException
import org.platanios.tensorflow.api.io.events.SummaryFileWriterCache
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.{Op, Output, Resource}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.ops.lookup.Lookup
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.FLOAT32

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.Summary

import java.nio.file.Path

/** Hooks that can be used to evaluate the performance of an estimator for a separate dataset, while training. This hook
  * creates a new session whenever invoked that loads the latest saved checkpoint and evaluates performance using the
  * provided set of evaluation metrics.
  *
  * @param  log          If `true`, the step rate is logged using the current logging configuration.
  * @param  summaryDir   If provided, summaries for the step rate will be saved in this directory. This is useful for
  *                      visualization using TensorBoard, for example.
  * @param  datasets     Datasets over which to evaluate and which produce elements of the same type as the train
  *                      dataset elements.
  * @param  metrics      Evaluation metrics to use.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to trigger this hook at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, the hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then the global step must be computable without using a feed map for the
  *                      [[Session.run()]] call (which should always be the case by default).
  * @param  name         Name to use for the evaluation hook when logging and saving metric values. This must follow the
  *                      same formatting guidelines as the name scopes used when constructing graphs.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Evaluator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI](
    log: Boolean = true,
    summaryDir: Path = null,
    datasets: Seq[(String, () => Dataset[TT, TO, TD, TS])],
    metrics: Seq[Metric[EI, Output]],
    trigger: HookTrigger = StepHookTrigger(100),
    triggerAtEnd: Boolean = true,
    randomSeed: Option[Int] = None,
    name: String = "Evaluator"
) extends TriggeredHook(trigger, triggerAtEnd)
    with ModelDependentHook[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")
  require(Op.checkNameScope(name), "Invalid evaluator name.")

  override private[learn] val priority: Int = -1000

  private[this] var graph              : Graph                                  = _
  private[this] var sessionCreator     : SessionCreator                         = _
  private[this] var initializedDatasets: Seq[(String, Dataset[TT, TO, TD, TS])] = _
  private[this] var dataInitializer    : Op                                     = _
  private[this] var evaluateOps        : Model.EvaluateOps[TT, TO, TD, TS, I]   = _

  override protected def begin(): Unit = {
    graph = Graph()
    Op.createWith(graph, nameScope = name) {
      randomSeed.foreach(graph.setRandomSeed)
      initializedDatasets = datasets.map(d => (d._1, d._2()))
      evaluateOps = Op.createWithNameScope("Model")(modelInstance.model.buildEvaluateOps(metrics))
      this.sessionCreator = ChiefSessionCreator(
        master = modelInstance.configuration.evaluationMaster,
        sessionScaffold = SessionScaffold(
          initOp = Some(ControlFlow.group(Set(
            Variable.initializer(Variable.globalVariables),
            Resource.initializer(Resource.sharedResources)))),
          localInitOp = Some(ControlFlow.group(Set(
            Variable.initializer(Variable.localVariables),
            Lookup.lookupsInitializer())))),
        sessionConfig = modelInstance.configuration.sessionConfig,
        checkpointPath = modelInstance.configuration.workingDir)
    }
  }

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]],
      session: Session
  ): Unit = Op.createWith(graph, nameScope = name) {
    Evaluator.logger.debug(s"Computing $name.")
    val session = MonitoredSession(sessionCreator, shouldRecover = true)
    val values = {
      try {
        val values = initializedDatasets.map {
          case (datasetName, dataset) =>
            graph.unFreeze()
            sessionCreator.removeLocalInitOp(dataInitializer)
            dataInitializer = evaluateOps.inputIterator.createInitializer(dataset)
            sessionCreator.addLocalInitOp(dataInitializer)
            graph.freeze()
            session.run(targets = dataInitializer +: evaluateOps.metricResets)
            var shouldStop = false
            while (!shouldStop)
              try {
                session.run(targets = evaluateOps.metricUpdates.toSet)
              } catch {
                case _: OutOfRangeException => shouldStop = true
              }
            val value = session.run(fetches = evaluateOps.metricValues)
            datasetName -> value
        }
        session.setShouldStop(true)
        values
      } catch {
        case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
          session.close()
          Seq.empty[(String, Seq[Tensor])]
        case t: Throwable =>
          session.closeWithoutHookEnd()
          throw t
      } finally {
        if (!session.closed)
          session.close()
      }
    }
    if (log) {
      val datasetValues = values.map(_._2)
      val rowNames = values.map(_._1)
      val firstColWidth = rowNames.map(_.length).max
      val colWidth = math.max(metrics.map(_.name.length).max, 15)
      Evaluator.logger.info(s"Step $step $name:")
      Evaluator.logger.info(s"╔═${"═" * firstColWidth}═╤${metrics.map(_ => "═" * (colWidth + 2)).mkString("╤")}╗")
      Evaluator.logger.info(f"║ ${" " * firstColWidth} │${metrics.map(s" %${colWidth}s ".format(_)).mkString("│")}║")
      Evaluator.logger.info(s"╟─${"─" * firstColWidth}─┼${metrics.map(_ => "─" * (colWidth + 2)).mkString("┼")}╢")
      rowNames.zip(datasetValues).foreach {
        case (datasetName, metricValues) =>
          val line = s"║ %${firstColWidth}s │".format(datasetName) + metricValues.map(metricValue => {
            if (metricValue.shape.rank == 0 &&
                (metricValue.dataType.isFloatingPoint || metricValue.dataType.isInteger)) {
              val castedValue = metricValue.cast(FLOAT32).scalar.asInstanceOf[Float]
              s" %${colWidth}.4f ".format(castedValue)
            } else {
              s" %${colWidth}s ".format("Not Scalar")
            }
          }).mkString("│") + "║"
          Evaluator.logger.info(line)
      }
      Evaluator.logger.info(s"╚═${"═" * firstColWidth}═╧${metrics.map(_ => "═" * (colWidth + 2)).mkString("╧")}╝")
    }
    if (summaryDir != null) {
      Evaluator.logger.info(s"Saving $name results at '$summaryDir'.")
      val datasetNames = values.map(_._1)
      val datasetValues = values.map(_._2).transpose
      val summaryProto = Summary.newBuilder()
      metrics.zip(datasetValues).foreach {
        case (metric, metricValues) =>
          datasetNames.zip(metricValues).foreach {
            case (datasetName, metricValue) =>
              if (metricValue.shape.rank == 0 &&
                  (metricValue.dataType.isFloatingPoint || metricValue.dataType.isInteger)) {
                val castedValue = metricValue.cast(FLOAT32).scalar.asInstanceOf[Float]
                val value = Summary.Value.newBuilder()
                value.setTag(s"$datasetName/${metric.name}")
                value.setSimpleValue(castedValue)
                summaryProto.addValue(value)
              } else {
                Evaluator.logger.warn(s"Skipping summary for non-scalar and/or non-numeric metric '$metric'.")
              }
          }
      }
      val writer = SummaryFileWriterCache.get(summaryDir)
      writer.writeSummary(summaryProto.build(), step)
      writer.flush()
    }
  }
}

object Evaluator {
  private[Evaluator] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Evaluation"))
}
