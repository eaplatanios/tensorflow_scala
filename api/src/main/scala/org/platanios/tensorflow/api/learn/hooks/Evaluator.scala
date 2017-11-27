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
import org.platanios.tensorflow.api.core.client.{Executable, Fetchable, Session}
import org.platanios.tensorflow.api.core.exception.OutOfRangeException
import org.platanios.tensorflow.api.io.events.SummaryFileWriterCache
import org.platanios.tensorflow.api.learn.{Counter, RECOVERABLE_EXCEPTIONS, SessionCreator, SessionWrapper}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.io.data.Dataset
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
  * @param  data         Dataset over which to evaluate and which produces elements of the same type as the train
  *                      dataset elements.
  * @param  metrics      Evaluation metrics to use.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to trigger this hook at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, the hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then the global step must be computable without using a feed map for the
  *                      [[Session.run()]] call (which should always be the case by default).
  * @param  name         Name to use for the evaluation hook when logging and saving metric values.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Evaluator[I, TT, TO, TD, TS](
    log: Boolean = true,
    summaryDir: Path = null,
    data: () => Dataset[TT, TO, TD, TS],
    metrics: Seq[Metric[(I, TO), Output]],
    trigger: HookTrigger = StepHookTrigger(100),
    triggerAtEnd: Boolean = true,
    name: String = null
) extends ModelDependentHook[I, TT, TO, TD, TS] {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")

  private[this] var sessionCreator: SessionCreator = _
  private[this] var step               : Variable    = _
  private[this] var iteratorInitializer: Op          = _
  private[this] var metricValues       : Seq[Output] = _
  private[this] var metricUpdates      : Seq[Output] = _
  private[this] var metricResets       : Seq[Op]     = _

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override protected def begin(sessionCreator: SessionCreator): Unit = {
    this.sessionCreator = sessionCreator
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'TensorLoggingHook'."))
    internalTrigger.reset()
    shouldTrigger = false
    iteratorInitializer = modelInstance.inputIterator.createInitializer(data())
    val streamingInstances = metrics.map(_.streaming((modelInstance.output, modelInstance.input)))
    metricValues = streamingInstances.map(_.value)
    metricUpdates = streamingInstances.map(_.update)
    metricResets = streamingInstances.map(_.reset)
    sessionCreator.addLocalInitOp(Variable.initializer(streamingInstances.flatMap(_.variables).toSet))
  }

  override protected def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override protected def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt)
    Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  override protected def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    lastStep = runResult.values.head.scalar.asInstanceOf[Long]
    if (shouldTrigger) {
      internalTrigger.updateLastTrigger(lastStep.toInt - 1)
      Evaluator.logger.info(s"Computing $name.")
      val session = new SessionWrapper(sessionCreator.createSession())
      val (step, values) = {
        try {
          session.run(targets = iteratorInitializer)
          while (!session.shouldStop)
            try {
              session.run(targets = metricUpdates.map(_.op).toSet)
            } catch {
              case _: OutOfRangeException => session.setShouldStop(true)
            }
          (lastStep, session.run(fetches = metricValues))
        } catch {
          case e if RECOVERABLE_EXCEPTIONS.contains(e.getClass) =>
            session.close()
            (-1L, Seq.empty[Tensor])
          case t: Throwable => throw t
        } finally {
          session.close()
        }
      }
      processResults(step, values)
    }
  }

  private[this] def processResults(step: Long, values: Seq[Tensor]): Unit = {
    if (log) {
      Evaluator.logger.info(s"Step $step $name:")
      metrics.zip(values).foreach {
        case (metric, metricValue) =>
          if (metricValue.shape.rank == 0 &&
              (metricValue.dataType.isFloatingPoint || metricValue.dataType.isInteger)) {
            val castedValue = metricValue.cast(FLOAT32).scalar.asInstanceOf[Float]
            Evaluator.logger.info(f"\t${metric.name}%10s: $castedValue%10.4f")
          } else {
          Evaluator.logger.warn(
            s"\tSkipping logging for non-scalar and/or non-floating-point/non-integer metric '$metric'.")
        }
      }
    }
    if (summaryDir != null) {
      Evaluator.logger.info(s"Saving evaluation results at '$summaryDir'.")
      val summaryProto = Summary.newBuilder()
      metrics.zip(values).foreach {
        case (metric, metricValue) =>
          if (metricValue.shape.rank == 0 &&
              (metricValue.dataType.isFloatingPoint || metricValue.dataType.isInteger)) {
            val castedValue = metricValue.cast(FLOAT32).scalar.asInstanceOf[Float]
            val value = Summary.Value.newBuilder()
            value.setTag(metric.name)
            value.setSimpleValue(castedValue)
            summaryProto.addValue(value)
          } else {
            Evaluator.logger.warn(
              s"Skipping summary for non-scalar and/or non-floating-point/non-integer metric '$metric'.")
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
