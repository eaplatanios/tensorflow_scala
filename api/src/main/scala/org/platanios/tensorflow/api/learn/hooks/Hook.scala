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

import org.platanios.tensorflow.api.core.client.{Executable, FeedMap, Fetchable, Session}
import org.platanios.tensorflow.api.learn.{Coordinator, MonitoredSession}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import org.tensorflow.framework.{RunMetadata, RunOptions}

// TODO: [HOOKS] !!! Implement the supposedly provided hooks.
// TODO: [HOOKS] !!! Go through the documentation again and change things as needed after MonitoredSession is implemented.

/** Hook to extend calls to `MonitoredSession.run()`.
  *
  * [[Hook]]s are useful to track training, report progress, request early stopping and more. They use the observer
  * pattern and notify at the following points:
  *   - When a session starts being used,
  *   - Before a call to [[Session.run()]],
  *   - After a call to [[Session.run()]],
  *   - When the session stops being used.
  *
  * A [[Hook]] encapsulates a piece of reusable/composable computation that can piggyback a call to
  * `MonitoredSession.run()`. A hook can add any feeds/fetches/targets to the run call, and when the run call finishes
  * executing with success, the hook gets the fetches it requested. Hooks are allowed to add ops to the graph in the
  * `begin()` method. The graph is finalized after the `begin()` method is called.
  *
  * There are a few pre-defined hooks that can be used without modification:
  *   - `Stop`: Requests to stop iterating based on the provided stopping criteria.
  *   - `TensorLoggingHook`: Logs the values of one or more tensors.
  *   - `TensorNaNHook`: Requests to stop iterating if the provided tensor contains `NaN` values.
  *   - `CheckpointSaverHook`: Saves checkpoints.
  *   - `SummarySaverHook`: Saves summaries to the provided summary writer.
  *
  * For more specific needs you can create custom hooks. For example:
  * {{{
  *   class ExampleHook extends Hook[Output, Unit, Tensor] {
  *     private[this] val logger: Logger = Logger(LoggerFactory.getLogger("Example Hook"))
  *     private[this] var exampleTensor: Output
  *
  *     override def begin(): Unit = {
  *       // You can add ops to the graph here.
  *       logger.info("Starting the session.")
  *       exampleTensor = ...
  *     }
  *
  *     override def afterSessionCreation(session: Session): Unit = {
  *       // When this is called, the graph is finalized and ops can no longer be added to it.
  *       logger.info("Session created.")
  *     }
  *
  *     override def beforeSessionRun[F, E, R](runContext: SessionRunContext[F, E, R])(implicit
  *       executableEv: Executable[E],
  *       fetchableEv: Fetchable.Aux[F, R]
  *      ): Hook.SessionRunArgs[Output, Unit, Tensor] = {
  *        logger.info("Before calling `Session.run()`.")
  *        Hook.SessionRunArgs(fetches = exampleTensor)
  *      }
  *
  *      override def afterSessionRun[F, E, R](runContext: SessionRunContext[F, E, R], runValues: Tensor)(implicit
  *        executableEv: Executable[E],
  *        fetchableEv: Fetchable.Aux[F, R]
  *      ): Unit = {
  *        logger.info("Done running one step. The value of the tensor is: ${runValues.summarize()}")
  *        if (needToStop)
  *          runContext.requestStop()
  *      }
  *
  *      override def end(session: Session): Unit = {
  *        logger.info("Done with the session.")
  *      }
  *   }
  * }}}
  *
  * To understand how hooks interact with calls to `MonitoredSession.run()`, look at following code:
  * {{{
  *   val session = MonitoredTrainingSession(hooks = someHook, ...)
  *   while (!session.shouldStop)
  *     session.run(...)
  *   session.close()
  * }}}
  *
  * The above user code loosely leads to the following execution:
  * {{{
  *   someHook.begin()
  *   val session = tf.Session()
  *   someHook.afterSessionCreation()
  *   while (!stopRequested) {
  *     someHook.beforeSessionRun(...)
  *     try {
  *       val result = session.run(mergedSessionRunArgs)
  *       someHook.afterSessionRun(..., result)
  *     } catch {
  *       case _: IndexOutOfBoundsException => stopRequested = true
  *     }
  *   }
  *   someHook.end()
  *   session.close()
  * }}}
  *
  * Note that if `session.run()` throws an [[IndexOutOfBoundsException]] then `someHook.afterSessionRun()` will not be
  * called, but `someHook.end()` will still be called. On the other hand, if `session.run()` throws any other exception,
  * then neither `someHook.afterSessionRun()` nor `someHook.end()` will be called.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Hook {
  /** Called once before creating the session. When called, the default graph is the one that will be launched in the
    * session. The hook can modify the graph by adding new operations to it. After the `begin` call the graph will be
    * finalized and the other callbacks will not be able to modify the graph anymore. A second `begin` call on the same
    * graph, should not change that graph. */
  def begin(): Unit = ()

  /** Called after a new session is created. This is called to signal the hooks that a new session has been created.
    * This callback has two essential differences with the situation in which `begin()` is called:
    *
    *   - When this is called, the graph is finalized and ops can no longer be added to it.
    *   - This method will also be called as a result of recovering a wrapped session (i.e., not only at the beginning
    *     of the overall session).
    *
    * @param  session     The session that has been created.
    * @param  coordinator The current coordinator.
    */
  def afterSessionCreation(session: Session, coordinator: Coordinator): Unit = ()

  /** Called before each call to `Session.run()`. You can return from this call a [[Hook.SessionRunArgs]] object
    * indicating ops or tensors to add to the upcoming run call. These ops/tensors will be run together with the
    * ops/tensors originally passed to the original run call. The run arguments you return can also contain feeds to be
    * added to the run call.
    *
    * The `runContext` argument is a [[Hook.SessionRunContext]] that provides information about the upcoming run call
    * (i.e., the originally requested ops/tensors, the session, etc.).
    *
    * At this point the graph is finalized and you should not add any new ops.
    *
    * @param  runContext Provides information about the upcoming run call (i.e., the originally requested ops/tensors,
    *                    the session, etc.).
    */
  def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = None

  /** Called after each call to `Session.run()`.
    *
    * The `runContext` argument is the same one passed to `beforeSessionRun()`. `runContext.requestStop()` can be called
    * to stop the iteration.
    *
    * The `runResult` argument contains fetched values for the tensors requested by `beforeSessionRun()`.
    *
    * If `Session.run()` throws any exception, then `afterSessionRun()` will not be called. Note the difference between
    * the `end()` and the `afterSessionRun()` behavior when `Session.run()` throws an [[IndexOutOfBoundsException]]. In
    * that case, `end()` is called but `afterSessionRun()` is not called.
    *
    * @param  runContext Provides information about the run call (i.e., the originally requested ops/tensors, the
    *                    session, etc.). Same value as that passed to `beforeSessionRun`.
    * @param  runResult  Result of the `Session.run()` call that includes the fetched values for the tensors requested
    *                    by `beforeSessionRun()`.
    */
  def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = ()

  /** Called at the end of the session usage (i.e., `Session.run()` will not be invoked again after this call).
    *
    * The `session` argument can be used in case the hook wants to execute any final ops, such as saving a last
    * checkpoint.
    *
    * If `Session.run()` throws any exception other than [[IndexOutOfBoundsException]] then `end()` will not be called.
    * Note the difference between the `end()` and the `afterSessionRun()` behavior when `Session.run()` throws an
    * [[IndexOutOfBoundsException]]. In that case, `end()` is called but `afterSessionRun()` is not called.
    *
    * @param  session Session that will not be used again after this call.
    */
  def end(session: Session): Unit = ()
}

/** Contains helper classes for the [[Hook]] class. */
object Hook {
  /** Provides information about the original request to `Session.run()` function. [[Hook]]s can stop the loop by
    * calling the `requestStop()` method of [[SessionRunContext]]. In the future we may use this object to add more
    * information about the request to run without changing the Hook API.
    *
    * @param  args    Arguments to the original request to `Session.run()`.
    * @param  session Session that will execute the run request.
    */
  case class SessionRunContext[F, E, R](args: SessionRunArgs[F, E, R], session: Session)(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ) {
    private[this] var _stopRequested: Boolean = false

    /** Returns a boolean value representing whether a stop has been requested or not. */
    def stopRequested: Boolean = _stopRequested

    /** Sets the stop requested field to `true`. Hooks can use this function to request stop of iterations.
      * [[MonitoredSession]] checks whether that field has been set or not. */
    def requestStop(): Unit = _stopRequested = true
  }

  /** Represents a complete set of arguments passed to `Session.run()`. */
  case class SessionRunArgs[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  )

  /** Represents the results of a call to `Session.run()`. */
  case class SessionRunResult[F, R](values: R, runMetadata: Option[RunMetadata])(implicit
      fetchableEv: Fetchable.Aux[F, R]
  )
}
