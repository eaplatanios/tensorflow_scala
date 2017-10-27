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

import org.platanios.tensorflow.api.core.client.{Executable, FeedMap, Fetchable, Session}
import org.platanios.tensorflow.api.core.exception.{AbortedException, UnavailableException}
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.{DebugOptions, RunMetadata, RunOptions}

import scala.collection.mutable
import scala.util.Try
import scala.util.control.Exception._

/** Wrapper around a [[Session]] that invokes [[Hook]] callbacks before and after calls to `Session.run()`.
  *
  * This wrapper is used as a base class for various session wrappers that provide additional functionality such as
  * monitoring and recovery.
  *
  * In addition to the methods provided by [[Session]] the wrapper provides a method to check for requested stops and
  * never throws any exceptions thrown by calls to `Session.close`.
  *
  * The list of hooks to call is passed in the constructor. Before each call to `Session.run()` the session calls the
  * `Hook.beforeSessionRun()` method of each hook, which can return additional ops or tensors to run. These are added to
  * the arguments of the call to `Session.run()`.
  *
  * When the `Session.run()` call finishes, the session invokes the `Hook.afterSessionRun()` method of each hook,
  * passing the values returned by the `Session.run()` call corresponding to the ops and tensors that each hook
  * requested.
  *
  * If any call to the hooks requests a stop via the `runContext`, the session will be marked as needing to stop and its
  * `shouldStop()` method will then return `true`.
  *
  * @param  session Session being wrapped.
  * @param  hooks   Hooks to invoke.
  *
  * @author Emmanouil Antonios Platanios
  */
// TODO: !!! [LEARN] [SESSIONS] This should probably not be extending session (given the confused functionality w.r.t. "runHelper".
class SessionWrapper private[learn](
    protected var session: Session, protected val hooks: Set[Hook] = Set.empty[Hook])
    extends Session(session.graphReference, session.nativeHandle, session.target) {
  protected var _closed      : Boolean = false
  protected var _shouldStop  : Boolean = false
  protected var _hooksEnabled: Boolean = true

  protected val activeHooks: mutable.Set[Hook] = mutable.Set[Hook](hooks.toSeq: _*)

  def enableHooks(): Unit = session match {
    case s: SessionWrapper => s.enableHooks()
    case _ => _hooksEnabled = true
  }

  def disableHooks(): Unit = session match {
    case s: SessionWrapper => s.disableHooks()
    case _ => _hooksEnabled = false
  }

  /** Adds the provided hooks to this session wrapper. Note that the `begin()` method of these hooks may not be called
    * given that they are added in an existing session and not provided in its constructor. */
  def addHooks(hooks: Set[Hook]): Unit = {
    activeHooks ++= hooks
  }

  /** Removes the provided hooks from this session wrapper. */
  def removeHooks(hooks: Set[Hook]): Unit = {
    activeHooks --= hooks
  }

  @throws[RuntimeException]
  override private[api] def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null, wantMetadata: Boolean = false
  )(implicit
      executable: Executable[E],
      fetchable: Fetchable.Aux[F, R]
  ): (R, Option[RunMetadata]) = {
    if (!_hooksEnabled || activeHooks.isEmpty) {
      super.runHelper(feeds, fetches, targets, options, wantMetadata)
    } else {
      // Invoke the hooks' `beforeSessionRun` callbacks.
      val runContext = Hook.SessionRunContext(Hook.SessionRunArgs(feeds, fetches, targets, options), session)
      val runOptions = if (options == null) RunOptions.getDefaultInstance else options
      val combinedArgs = invokeHooksBeforeSessionRun(runContext, runOptions)

      // Do session run.
      val result = super.runHelper(
        combinedArgs.feeds, combinedArgs.fetches, combinedArgs.targets, combinedArgs.options, wantMetadata)

      // Invoke the hooks' `afterSessionRun` callbacks.
      activeHooks.zipWithIndex.foreach(hook => {
        hook._1.afterSessionRun(runContext, Hook.SessionRunResult(result._1._2(hook._2), result._2))
      })

      // Update the `_shouldStop` flag and return.
      _shouldStop ||= runContext.stopRequested
      (result._1._1, result._2)
    }
  }

  /** Invoked `Hook.beforeSessionRun()` for all hooks and manages their feed maps, fetches, and run options. */
  @throws[RuntimeException]
  private[this] def invokeHooksBeforeSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R], runOptions: RunOptions
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Hook.SessionRunArgs[(F, Seq[Seq[Output]]), (E, Seq[Op]), (R, Seq[Seq[Tensor]])] = {
    var hooksFeedMap = FeedMap.empty
    val hooksFetchesList = mutable.ListBuffer.empty[Seq[Output]]
    val hooksTargetsList = mutable.ListBuffer.empty[Op]
    var hooksRunOptions = RunOptions.newBuilder(runOptions).build()
    activeHooks.foreach(hook => hook.beforeSessionRun(runContext) match {
      case Some(runArgs) =>
        if (runArgs.feeds.nonEmpty) {
          if (hooksFeedMap.nonEmpty && hooksFeedMap.intersects(runArgs.feeds))
            throw new RuntimeException("The same tensor is fed by two hooks.")
          hooksFeedMap = hooksFeedMap ++ runArgs.feeds
        }
        if (runArgs.fetches.nonEmpty)
          hooksFetchesList.append(runArgs.fetches)
        else
          hooksFetchesList.append(Seq.empty) // TODO: !!! [HOOKS] Can we avoid these empty sequences entirely?
        if (runArgs.targets.nonEmpty)
          hooksTargetsList.appendAll(runArgs.targets)
        if (runArgs.options != null)
          hooksRunOptions = mergeRunOptions(hooksRunOptions, runArgs.options)
      case None =>
        hooksFetchesList.append(Seq.empty) // TODO: !!! [HOOKS] Can we avoid these empty sequences entirely?
    })
    val feeds = runContext.args.feeds
    if (feeds.nonEmpty && hooksFeedMap.nonEmpty && feeds.intersects(hooksFeedMap))
      throw new RuntimeException("The same tensor is fed by the user and by a hook.")
    val combinedFeeds = feeds ++ hooksFeedMap
    val combinedFetches = (runContext.args.fetches, hooksFetchesList)
    val combinedTargets = (runContext.args.targets, hooksTargetsList)
    Hook.SessionRunArgs(combinedFeeds, combinedFetches, combinedTargets, runOptions)
  }

  /** Merges an instance of [[RunOptions]] into another one, returning a new instance of [[RunOptions]].
    *
    * During the merging, the numerical fields including `TraceLevel`, `TimeoutInMs`, `InterOpThreadPool`, etc., are set
    * to the larger one of the two. The boolean fields are set to the logical OR of the two. The `DebugTensorWatchOpts`
    * of the original run options is extended to include that from the new one.
    *
    * @param  oldOptions Original run options.
    * @param  newOptions New run options to merge into `oldRunOptions`.
    * @return Merged run options as a new instance of [[RunOptions]].
    */
  private[this] def mergeRunOptions(oldOptions: RunOptions, newOptions: RunOptions): RunOptions = {
    val runOptionsBuilder = RunOptions.newBuilder(oldOptions)
    runOptionsBuilder.setTraceLevelValue(Math.max(oldOptions.getTraceLevelValue, newOptions.getTraceLevelValue))
    runOptionsBuilder.setTimeoutInMs(Math.max(oldOptions.getTimeoutInMs, newOptions.getTimeoutInMs))
    runOptionsBuilder.setInterOpThreadPool(Math.max(oldOptions.getInterOpThreadPool, newOptions.getInterOpThreadPool))
    runOptionsBuilder.setOutputPartitionGraphs(
      oldOptions.getOutputPartitionGraphs || newOptions.getOutputPartitionGraphs)
    runOptionsBuilder.mergeDebugOptions(
      DebugOptions.newBuilder()
          .addAllDebugTensorWatchOpts(newOptions.getDebugOptions.getDebugTensorWatchOptsList)
          .build())
    runOptionsBuilder.build()
  }

  /** Returns `true` if this session should not be used anymore. This method always return `true` if the session has
    * been closed already. */
  final def shouldStop: Boolean = {
    if (checkStop || closed) {
      true
    } else {
      session match {
        case s: SessionWrapper => s.shouldStop
        case _ => false
      }
    }
  }

  /** Resets the `shouldStop` flag of this session wrapper to `false`. */
  def resetShouldStop(): Unit = _shouldStop = false

  /** Overridable method that returns `true` if this session should not be used anymore. */
  private[learn] def checkStop: Boolean = _shouldStop

  override def closed: Boolean = _closed

  override def close(): Unit = {
    if (!closed) {
      ignoring(RECOVERABLE_EXCEPTIONS.toSeq: _*)(super.close())
      _closed = true
    }
  }
}

/** Session wrapper that recreates a session upon certain kinds of errors.
  *
  * The constructor is passed a [[SessionCreator]] object, not a [[Session]].
  *
  * Calls to `run()` are delegated to the wrapped session. If a call throws an `AbortedException` or an
  * `UnavailableException`, the wrapped session is closed, and a new one is created by invoking the session creator.
  *
  * @param  sessionCreator Factory for creating new sessions.
  *
  * @author Emmanouil Antonios Platanios
  */
case class RecoverableSession private[learn](sessionCreator: SessionCreator)
    extends SessionWrapper(RecoverableSession.createSession(sessionCreator)) {
  override private[learn] def checkStop: Boolean = {
    if (closed) {
      // If the session has been closed, computation should stop.
      true
    } else {
      // If any exception is thrown, we should stop.
      Try(catching(RECOVERABLE_EXCEPTIONS.toSeq: _*).withApply(e => {
        RecoverableSession.logger.info(
          "An exception was thrown while considering whether the session is complete. This may be due to a " +
              "preemption in a connected worker or parameter server. The current session will be closed and a new " +
              "session will be created. Exception: " + e)
        close()
        session = sessionCreator.createSession()
        _closed = false
        // Since we have just recreated the session, the overall computation should not stop.
        false
      }).apply({
        session match {
          case s: SessionWrapper => s.checkStop
          case _ => false
        }
      })).getOrElse(true)
    }
  }

  override private[api] def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null, wantMetadata: Boolean = false
  )(implicit
      executable: Executable[E],
      fetchable: Fetchable.Aux[F, R]
  ): (R, Option[RunMetadata]) = {
    var result: (R, Option[RunMetadata]) = null
    while (result == null) {
      if (closed) {
        session = RecoverableSession.createSession(sessionCreator)
        _closed = false
      }
      handling(RECOVERABLE_EXCEPTIONS.toSeq: _*).by(e => {
        RecoverableSession.logger.info(
          "An exception was thrown. This may be due to a preemption in a connected worker or parameter server. " +
              "The current session will be closed and a new session will be created. Exception: " + e)
        close()
      }) {
        result = session.runHelper(feeds, fetches, targets, options, wantMetadata)
      }
    }
    result
  }
}

/** Contains helper methods used internally by recoverable sessions. */
object RecoverableSession {
  private[RecoverableSession] val logger = Logger(LoggerFactory.getLogger("Learn / Recoverable Session"))

  private[RecoverableSession] def createSession(sessionCreator: SessionCreator): Session = {
    var session: Session = null
    while (session == null) {
      handling(RECOVERABLE_EXCEPTIONS.toSeq: _*).by(e => RecoverableSession.logger.info(
        "An exception was thrown while a session was being created. This may be due to a preemption of a connected " +
            "worker or parameter server. A new session will be created. Exception: " + e)) {
        session = sessionCreator.createSession()
      }
    }
    session
  }
}

/** Session wrapper that handles initialization, recovery, and hooks.
  *
  * Example usage:
  * {{{
  *   val stopAtStepHook = StopAtStepHook(5, true)
  *   val session = MonitoredSession(ChiefSessionCreator(...), Seq(stopAtStepHook))
  *   while (!session.shouldStop) {
  *     session.run(...)
  *   }
  * }}}
  *
  * '''Initialization:''' At creation time the monitored session does following things, in the presented order:
  *
  *   - Invoke `Hook.begin()` for each hook.
  *   - Add any scaffolding ops and freeze the graph using `SessionScaffold.build()`.
  *   - Create a session.
  *   - Initialize the model using the initialization ops provided by the session scaffold.
  *   - Restore variable values, if a checkpoint exists.
  *   - Invoke `Hook.afterSessionCreation()` for each hook.
  *
  * '''Run:''' When `MonitoredSession.run()` is called, the monitored session does the following things, in the
  * presented order:
  *
  *   - Invoke `Hook.beforeSessionRun()` for each hook.
  *   - Invoke `Session.run()` with the combined feeds, fetches, and targets (i.e., user-provided and hook-provided).
  *   - Invoke `Hook.afterSessionRun()` for each hook.
  *   - Return the result of run call that the user requested.
  *   - For certain types of acceptable exceptions (e.g., aborted or unavailable), recover or reinitialize the session
  *     before invoking the `Session.run()` call, again.
  *
  * '''Exit:''' When `MonitoredSession.close()`, the monitored session does following things, in the presented order:
  *
  *   - Invoke `Hook.end()` for each hook, if no exception has been thrown (other than `AbortedException`, or
  *     `UnavailableException`).
  *
  * === How to Create `MonitoredSession`s ===
  *
  * In most cases you can set the constructor arguments as follows:
  * {{{
  *   MonitoredSession(ChiefSessionCreator(master = ..., sessionConfig = ...))
  * }}}
  *
  * In a distributed setting for a non-chief worker, you can use the following:
  * {{{
  *   MonitoredSession(WorkerSessionCreator(master = ..., sessionConfig = ...))
  * }}}
  *
  * See `MonitoredTrainingSession` for an example usage based on chief or worker.

  * @param  baseSession Session being wrapped.
  * @param  hooks       Hooks to use.
  */
class MonitoredSession private[learn](
    private val baseSession: Session, hooks: Set[Hook]
) extends SessionWrapper(baseSession) {
  private[this] val graphWasFrozen: Boolean = Op.currentGraph.isFrozen

  /** Overridable method that returns `true` if this session should not be used anymore. */
  override private[learn] def checkStop: Boolean = session match {
    case s: SessionWrapper => s.shouldStop
    case _ => false
  }

  /** Closes this session without invoking the `Hook.end()` method for the hooks (e.g., for when exceptions occur). */
  private[learn] def closeWithoutHookEnd(): Unit = {
    try {
      if (closed)
        throw new RuntimeException("This session has already been closed.")
      super.close()
    } finally {
      if (!graphWasFrozen)
        graph.unFreeze()
    }
  }

  @throws[RuntimeException]
  override def close(): Unit = {
    try {
      activeHooks.foreach(_.end(baseSession))
    } catch {
      case _: Throwable => ()
    } finally {
      closeWithoutHookEnd()
    }
  }
}

/** Contains helper methods for creating monitored sessions. */
object MonitoredSession {
  /** Creates a new monitored session.
    *
    * @param  sessionCreator Factory used for creating new sessions (e.g., when recovering from an exception).
    *                        Typically, a [[ChiefSessionCreator]] or a [[WorkerSessionCreator]].
    * @param  hooks          Hooks to use.
    * @param  shouldRecover  Boolean flag indicating whether to recover from [[AbortedException]]s and
    *                        [[UnavailableException]]s.
    * @return Created monitored session.
    */
  def apply(
      sessionCreator: SessionCreator = ChiefSessionCreator(),
      hooks: Set[Hook] = Set.empty,
      shouldRecover: Boolean = true
  ): MonitoredSession = {
    hooks.foreach(_.begin())
    val hookedSessionCreator = HookedSessionCreator(sessionCreator, hooks)
    val session = if (shouldRecover) RecoverableSession(hookedSessionCreator) else hookedSessionCreator.createSession()
    new MonitoredSession(session, hooks)
  }
}
