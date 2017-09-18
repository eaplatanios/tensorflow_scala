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
import org.platanios.tensorflow.api.learn.Hook.SessionRunContext
import org.platanios.tensorflow.api.ops.{Op, Output}

import org.tensorflow.framework.RunOptions

/** Hook to extend calls to [[MonitoredSession.run]].
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Hook[HF, HE, HR](implicit
    hookExecutableEv: Executable[HE],
    hookFetchableEv: Fetchable.Aux[HF, HR]
) {
  /** Called once before creating the session. When called, the default graph is the one that will be launched in the
    * session. The hook can modify the graph by adding new operations to it. After the `begin` call the graph will be
    * finalized and the other callbacks will not be able to modify the graph anymore. A second `begin` call on the same
    * graph, should not change that graph. */
  def begin(): Unit

  /** Called after a new session is created. This is called to signal the hooks that a new session has been created.
    * This callback has two essential differences with the situation in which `begin` is called:
    *
    *   - When this is called, the graph is finalized and ops can no longer be added to it.
    *   - This method will also be called as a result of recovering a wrapped session (i.e., not only at the beginning
    * of the overall session).
    *
    * @param  session The session that has been created.
    */
  def afterSessionCreation(session: Session): Unit

  /** Called before each call to [[Session.run]]. You can return from this call a [[Hook.SessionRunArgs]] object
    * indicating ops or tensors to add to the upcoming run call. These ops/tensors will be run together with the
    * ops/tensors originally passed to the original run call. The run arguments you return can also contain feeds to be
    * added to the run call.
    *
    * The `runContext` argument is a [[SessionRunContext]] that provides information about the upcoming run call (i.e.,
    * the originally requested ops/tensors, the session, etc.).
    *
    * At this point the graph is finalized and you should not add any new ops.
    *
    * @param  runContext Provides information about the upcoming run call (i.e., the originally requested ops/tensors,
    *                    the session, etc.).
    */
  def beforeSessionRun[F, E, R](runContext: SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Hook.SessionRunArgs[HF, HE, HR]

  /** Called after each call to [[Session.run]].
    *
    * The `runContext` argument is the same one passed to `beforeSessionRun`. `runContext.requestStop()` can be called
    * to stop the iteration.
    *
    * The `runValues` argument contains fetched values for the tensors requested by `beforeSessionRun`.
    *
    * If [[Session.run]] throws an [[IndexOutOfBoundsException]] then `afterSessionRun()` will not be called. Note the
    * difference between the `end()` and the `afterSessionRun()` behavior when [[Session.run]] throws an
    * [[IndexOutOfBoundsException]]. In that case, `end()` is called but `afterSessionRun()` is not called.
    *
    * @param  runContext Provides information about the run call (i.e., the originally requested ops/tensors,
    *                    the session, etc.). Same value as that passed to `beforeSessionRun`.
    * @param  runValues  Fetched values for the tensors requested by `beforeSessionRun`.
    */
  def afterSessionRun[F, E, R](runContext: SessionRunContext[F, E, R], runValues: HR)(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit

  /** Called at the end of the session usage (i.e., [[Session.run]] will not be invoked again after this call).
    *
    * The `session` argument can be used in case the hook wants to execute any final ops, such as saving a last
    * checkpoint.
    *
    * If [[Session.run]] throws any exception other than [[IndexOutOfBoundsException]] then `end()` will not be called.
    * Note the difference between the `end()` and the `afterSessionRun()` behavior when [[Session.run]] throws an
    * [[IndexOutOfBoundsException]]. In that case, `end()` is called but `afterSessionRun()` is not called.
    *
    * @param  session Session that will not be used again after this call.
    */
  def end(session: Session): Unit
}

/** Contains helper classes for the [[Hook]] class. */
object Hook {
  /** Represents a complete set of arguments passed to [[Session.run]]. */
  case class SessionRunArgs[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  )

  /** Provides information about the original request to [[Session.run]] function. [[Hook]]s can stop the loop by
    * calling the `requestStop()` method of [[SessionRunContext]]. In the future we may use this object to add more
    * information about the request to run without changing the Hook API.
    *
    * @param  args    Arguments to the original request to [[Session.run]].
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
}
