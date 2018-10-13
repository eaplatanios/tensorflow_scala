/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.client.{Session, SessionConfig}
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.ops.{Op, UntypedOp}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow

import java.nio.file.Path

import scala.collection.mutable

/** Factory for sessions.
  *
  * @author Emmanouil Antonios Platanios
  */
trait SessionCreator {
  protected var extraInitOps: mutable.Set[UntypedOp] = mutable.Set.empty[UntypedOp]
  protected var extraLocalInitOps: mutable.Set[UntypedOp]   = mutable.Set.empty[UntypedOp]

  def addInitOp(op: UntypedOp): Unit = extraInitOps += op
  def addLocalInitOp(op: UntypedOp): Unit = extraLocalInitOps += op

  def removeInitOp(op: UntypedOp): Unit = extraInitOps -= op
  def removeLocalInitOp(op: UntypedOp): Unit = extraLocalInitOps -= op

  protected lazy val initOp: UntypedOp = ControlFlow.noOp("InitOp")
  protected lazy val localInitOp: UntypedOp = ControlFlow.noOp("LocalInitOp")

  /** Creates a new [[Session]]. */
  def createSession(): Session
}

/** Session factory for `CHIEF`s.
  *
  * @param  master          TensorFlow master to use.
  * @param  sessionScaffold Session scaffold used for gathering and/or building supportive ops. If not specified, a
  *                         default one is created. The session scaffold is used to finalize the graph.
  * @param  sessionConfig   Session configuration to be used for the new sessions.
  * @param  checkpointPath  Path to either a checkpoint file to restore the model from, or a directory containing
  *                         multiple checkpoint files, in which case the latest checkpoint in the directory will be
  *                         used.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ChiefSessionCreator(
    master: String = "",
    sessionScaffold: SessionScaffold = SessionScaffold(),
    sessionConfig: Option[SessionConfig] = None,
    checkpointPath: Option[Path] = None
) extends SessionCreator {
  private[this] var builtSessionScaffold: BuiltSessionScaffold = _
  private[this] var sessionManager      : SessionManager       = _

  override protected lazy val initOp: UntypedOp = {
    if (extraInitOps.isEmpty)
      builtSessionScaffold.initOp
    else
      ControlFlow.group(extraInitOps.toSet + builtSessionScaffold.localInitOp, name = "Init")
  }

  override protected lazy val localInitOp: UntypedOp = {
    if (extraLocalInitOps.isEmpty)
      builtSessionScaffold.localInitOp
    else
      ControlFlow.group(extraLocalInitOps.toSet + builtSessionScaffold.localInitOp, name = "LocalInit")
  }

  override def createSession(): Session = {
    if (builtSessionScaffold == null)
      builtSessionScaffold = sessionScaffold.build()
    val initOp = this.initOp
    val localInitOp = this.localInitOp
    Op.currentGraph.freeze()
    if (sessionManager == null)
      sessionManager = SessionManager(
        graph = Op.currentGraph,
        readyOp = Option(builtSessionScaffold.readyOp),
        readyForLocalInitOp = Option(builtSessionScaffold.readyForLocalInitOp),
        localInitOp = Option(localInitOp))
    sessionManager.prepareSession(
      master = master,
      saver = builtSessionScaffold.saver,
      checkpointPath = checkpointPath,
      sessionConfig = sessionConfig,
      initOp = Option(initOp),
      initFeedMap = builtSessionScaffold.initFeedMap,
      initFunction = builtSessionScaffold.internalInitFunction,
      localInitFunction = builtSessionScaffold.internalLocalInitFunction)
  }
}

/** Session factory for `WORKER`s.
  *
  * @param  master          TensorFlow master to use.
  * @param  sessionScaffold Session scaffold used for gathering and/or building supportive ops. If not specified, a
  *                         default one is created. The session scaffold is used to finalize the graph.
  * @param  sessionConfig   Session configuration to be used for the new sessions.
  *
  * @author Emmanouil Antonios Platanios
  */
case class WorkerSessionCreator(
    master: String = "",
    sessionScaffold: SessionScaffold = SessionScaffold(),
    sessionConfig: Option[SessionConfig] = None
) extends SessionCreator {
  private[this] var builtSessionScaffold: BuiltSessionScaffold = _
  private[this] var sessionManager      : SessionManager       = _

  override protected lazy val initOp: UntypedOp = {
    if (extraInitOps.isEmpty)
      builtSessionScaffold.initOp
    else
      ControlFlow.group(extraInitOps.toSet + builtSessionScaffold.localInitOp, name = "Init")
  }

  override protected lazy val localInitOp: UntypedOp = {
    if (extraLocalInitOps.isEmpty)
      builtSessionScaffold.localInitOp
    else
      ControlFlow.group(extraLocalInitOps.toSet + builtSessionScaffold.localInitOp, name = "LocalInit")
  }

  override def createSession(): Session = {
    if (builtSessionScaffold == null)
      builtSessionScaffold = sessionScaffold.build()
    val localInitOp = this.localInitOp
    Op.currentGraph.freeze()
    if (sessionManager == null)
      sessionManager = SessionManager(
        graph = Op.currentGraph,
        readyOp = Option(builtSessionScaffold.readyOp),
        readyForLocalInitOp = Option(builtSessionScaffold.readyForLocalInitOp),
        localInitOp = Option(localInitOp))
    sessionManager.waitForSession(
      master = master,
      sessionConfig = sessionConfig,
      maxWaitSeconds = 30 * 60 // Wait up to 30 minutes for the session to become ready.
    )
  }
}

/** Session creator that attaches hooks to another session creator.
  * 
  * @param  sessionCreator Wrapped session creator.
  * @param  hooks          Hooks to use.
  *
  * @author Emmanouil Antonios Platanios
  */
private[learn] case class HookedSessionCreator private[learn](
    sessionCreator: SessionCreator,
    hooks: Set[Hook]
) extends SessionCreator {
  override def createSession(): Session = {
    val session = Some(sessionCreator.createSession())
    // Inform the hooks that a new session has been created.
    hooks.foreach(_.internalAfterSessionCreation(session.get))
    new SessionWrapper(session.get, hooks)
  }
}
