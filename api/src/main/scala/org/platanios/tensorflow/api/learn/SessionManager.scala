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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session, SessionConfig}
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.types.STRING

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}

import scala.collection.JavaConverters._
import scala.concurrent.TimeoutException
import scala.util.Try

/** Training helper class that creates sessions and restores from checkpoints.
  *
  * This class is a small wrapper that takes care of session creation and checkpoint recovery. It also provides
  * functions that facilitate coordination among multiple training threads or processes:
  *   - Managing checkpoints of trained variables as the training progresses.
  *   - Initializing variables on startup, restoring them from the most recent checkpoint after a crash, or waiting for
  *     checkpoints to become available.
  *
  * Example usage:
  * {{{
  *   tf.createWith(Graph()) {
  *     // Add operations to the graph.
  *     ...
  *     // Create a session manager that will checkpoint the model in '/tmp/mydir'.
  *     val sm = SessionManager()
  *     val session = sm.prepareSession(master, Some(saver), Some("/tmp/mydir"), initOp = Some(initOp))
  *     // Use the session to train the graph.
  *     while (true)
  *       session.run(...)
  *   }
  * }}}
  * `prepareSession()` initializes or restores a model. In order to do so, it requires a `saver` and/or an `initOp` as
  * arguments.
  *
  * A second process could wait for the model to be ready, by doing the following:
  * {{{
  *   tf.createWith(Graph()) {
  *     // Add operations to the graph.
  *     ...
  *     // Create a session manager that will wait for the model to become ready.
  *     val sm = SessionManager()
  *     val session = sm.waitForSession(master)
  *     // Use the session to train the graph.
  *     while (true)
  *       session.run(...)
  *   }
  * }}}
  * `waitForSession()` waits for a model to be initialized by other processes.
  *
  * @param  graph                  Graph that the model will use. Defaults to the current graph.
  * @param  readyOp                Op used to check if the model is ready. The model is considered ready if this op
  *                                returns an empty one-dimensional [[STRING]] tensor. If the op returns a non empty
  *                                one-dimensional [[STRING]] tensor, the elements of that tensor are concatenated and
  *                                used to indicate to the user why the model is not ready. If this is `None`, then the
  *                                model is not checked for readiness.
  * @param  readyForLocalInitOp    Op used to check if the model is ready to execute `localInitOp`. The model is
  *                                considered ready if this op returns an empty one-dimensional [[STRING]] tensor. If
  *                                the op returns a non-empty one-dimensional [[STRING]] tensor, the elements of that
  *                                tensor are concatenated and used to indicate to the user why the model is not ready.
  *                                If this op is provided, then a `localInitOp` must also be provided.
  * @param  localInitOp            Op run immediately after session creation. Usually used to initialize tables and
  *                                local variables.
  * @param  recoveryWaitNumSeconds Number of seconds between checks that the model is ready. It is used by processes to
  *                                wait for a model to be initialized or restored. Defaults to 30 seconds.
  * @throws InvalidArgumentException If a `readyForLocalInitOp` is provided, but no `localInitOp` is provided with it.
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[InvalidArgumentException]
private[learn] case class SessionManager(
    graph: Graph = Op.currentGraph, readyOp: Option[Output] = None, readyForLocalInitOp: Option[Output] = None,
    localInitOp: Option[Op] = None, recoveryWaitNumSeconds: Int = 30) {
  if (readyForLocalInitOp.isDefined && localInitOp.isEmpty)
    throw InvalidArgumentException("If you pass a 'readyForLocalInitOp', you must also pass a 'localInitOp'.")

  /** Creates a [[Session]] and makes sure the model is ready to be used.
    *
    * This method creates a [[Session]] on `master`. If a `saver` object is passed in and `checkpointPath` points to a
    * directory containing valid checkpoint files, then this method will try to recover the model from the saved
    * checkpoints. If no checkpoint files are available and `waitForCheckpoint` is `true`, then the process will check
    * every `recoveryWaitNumSeconds`, and for up to `maxWaitSeconds`, for recovery to succeed.
    *
    * If the model cannot be recovered successfully then it is initialized by either running the provided `initOp`, or
    * by calling the provided `initFunction`. The `localInitOp` is also run after `initOp` and `initFunction`,
    * regardless of whether the model was recovered successfully, but only if `readyForLocalInitOp` passes.
    *
    * If the model cannot be recovered and no `initOp`, `initFunction`, or `localInitOp` is provided, then an exception
    * is thrown.
    *
    * @param  master            TensorFlow master to use.
    * @param  saver             Saver to use for restoring the model.
    * @param  checkpointPath    Path to either a checkpoint file to restore the model from, or a directory containing
    *                           multiple checkpoint files, in which case the latest checkpoint in the directory will be
    *                           used.
    * @param  waitForCheckpoint Boolean flag specifying whether to wait for a checkpoint to become available, if none is
    *                           readily available when this function is called.
    * @param  maxWaitSeconds    Maximum time to wait for checkpoints to become available.
    * @param  sessionConfig     Session configuration to be used for the new session.
    * @param  initOp            Op used to initialize the model.
    * @param  initFeedMap       Feed map passed to the [[Session.run()]] call when executing `initOp`.
    * @param  initFunction      Function used to initialize the model that is called after the optional `initOp` is
    *                           executed. The function takes one argument: the session being initialized.
    * @param  localInitFunction Function used to initialize the model that is called after the optional `localInitOp` is
    *                           executed. The function takes one argument: the session being initialized.
    * @return Created session with a model that is ready to be used.
    * @throws InvalidArgumentException If the model cannot be recovered and no `initOp`, `initFunction`, or
    *                                  `localInitOp` is provided, then an exception is thrown.
    */
  @throws[InvalidArgumentException]
  def prepareSession(
      master: String, saver: Option[Saver] = None, checkpointPath: Option[Path], waitForCheckpoint: Boolean = false,
      maxWaitSeconds: Int = 7200, sessionConfig: Option[SessionConfig] = None, initOp: Option[Op] = None,
      initFeedMap: FeedMap = FeedMap.empty, initFunction: Option[(Session) => Unit] = None,
      localInitFunction: Option[(Session) => Unit] = None): Session = {
    val (session, isLoadedFromCheckpoint) = restoreCheckpoint(
      master, saver, checkpointPath, waitForCheckpoint, maxWaitSeconds, sessionConfig)
    if (!isLoadedFromCheckpoint) {
      if (initOp.isEmpty && initFunction.isEmpty && localInitOp.isEmpty)
        throw InvalidArgumentException(
          "Model is not initialized and no 'initOp', 'initFunction', or 'localInitOp' was provided.")
      initOp.foreach(op => session.run(feeds = initFeedMap, targets = op))
      initFunction.foreach(f => f(session))
    }
    tryLocalInitOp(session).foreach(
      message => throw InvalidArgumentException(
        s"Initialization ops did not make the model ready for local initialization. " +
            s"[initOp: $initOp, initFunction: $initFunction, error: $message]."))
    localInitFunction.foreach(f => f(session))
    isModelReady(session).foreach(
      message => throw InvalidArgumentException(
        s"Initialization ops did not make the model ready. " +
            s"[initOp: $initOp, initFunction: $initFunction, localInitOp: $localInitOp, error: $message]."))
    session
  }

  /** Creates a [[Session]], recovering from a saved checkpoint, if possible.
    *
    * This method creates a new session on `master`. If the session is not initialized and can be recovered from a
    * checkpoint, it also performs that recovery.
    *
    * @param  master            TensorFlow master to use.
    * @param  saver             Saver to use for restoring the model.
    * @param  checkpointPath    Path to either a checkpoint file to restore the model from, or a directory containing
    *                           multiple checkpoint files, in which case the latest checkpoint in the directory will be
    *                           used.
    * @param  waitForCheckpoint Boolean flag specifying whether to wait for a checkpoint to become available, if none is
    *                           readily available when this function is called.
    * @param  maxWaitSeconds    Maximum time to wait for checkpoints to become available.
    * @param  sessionConfig     Session configuration to be used for the new session.
    * @return Tuple containing the created session and a boolean flag indicating whether the session could be recovered
    *         and initialized.
    */
  def recoverSession(
      master: String, saver: Option[Saver] = None, checkpointPath: Option[Path] = None,
      waitForCheckpoint: Boolean = false, maxWaitSeconds: Int = 7200,
      sessionConfig: Option[SessionConfig] = None): (Session, Boolean) = {
    val (session, isLoadedFromCheckpoint) = restoreCheckpoint(
      master, saver, checkpointPath, waitForCheckpoint, maxWaitSeconds, sessionConfig)

    // Always try to run the local initialization op.
    val localInitMessage = tryLocalInitOp(session)

    if (!isLoadedFromCheckpoint) {
      // We do not need to run checks for readiness.
      (session, false)
    } else {
      localInitMessage match {
        case Some(message) =>
          SessionManager.logger.info(
            s"Restoring model from $checkpointPath did not make it ready for local initialization: $message.")
          (session, false)
        case None => isModelReady(session) match {
          case Some(message) =>
            SessionManager.logger.info(
              s"Restoring model from $checkpointPath did not make it ready: $message.")
            (session, false)
          case None =>
            SessionManager.logger.info(s"Restored model from $checkpointPath.")
            (session, true)
        }
      }
    }
  }

  /** Creates a new [[Session]] and waits for the model to be ready.
    *
    * This method creates a new session on `master`. It then waits for the model to be initialized or recovered from a
    * checkpoint. It is expected that another thread or process will make the model ready, and that this is intended to
    * be used by threads/processes that participate in a distributed training configuration where a different
    * thread/process is responsible for initializing or recovering the model being trained.
    *
    * '''NOTE:''' The amount of time this method waits for the session is bounded by `maxWaitSeconds`. By default, this
    * function will wait indefinitely.
    *
    * @param  master         TensorFlow master to use.
    * @param  sessionConfig  Session configuration to be used for the new session.
    * @param  maxWaitSeconds Maximum time to wait (in seconds) for the session to become available. If negative, this
    *                        function will wait indefinitely.
    * @return Created session.
    * @throws TimeoutException If the `maxWaitSeconds` timeout is exceeded.
    */
  @throws[TimeoutException]
  def waitForSession(master: String, sessionConfig: Option[SessionConfig] = None, maxWaitSeconds: Int = -1): Session = {
    val startTime = System.currentTimeMillis()
    var session: Session = null
    var isReady = false
    while (!isReady) {
      session = Session(graph, master, sessionConfig)
      var isModelReadyMessage: Option[String] = None
      val localInitMessage = tryLocalInitOp(session)
      if (localInitMessage.isEmpty) {
        // Successful if `localInitOp` is `None`, or `readyForLocalInitOp` passes.
        isModelReadyMessage = isModelReady(session)
        if (isModelReadyMessage.isEmpty)
          isReady = true
      }
      if (!isReady) {
        // Close the session and ignore any exceptions that may be thrown.
        Try(session.close())
        if (maxWaitSeconds >= 0 && Math.max(
          0, maxWaitSeconds * 1000 - System.currentTimeMillis() + startTime) - recoveryWaitNumSeconds * 1000 < 0) {
          throw new TimeoutException(s"The session was not ready after waiting $maxWaitSeconds secs.")
        }
        SessionManager.logger.info(
          "Waiting for the session to be ready. " +
              s"[readyForLocalInitOp: ${localInitMessage.get}, readyOp: ${isModelReadyMessage.get}].")
        Thread.sleep(recoveryWaitNumSeconds * 1000)
      }
    }
    session
  }

  /** Creates a [[Session]] and tries to restore a checkpoint.
    *
    * @param  master            TensorFlow master to use.
    * @param  saver             Saver to use for restoring the model.
    * @param  checkpointPath    Path to either a checkpoint file to restore the model from, or a directory containing
    *                           multiple checkpoint files, in which case the latest checkpoint in the directory will be
    *                           used.
    * @param  waitForCheckpoint Boolean flag specifying whether to wait for a checkpoint to become available, if none is
    *                           readily available when this function is called.
    * @param  maxWaitSeconds    Maximum time to wait for checkpoints to become available.
    * @param  sessionConfig     Session configuration to be used for the new session.
    * @return Tuple containing the newly created session and a boolean value specifying whether the checkpoint
    *         restoration was successful or not.
    */
  private[this] def restoreCheckpoint(
      master: String, saver: Option[Saver] = None, checkpointPath: Option[Path] = None,
      waitForCheckpoint: Boolean = false, maxWaitSeconds: Int = 7200,
      sessionConfig: Option[SessionConfig] = None): (Session, Boolean) = {
    val session = Session(graph, master, sessionConfig)
    (saver, checkpointPath) match {
      case (Some(_saver), Some(_checkpointPath)) =>
        if (Files.isRegularFile(_checkpointPath)) {
          _saver.restore(session, _checkpointPath)
          (session, true)
        } else {
          // Wait up until `maxWaitSeconds` for the checkpoint to become available.
          var waitTime = 0
          var checkpointState = Saver.loadCheckpointState(_checkpointPath)
          var timeout = false
          while (!timeout && (checkpointState.isEmpty ||
              checkpointState.get.getModelCheckpointPath == null ||
              checkpointState.get.getModelCheckpointPath == "")) {
            if (waitForCheckpoint && waitTime < maxWaitSeconds) {
              SessionManager.logger.info("Waiting for a checkpoint to become available.")
              Thread.sleep(recoveryWaitNumSeconds * 1000)
              waitTime += recoveryWaitNumSeconds
              checkpointState = Saver.loadCheckpointState(_checkpointPath)
            } else {
              timeout = true
            }
          }
          if (timeout) {
            (session, false)
          } else {
            // Load the checkpoint.
            _saver.restore(session, Paths.get(checkpointState.get.getModelCheckpointPath))
            _saver.recoverLastCheckpoints(checkpointState.get.getAllModelCheckpointPathsList.asScala.map(Paths.get(_)))
            (session, true)
          }
        }
      case _ =>
        // If either the saver or the checkpoint path is not specified we cannot restore any checkpoints and we thus
        // just return the created session.
        (session, false)
    }
  }

  /** Checks if the model is ready or not, as determined by `readyOp`.
    *
    * @param  session Session to use.
    * @return Option that is `None` if the model is ready, and that contains an informative message, if not.
    */
  private[this] def isModelReady(session: Session): Option[String] = {
    SessionManager.isReady(readyOp, session, "The model is not ready.")
  }

  /** Checks if the model is ready for local initialization or not, as determined by `readyOp`.
    *
    * @param  session Session to use.
    * @return Option that is `None` if the model is ready, and that contains an informative message, if not.
    */
  private[this] def isModelReadyForLocalInit(session: Session): Option[String] = {
    SessionManager.isReady(readyForLocalInitOp, session, "The model is not ready for local initialization.")
  }

  /** Tries to execute `localInitOp` if it is provided and is ready for local initialization.
    *
    * @param  session Session to use.
    * @return Option that is `None` if the model is ready and `localInitOp` was executed, and that contains an
    *         informative message, if the model is not ready.
    */
  private[this] def tryLocalInitOp(session: Session): Option[String] = {
    localInitOp.flatMap(op => {
      isModelReadyForLocalInit(session) match {
        case None =>
          session.run(targets = op)
          None
        case s => s
      }
    })
  }
}

object SessionManager {
  private[SessionManager] val logger = Logger(LoggerFactory.getLogger("Learn / Session Manager"))

  /** Checks if the model is ready or not, as determined by `readyOp`.
    *
    * @param  readyOp An op which defines the readiness of the model.
    * @param  session Session to use.
    * @param  message Message to log as a warning if the model is not ready.
    * @return Option that is `None` if the model is ready, and that contains an informative message, if not.
    */
  private[SessionManager] def isReady(readyOp: Option[Output], session: Session, message: String): Option[String] = {
    readyOp.flatMap(op => {
      try {
        val readyValue = session.run(fetches = op)
        if (readyValue.size == 0) {
          None
        } else {
          // TODO: Depending on what 'readyOp' returns, this message may be confusing.
          Some(s"Variables not initialized: ${readyValue.entriesIterator.mkString(", ")}.")
        }
      } catch {
        case e: Exception =>
          if (!e.getMessage.contains("uninitialized"))
            logger.warn(message)
          Some(s"An exception was thrown while checking if the model is ready: ${e.getMessage}.")
      }
    })
  }
}
