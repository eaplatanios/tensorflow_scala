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
import org.platanios.tensorflow.api.ops.{Op, Output}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.{RunMetadata, RunOptions}

import scala.util.{Failure, Success, Try}
import scala.util.control.Exception._

/** Wrapper around a [[Session]].
  *
  * This wrapper is used as a base class for various session wrappers that provide additional functionality such as
  * monitoring, coordination, and recovery.
  *
  * In addition to the methods provided by [[Session]] the wrapper provides a method to check for requested stops and
  * never throws any exceptions thrown by calls to `Session.close`.
  *
  * @param  session Session being wrapped.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class SessionWrapper private[learn](protected var session: Session)
    extends Session(session.graphReference, session.nativeHandle, session.target) {
  private[this] var _closed: Boolean = false

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

  /** Overridable method that returns `true` if this session should not be used anymore. */
  protected def checkStop: Boolean

  override def closed: Boolean = _closed

  override def close(): Unit = {
    if (!closed) {
      ignoring(SessionWrapper.PREEMPTION_ERRORS.toSeq: _*)(super.close())
      _closed = true
    }
  }
}

object SessionWrapper {
  // TODO: !!! We need to fill this and find a way to more generally better represent TensorFlow exceptions.
  private[learn] val PREEMPTION_ERRORS: Set[Class[_]] = Set.empty // Set(classOf[IndexOutOfBoundsException])
}

/** Session wrapper that recreates a session upon certain kinds of errors.
  *
  * The constructor is passed a [[SessionCreator]] object, not a [[Session]].
  *
  * TODO: !!! Change the names of acceptable exceptions.
  * Calls to `run()` are delegated to the wrapped session. If a call throws an `AbortedError` or an `UnavailableError`,
  * the wrapped session is closed, and a new one is created by invoking the session creator.
  *
  * @param  sessionCreator Factory for creating new sessions.
  *
  * @author Emmanouil Antonios Platanios
  */
class RecoverableSession private[learn](sessionCreator: SessionCreator)
    extends SessionWrapper(RecoverableSession.createSession(sessionCreator)) {
  override protected def checkStop: Boolean = {
    if (closed) {
      // If the session has been closed, computation should stop.
      true
    } else {
      // If any exception is thrown, we should stop.
      Try(catching(SessionWrapper.PREEMPTION_ERRORS.toSeq: _*).withApply(e => {
        RecoverableSession.logger.info(
          "An exception was thrown while considering whether the session is complete. This may be due to a " +
              "preemption in a connected worker or parameter server. The current session will be closed and a new " +
              "session will be created. Exception: " + e)
        this.close()
        session = sessionCreator.createSession()
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

  override protected def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null, wantMetadata: Boolean = false
  )(implicit
      executable: Executable[E],
      fetchable: Fetchable.Aux[F, R]
  ): (R, Option[RunMetadata]) = {
    var result: (R, Option[RunMetadata]) = null
    while (result == null) {
      if (closed)
        session = RecoverableSession.createSession(sessionCreator)
      handling(SessionWrapper.PREEMPTION_ERRORS.toSeq: _*).by(e => {
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

object RecoverableSession {
  private[RecoverableSession] val logger = Logger(LoggerFactory.getLogger("Learn / Recoverable Session"))

  private[RecoverableSession] def createSession(sessionCreator: SessionCreator): Session = {
    var session: Session = null
    while (session == null) {
      handling(SessionWrapper.PREEMPTION_ERRORS.toSeq: _*).by(e => RecoverableSession.logger.info(
        "An exception was thrown while a session was being created. This may be due to a preemption of a connected " +
            "worker or parameter server. A new session will be created. Exception: " + e)) {
        session = sessionCreator.createSession()
      }
    }
    session
  }
}

/** Session wrapper that works with a [[Coordinator]].
  *
  * Calls to `run()` are delegated to the wrapped session. If a call throws an exception, the exception is reported to
  * the coordinator.
  *
  * In addition, after each call to `run()` this session asks the coordinator if the session should stop. In that case,
  * it will will join all the threads registered with the coordinator before returning.
  *
  * If the coordinator was requested to stop with an exception, that exception will be re-thrown from the call to
  * `run()`.
  *
  * @param  session                Session being wrapped.
  * @param  coordinator            Coordinator to use.
  * @param  stopGracePeriodSeconds Grace period (i.e., number of seconds) given to threads to stop after `close()` has
  *                                been called.
  *
  * @author Emmanouil Antonios Platanios
  */
class CoordinatedSession private[learn](session: Session, coordinator: Coordinator, stopGracePeriodSeconds: Int = 120)
  extends SessionWrapper(session) {
  override protected def checkStop: Boolean = {
    // If the coordinator was asked to stop due to an exception, then that exception needs to be propagated.
    coordinator.reThrowRequestedStopCause()
    // At this point, no exceptions are recorded in the coordinator.
    coordinator.shouldStop
  }

  override def close(): Unit = {
    coordinator.requestStop()
    try {
      coordinator.join(gracePeriodSeconds = stopGracePeriodSeconds, ignoreLiveThreads = true)
    } finally {
      // We intentionally suppress exceptions from the `close()` here since any useful exceptions have are already been
      // reported by `join()`.
      Try(super.close())
    }
  }

  override protected def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null, wantMetadata: Boolean = false
  )(implicit
      executable: Executable[E],
      fetchable: Fetchable.Aux[F, R]
  ): (R, Option[RunMetadata]) = {
    Try(session.runHelper(feeds, fetches, targets, options, wantMetadata)) match {
      case Success(result) => result
      case Failure(cause) =>
        // A non-preemption exception could have been caused by a preemption error in the coordinator. If this is the
        // case, raise that exception instead because it's the root cause. Otherwise, stick to the original cause.
        try {
          coordinator.reThrowRequestedStopCause()
          throw cause
        } catch {
          case e if SessionWrapper.PREEMPTION_ERRORS.contains(e.getClass) => throw e
          case _ => throw cause
        }
    }
  }
}
