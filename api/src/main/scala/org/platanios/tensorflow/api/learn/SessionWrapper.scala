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

import org.platanios.tensorflow.api.core.client.Session

import scala.util.control

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
abstract class SessionWrapper(session: Session)
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
      control.Exception.ignoring(SessionWrapper.PREEMPTION_ERRORS.toSeq: _*)(super.close())
      _closed = true
    }
  }
}

object SessionWrapper {
  // TODO: !!! We need to fill this and find a way to more generally better represent TensorFlow exceptions.
  private[SessionWrapper] val PREEMPTION_ERRORS: Set[Class[_]] = Set(classOf[IndexOutOfBoundsException])
}
