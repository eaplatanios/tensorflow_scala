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

package org.platanios.tensorflow.api.utilities

import scala.collection.mutable

/** Wrapper around a pointer to a native object, used to handle disposing it when the JVM garbage collector collects
  * their references on the Scala side.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] class NativeHandleWrapper private[api] (var handle: Long) {
  /** Lock for the native handle. */
  private[api] object Lock

  /** Number of active references to the underlying native object. This is useful for graphs, for example, which are
    * used by sessions. */
  private[api] var referenceCount: Int = 0

  /** List of functions that will be called before calling the cleanup functions. Such functions are usually used to
    * clean up native resources used by the native object. */
  private[api] val preCleanupFunctions: mutable.ListBuffer[() => Unit] = mutable.ListBuffer.empty

  /** Adds a cleanup function. That is, a function that will be called before calling the cleanup functions. Such
    * functions are usually used to clean up native resources used by the native object. */
  private[api] def addPreCleanupFunction(function: () => Unit): Unit = {
    preCleanupFunctions.append(function)
  }

  /** List of functions that will be called right before disposing the native object. Such functions are usually used to
    * clean up native resources used by the native object. */
  private[api] val cleanupFunctions: mutable.ListBuffer[() => Unit] = mutable.ListBuffer.empty

  /** Adds a cleanup function. That is, a function that will be called right before disposing the native object. Such
    * functions are usually used to clean up native resources used by the native object. */
  private[api] def addCleanupFunction(function: () => Unit): Unit = {
    cleanupFunctions.append(function)
  }
}

private[api] object NativeHandleWrapper {
  private[api] def apply(handle: Long): NativeHandleWrapper = {
    new NativeHandleWrapper(handle)
  }
}
