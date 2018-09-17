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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

/** Eager tensor op execution context.
  *
  * This is equivalent to a [[org.platanios.tensorflow.api.core.client.Session]], with the exception that it facilitates
  * eager execution of tensor ops (as opposed to symbolic execution which requires a computation graph to be constructed
  * beforehand).
  *
  * @param  nativeHandleWrapper Wrapper around the pointer to the native server object.
  * @param  closeFn             Function used to delete the native server object (i.e., free relevant memory).
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] final case class Context private (
    protected val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends Closeable {
  /** Lock for the native handle. */
  private[Context] def NativeHandleLock = nativeHandleWrapper.Lock

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = nativeHandleWrapper.handle
}

/** Contains helper functions for dealing with eager tensor op execution contexts. */
private[api] object Context {
  /** Creates a new eager tensor op execution context. */
  def apply(
      sessionConfig: Option[SessionConfig] = None
  ): Context = {
    val nativeHandle = NativeTensor.eagerAllocateContext(sessionConfig.map(_.configProto.toByteArray).orNull)
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandleWrapper.handle != 0) {
          NativeTensor.eagerDeleteContext(nativeHandleWrapper.handle)
          nativeHandleWrapper.handle = 0
        }
      }
    }
    val context = Context(nativeHandleWrapper, closeFn)
    // Keep track of references in the Scala side and notify the native library when the context is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(context, closeFn)
    context
  }
}
