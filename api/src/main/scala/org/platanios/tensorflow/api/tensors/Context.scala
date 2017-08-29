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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.utilities.Closeable
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

/** Eager tensor op execution context.
  *
  * This is equivalent to a [[org.platanios.tensorflow.api.core.client.Session]], with the exception that it facilitates
  * eager execution of tensor ops (as opposed to symbolic execution which requires a computation graph to be constructed
  * beforehand).
  *
  * @param  nativeHandle Native handle (i.e., pointer) to the underlying native library TensorFlow context.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] final case class Context private (private[api] var nativeHandle: Long) extends Closeable {
  /** Lock for the native handle. */
  private[this] object NativeHandleLock

  /** Closes this [[Context]] and releases any resources associated with it. Note that a [[Context]] is not usable after
    * it has been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        NativeTensor.eagerDeleteContext(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

/** Contains helper functions for dealing with eager tensor op execution contexts. */
private[api] object Context {
  /** Creates a new eager tensor op execution context. */
  def apply(): Context = Context(NativeTensor.eagerAllocateContext())
}
