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

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] final case class Context private (private[tensors] var nativeHandle: Long) extends Closeable {
  private[this] object NativeHandleLock
  private[this] var referenceCount: Int = 0

  /** Closes this [[Context]] and releases any resources associated with it. Note that a [[Context]] is not usable after
    * it has been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        while (referenceCount > 0) {
          try {
            NativeHandleLock.wait()
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
              // TODO: [CLIENT] Possible leak of the session and graph in this case?
              return
          }
        }
        NativeTensor.eagerDeleteContext(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

private[api] object Context {
  def apply(): Context = Context(NativeTensor.eagerAllocateContext())
}
