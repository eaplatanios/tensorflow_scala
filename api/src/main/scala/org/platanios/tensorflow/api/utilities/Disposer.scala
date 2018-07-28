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

import java.lang.Thread.currentThread
import java.lang.ref.{PhantomReference, Reference, ReferenceQueue}
import java.security.{AccessController, PrivilegedAction}
import java.util
import java.util.concurrent.ConcurrentHashMap

import scala.annotation.tailrec

/** This class is used for registering and disposing the native data associated with Scala objects.
  *
  * The object can register itself by calling the [[Disposer.add]] method and providing a disposing function to it. This
  * function will be called in order to dispose the native data. It accepts no arguments and returns nothing.
  *
  * When the object becomes unreachable, the provided disposing function for that object will be called.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] object Disposer {
  private val queue  : ReferenceQueue[Any]                  = new ReferenceQueue[Any]
  private val records: util.Map[Reference[Any], () => Unit] = new ConcurrentHashMap[Reference[Any], () => Unit]

  /** Performs the actual registration of the target object to be disposed.
    *
    * @param target Disposable object to register.
    */
  def add(target: Any, disposer: () => Unit ): Reference[Any] = {
    val reference = new PhantomReference(target, queue)
    records.put(reference, disposer)
    // TODO: make sure reference isn't GC'd before this point (e.g., with org.openjdk.jmh.infra.Blackhole::consume).
    reference
  }

  AccessController.doPrivileged(new PrivilegedAction[Unit] {
    override def run(): Unit = {
      // The thread must be a member of a thread group which will not get GCed before the VM exit. For this reason, we
      // make its parent the top-level thread group.
      @tailrec def rootThreadGroup(group: ThreadGroup = currentThread.getThreadGroup): ThreadGroup = {
        group.getParent match {
          case null => group
          case parent => rootThreadGroup(parent)
        }
      }
     
      new Thread(rootThreadGroup(), "TensorFlow Scala API Disposer") {
        override def run = while (true) {
          // Blocks until there is a reference in the queue.
          val referenceToDispose = queue.remove
          records.remove(referenceToDispose).apply()
          referenceToDispose.clear()
        }
        setContextClassLoader(null)
        setDaemon(true)
        setPriority(Thread.MAX_PRIORITY)
        start()
      }
    }
  })
}
