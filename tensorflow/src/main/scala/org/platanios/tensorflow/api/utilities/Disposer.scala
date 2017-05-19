// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.utilities

import sun.misc.ThreadGroupUtils

import java.lang.ref.{PhantomReference, Reference, ReferenceQueue}
import java.security.{AccessController, PrivilegedAction}
import java.util

/** This class is used for registering and disposing the native data associated with Scala objects.
  *
  * The object can register itself by calling the [[Disposer.add]] method and providing a disposing function to it. This
  * function will be called in order to dispose the native data. It accepts no arguments and returns nothing.
  *
  * When the object becomes unreachable, the provided disposing function for that object will be called.
  */
private[api] object Disposer {
  private val queue  : ReferenceQueue[Any]                        = new ReferenceQueue[Any]
  private val records: util.Hashtable[Reference[Any], () => Unit] = new util.Hashtable[Reference[Any], () => Unit]

  /** Performs the actual registration of the target object to be disposed.
    *
    * @param target Disposable object to register.
    */
  def add(target: Any, disposer: () => Unit): Unit = synchronized {
    val reference = new PhantomReference(target, Disposer.queue)
    Disposer.records.put(reference, disposer)
  }

  AccessController.doPrivileged(new PrivilegedAction[Unit] {
    override def run(): Unit = {
      // The thread must be a member of a thread group which will not get GCed before the VM exit. For this reason, we
      // make its parent the top-level thread group.
      val rootThreadGroup: ThreadGroup = ThreadGroupUtils.getRootThreadGroup
      val thread: Thread = new Thread(rootThreadGroup, new Disposer(), "TensorFlow Scala API Disposer")
      thread.setContextClassLoader(null)
      thread.setDaemon(true)
      thread.setPriority(Thread.MAX_PRIORITY)
      thread.start()
    }
  })
}

/** Disposer thread runnable. */
private class Disposer extends Runnable {
  override def run(): Unit = {
    while (true) {
      var referenceToBeDisposed = Disposer.queue.remove // Blocks until there is a reference in the queue
      referenceToBeDisposed.clear()
      var disposer = Disposer.records.remove(referenceToBeDisposed)
      disposer()
      referenceToBeDisposed = null
      disposer = null
    }
  }
}
