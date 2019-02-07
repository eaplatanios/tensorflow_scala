/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.jni

import scala.collection.mutable

/** Keeps a map from unique tokens (i.e., integer IDs) to Scala functions (i.e., callbacks), each of which takes an
  * array with handles of tensors as input and returns an array with handles of tensors as output.
  *
  * @author Emmanouil Antonios Platanios
  */
object ScalaCallbacksRegistry {
  private[this] var uniqueId  = 0
  private[this] val callbacks = mutable.Map.empty[Int, Array[Long] => Array[Long]]

  /** Number of callbacks currently registered. */
  def size: Int = callbacks.size

  /** Registers the provided callback function and returns a unique token to use when creating ops invoking it. */
  def register(function: Array[Long] => Array[Long]): Int = this synchronized {
    val token = uniqueId
    callbacks.update(uniqueId, function)
    uniqueId += 1
    token
  }

  /** De-registers (i.e., removes from this registry) the function that corresponds to the provided token. */
  def deregister(token: Int): Unit = this synchronized {
    callbacks.remove(token)
  }

  /** Invokes the callback identified by `token` using the provides input arguments. */
  def call(token: Int, inputs: Array[Long]): Array[Long] = {
    val callback = this synchronized callbacks(token)
    callback(inputs)
  }
}
