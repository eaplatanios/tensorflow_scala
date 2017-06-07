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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.ops.Basic._
import org.platanios.tensorflow.api.ops.Math._
import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.tf.createWith

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SessionSpec extends FlatSpec with Matchers {
  "Session run fetch by name" should "return the correct result" in {
    val graph = Graph()
    createWith(graph = graph) {
      val a = constant(tf.Tensor(tf.Tensor(2, 3)), name = "A")
      val x = placeholder(dataType = tf.INT32, name = "X")
      subtract(constant(1), matMul(a = a, b = x, transposeB = true), name = "Y")
    }
    val session = Session(graph = graph)
    val feeds = Map(graph.getOutputByName("X:0") -> tf.Tensor(tf.Tensor(5, 7)))
    val fetches = graph.getOutputByName("Y:0")
    val output = session.run(feeds, fetches)
    val expectedResult = tf.Tensor(tf.Tensor(-30))
    assert(output.scalar === expectedResult.scalar)
    graph.close()
  }
}
