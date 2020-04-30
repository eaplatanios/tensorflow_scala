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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * @author Emmanouil Antonios Platanios
  */
class SessionSpec extends AnyFlatSpec with Matchers {
  "Session run fetch by name" should "return the correct result" in {
    val graph = Graph()
    tf.createWith(graph = graph) {
      val a = tf.constant(Tensor(Tensor(2, 3)), name = "A")
      val x = tf.placeholder[Int](Shape(1, 2), name = "X")
      tf.subtract(tf.constant(1), tf.matmul(a = a, b = x, transposeB = true), name = "Y")
    }
    val session = Session(graph = graph)
    val feeds = Map(graph.getOutputByName("X:0").asInstanceOf[Output[Int]] -> Tensor(Tensor(5, 7)))
    val fetches = graph.getOutputByName("Y:0").asInstanceOf[Output[Int]]
    val output = session.run(feeds, fetches)
    val expectedResult = Tensor(Tensor(-30))
    assert(output.scalar == expectedResult.scalar)
    graph.close()
  }
}
