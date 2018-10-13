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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class VariableSpec extends FlatSpec with Matchers {
  "Variable creation" must "work" in {
    val graph = Graph()
    val variable = tf.createWith(graph = graph) {
      val initializer = tf.ConstantInitializer(Tensor(Tensor(2, 3)))
      tf.variable[Long]("variable", Shape(1, 2), initializer)
    }
    assert(variable.dataType == INT64)
    assert(graph.getCollection(Graph.Keys.GLOBAL_VARIABLES).contains(variable))
    assert(graph.getCollection(Graph.Keys.TRAINABLE_VARIABLES).contains(variable))
    val session = Session(graph = graph)
    session.run(targets = Set(variable.initializer))
    val outputs = session.run(fetches = variable.value)
    val expectedResult = Tensor[Long](Tensor(2, 3))
    assert(outputs(0, 0).scalar == expectedResult(0, 0).scalar)
    assert(outputs(0, 1).scalar == expectedResult(0, 1).scalar)
    session.close()
    graph.close()
  }

  "Variable assignment" must "work" in {
    val graph = Graph()
    val (variable, variableAssignment) = tf.createWith(graph = graph) {
      val a = tf.constant(Tensor(Tensor(5L, 7L)), name = "A")
      val initializer = tf.ConstantInitializer(Tensor(Tensor(2, 3)))
      val variable = tf.variable[Long]("variable", Shape(1, 2), initializer)
      val variableAssignment = variable.assign(a)
      (variable, variableAssignment)
    }
    assert(variable.dataType == INT64)
    assert(graph.getCollection(Graph.Keys.GLOBAL_VARIABLES).contains(variable))
    assert(graph.getCollection(Graph.Keys.TRAINABLE_VARIABLES).contains(variable))
    val session = Session(graph = graph)
    session.run(targets = Set(variable.initializer))
    session.run(targets = Set(variableAssignment))
    val output = session.run(fetches = variable.value)
    val expectedResult = Tensor[Long](Tensor(5, 7))
    assert(output(0, 0).scalar == expectedResult(0, 0).scalar)
    assert(output(0, 1).scalar == expectedResult(0, 1).scalar)
    session.close()
    graph.close()
  }
}
