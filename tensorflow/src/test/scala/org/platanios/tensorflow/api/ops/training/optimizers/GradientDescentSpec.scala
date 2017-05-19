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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.{Basic, Op}

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientDescentSpec extends FlatSpec with Matchers {
  "Gradient descent" must "work for dense updates to resource-based variables" in {
    for (dataType <- Set[tf.DataType](tf.FLOAT32, tf.FLOAT64)) {
      val value0 = tf.Tensor(dataType, 1.0, 2.0)
      val value1 = tf.Tensor(dataType, 3.0, 4.0)
      val updatedValue0 = tf.Tensor(dataType, 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1)
      val updatedValue1 = tf.Tensor(dataType, 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01)
      val graph = tf.Graph()
      val (variable0, variable1, gdOp) = Op.createWith(graph) {
        val variable0 = tf.Variable(tf.constantInitializer(tf.Tensor(1, 2)), shape = tf.Shape(2), dataType = dataType)
        val variable1 = tf.Variable(tf.constantInitializer(tf.Tensor(3, 4)), shape = tf.Shape(2), dataType = dataType)
        val gradient0 = Basic.constant(tf.Tensor(0.1, 0.1), dataType = dataType)
        val gradient1 = Basic.constant(tf.Tensor(0.01, 0.01), dataType = dataType)
        val gdOp = GradientDescent(3.0).applyGradients(Seq((gradient0, variable0), (gradient1, variable1)))
        (variable0, variable1, gdOp)
      }
      val session = tf.Session(graph)
      session.run(targets = graph.trainableVariablesInitializer())
      var variable0Value = session.run(fetches = variable0.value)
      var variable1Value = session.run(fetches = variable1.value)
      assert(variable0Value === value0 +- 1e-6)
      assert(variable1Value === value1 +- 1e-6)
      session.run(targets = gdOp)
      variable0Value = session.run(fetches = variable0.value)
      variable1Value = session.run(fetches = variable1.value)
      assert(variable0Value === updatedValue0 +- 1e-6)
      assert(variable1Value === updatedValue1 +- 1e-6)
    }
  }
}
