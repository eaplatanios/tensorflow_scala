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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.variables.Variable

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientDescentSpec extends AnyFlatSpec with Matchers {
  "Gradient descent" must "work for dense updates to resource-based variables" in {
      val value0 = Tensor[Double](1.0, 2.0)
      val value1 = Tensor[Double](3.0, 4.0)
      val updatedValue0 = Tensor[Double](1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1)
      val updatedValue1 = Tensor[Double](3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01)
      val graph = Graph()
      val (variable0, variable1, gdOp) = tf.createWith(graph) {
        val variable0 = tf.variable[Double]("v0", Shape(2), tf.ConstantInitializer(Tensor(1, 2)))
        val variable1 = tf.variable[Double]("v1", Shape(2), tf.ConstantInitializer(Tensor(3, 4)))
        val gradient0 = tf.constant(Tensor[Double](0.1, 0.1))
        val gradient1 = tf.constant(Tensor[Double](0.01, 0.01))
        val gdOp = GradientDescent(3.0f).applyGradients(Seq(
          (gradient0, variable0.asInstanceOf[Variable[Any]]),
          (gradient1, variable1.asInstanceOf[Variable[Any]])))
        (variable0.value, variable1.value, gdOp)
      }
      val session = Session(graph)
      session.run(targets = graph.trainableVariablesInitializer())
      var variable0Value = session.run(fetches = variable0)
      var variable1Value = session.run(fetches = variable1)
      // TODO: !!! ??? [TENSORS]
      // assert(variable0Value === value0 +- 1e-6)
      // assert(variable1Value === value1 +- 1e-6)
      session.run(targets = gdOp)
      variable0Value = session.run(fetches = variable0)
      variable1Value = session.run(fetches = variable1)
      // assert(variable0Value === updatedValue0 +- 1e-6)
      // assert(variable1Value === updatedValue1 +- 1e-6)
  }
}
