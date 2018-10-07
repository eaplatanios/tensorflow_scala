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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{FLOAT64, INT32}
import org.platanios.tensorflow.api.utilities.using

import org.scalatest.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class FunctionSuite extends JUnitSuite {
  @Test def testIdentitySingleInputSingleOutputFunction(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val function = Function("identity", identity[Output[Double]])
      val input = Basic.constant(Tensor(2.4, -5.6))
      val output = function(input)
      val session = Session()
      val outputValue = session.run(fetches = output)
      assert(outputValue.dataType == FLOAT64)
      assert(outputValue.shape == Shape(2))
      assert(outputValue.entriesIterator.toSeq == Seq(2.4, -5.6))
    }
  }

  @Test def testGeneralSingleInputSingleOutputFunction(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val flatten = Function("flatten", (o: Output[Double]) => o.reshape(Shape(-1)))
      val input = Basic.constant(Tensor(Tensor(2.4, -5.6), Tensor(-0.3, 1.9)))
      val flattenOutput = flatten(input)
      val session = Session()
      val flattenOutputValue = session.run(fetches = flattenOutput)
      assert(flattenOutputValue.dataType == FLOAT64)
      assert(flattenOutputValue.shape == Shape(4))
      assert(flattenOutputValue.entriesIterator.toSeq == Seq(2.4, -5.6, -0.3, 1.9))

      val toInt32 = Function("cast", (o: Output[Double]) => o.castTo[Int])
      val toInt32Output = toInt32(input)
      val toInt32OutputValue = session.run(fetches = toInt32Output)
      assert(toInt32OutputValue.dataType == INT32)
      assert(toInt32OutputValue.shape == Shape(2, 2))
      assert(toInt32OutputValue.entriesIterator.toList == Seq(2, -5, 0, 1))
    }
  }

  @Test def testGeneralDependentInputsSingleOutputFunction(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val one = Basic.constant(1.0)
      val addOne = Function("addOne", (o: Output[Double]) => o + one)
      val input = Basic.constant(Tensor(Tensor(2.4, -5.6), Tensor(-0.3, 1.9)))
      val addOneOutput = addOne(input)
      val session = Session()
      val addOneOutputValue = session.run(fetches = addOneOutput)
      assert(addOneOutputValue.dataType == FLOAT64)
      assert(addOneOutputValue.shape == Shape(2, 2))
      assert(addOneOutputValue.entriesIterator.toList == Seq(3.4, -4.6, 0.7, 2.9))
    }
  }
}
