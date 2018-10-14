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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.tensors.ops.{Math => TensorMath}
import org.platanios.tensorflow.api.utilities.using

import org.junit.Test
import org.scalatest.junit.JUnitSuite

/**
  * @author Emmanouil Antonios Platanios
  */
class CallbackSuite extends JUnitSuite {
  def square[T: IsNotQuantized : TF](input: Tensor[T]): Tensor[T] = {
    input.square
  }

  def add[T: IsNumeric : TF](inputs: Seq[Tensor[T]]): Tensor[T] = {
    TensorMath.addN(inputs)
  }

  @Test def testIdentitySingleInputSingleOutputCallback(): Unit = using(Graph()) { graph =>
    val (input, output) = Op.createWith(graph) {
      val input = Basic.placeholder[Float]()
      val output = Callback.callback(square[Float], input, FLOAT32)
      (input, output)
    }
    val session = Session(graph = graph)
    val outputValue = session.run(Map(input -> Tensor(2f, 5f)), output)
    assert(outputValue.dataType == FLOAT32)
    assert(outputValue.shape == Shape(2))
    assert(outputValue(0).scalar == 4f)
    assert(outputValue(1).scalar == 25f)
  }

  @Test def testIdentityMultipleInputSingleOutputCallback(): Unit = using(Graph()) { graph =>
    val (input1, input2, input3, output) = Op.createWith(graph) {
      val input1 = Basic.placeholder[Double]()
      val input2 = Basic.placeholder[Double]()
      val input3 = Basic.placeholder[Double]()
      val output = Callback.callback(add[Double], Seq(input1, input2, input3), FLOAT64)
      (input1, input2, input3, output)
    }
    val session = Session(graph = graph)
    val outputValue = session.run(Map(
      input1 -> Tensor(2.0, 5.0),
      input2 -> Tensor(-1.3, 3.1),
      input3 -> Tensor(8.9, -4.1)
    ), output)
    assert(outputValue.dataType == FLOAT64)
    assert(outputValue.shape == Shape(2))
    assert(outputValue(0).scalar == 9.6)
    assert(outputValue(1).scalar == 4.0)
  }
}
