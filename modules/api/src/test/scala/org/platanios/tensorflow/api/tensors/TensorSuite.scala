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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api._

import org.scalatest.junit.JUnitSuite
import org.junit.Test

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorSuite extends JUnitSuite {
  @Test def testCreateNumericTensor(): Unit = {
    val tensor1: Tensor[Int] = -2
    assert(tensor1.dataType === INT32)
    assert(tensor1.shape === Shape())
    assert(tensor1.scalar === -2)
    val tensor2 = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor2.dataType === INT32)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1).scalar === -5)
    val tensor3 = Tensor(Tensor(Tensor(2.0, 3.0), Tensor(0.0, 0.0), Tensor(5.0, 7.0)),
                         Tensor(Tensor(1.0, 23.0), Tensor(4.0, -5.0), Tensor(7.0, 9.0)),
                         Tensor(Tensor(56.0, 1.0), Tensor(-2.0, -4.0), Tensor(-7.0, -9.0)))
    assert(tensor3.dataType === FLOAT64)
    assert(tensor3.shape === Shape(3, 3, 2))
    assert(tensor3(1, 1, 1).scalar === -5.0)
    val tensor4: Tensor[Double] = Tensor(5, 6.0)
    assert(tensor4.dataType === FLOAT64)
    assert(tensor4.shape === Shape(2))
    assert(tensor4(0).scalar === 5.0)
    assert(tensor4(1).scalar === 6.0)
  }

  @Test def testCreateNumericTensorWithDataType(): Unit = {
    val tensor1 = Tensor(INT32,
                         Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor1.dataType === INT32)
    assert(tensor1.shape === Shape(3, 3, 2))
    assert(tensor1(1, 1, 1).scalar === -5)
    val tensor2 = Tensor(FLOAT64,
                         Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor2.dataType === FLOAT64)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1).scalar === -5.0)
  }

  @Test def testCreateStringTensor(): Unit = {
    val tensor1: Tensor[String] = "foo"
    assert(tensor1.dataType === STRING)
    assert(tensor1.shape === Shape())
    assert(tensor1.scalar === "foo")
    val tensor2: Tensor[String] = Tensor("foo", "bar")
    assert(tensor2.dataType === STRING)
    assert(tensor2.shape === Shape(2))
    assert(tensor2(0).scalar === "foo")
    assert(tensor2(1).scalar === "bar")
    val tensor3 = Tensor(Tensor(Tensor("0,0,0", "0,0,1"), Tensor("0,1,0", "0,1,1"), Tensor("0,2,0", "0,2,1")),
                         Tensor(Tensor("1,0,0", "1,0,1"), Tensor("1,1,0", "1,1,1"), Tensor("1,2,0", "1,2,1")),
                         Tensor(Tensor("2,0,0", "2,0,1"), Tensor("2,1,0", "2,1,1"), Tensor("2,2,0", "2,2,1")))
    assert(tensor3.dataType === STRING)
    assert(tensor3.shape === Shape(3, 3, 2))
    assert(tensor3(0, 0, 0).scalar === "0,0,0")
    assert(tensor3(0, 0, 1).scalar === "0,0,1")
    assert(tensor3(0, 1, 0).scalar === "0,1,0")
    assert(tensor3(0, 1, 1).scalar === "0,1,1")
    assert(tensor3(0, 2, 0).scalar === "0,2,0")
    assert(tensor3(0, 2, 1).scalar === "0,2,1")
    assert(tensor3(2, 0, 0).scalar === "2,0,0")
    assert(tensor3(2, 0, 1).scalar === "2,0,1")
    assert(tensor3(2, 1, 0).scalar === "2,1,0")
    assert(tensor3(2, 1, 1).scalar === "2,1,1")
    assert(tensor3(2, 2, 0).scalar === "2,2,0")
    assert(tensor3(2, 2, 1).scalar === "2,2,1")
  }

  @Test def testUnsupportedScalaTypeError(): Unit = {
    assertDoesNotCompile("val tensor: Tensor = Tensor(5.asInstanceOf[Any])")
  }

  @Test def testTensorSlice(): Unit = {
    val tensor = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                        Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                        Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor.shape === Shape(3, 3, 2))
    assert(tensor(::, NewAxis, 1 :: 3, 0 :: -1).shape === Shape(3, 1, 2, 1))
    assert(tensor(---).shape === Shape(3, 3, 2))
    assert(tensor(::, ---).shape === Shape(3, 3, 2))
    assert(tensor(---, -1).shape === Shape(3, 3))
    assert(tensor(---, 0 ::).shape === Shape(3, 3, 2))
    assert(tensor(---, NewAxis).shape === Shape(3, 3, 2, 1))
    assert(tensor(::, ---, NewAxis).shape === Shape(3, 3, 2, 1))
    assert(tensor(---, -1, NewAxis).shape === Shape(3, 3, 1))
    assert(tensor(---, 0 ::, NewAxis).shape === Shape(3, 3, 2, 1))
  }

  // TODO: [TENSORS] Tensor convertible tests.
}
