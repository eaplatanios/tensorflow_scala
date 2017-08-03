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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.using
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Basic, ControlFlow, Op, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor

import org.scalatest.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class ExecutableSuite extends JUnitSuite {
  def executable[T: Executable](value: T): T = value

  @Test def testOpExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      executable(ControlFlow.noOp(name = "NoOp"))
      succeed
    }
  }

  @Test def testOutputLikeExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val output1 = Basic.constant(Tensor(Tensor(2L), Tensor(1L)), name = "Constant_1")
      val output2 = Basic.constant(Tensor(2L, 1L), name = "Constant_2")
      val output3 = Basic.constant(Tensor(3L), name = "Constant_3")
      executable(output1)
      executable(OutputIndexedSlices(output1, output2, output3))
      executable(SparseOutput(output1, output2, output3))
      succeed
    }
  }

  @Test def testSetExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Set(op1, op2, op3))
      succeed
    }
  }

  @Test def testSeqExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Seq(op1, op2, op3))
      succeed
    }
  }

  @Test def testListExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(List(op1, op2, op3))
      succeed
    }
  }

  @Test def testArrayExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Array(op1, op2, op3))
      succeed
    }
  }

  @Test def testNestedSetExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Set(Set(op1), Set(op2, op3)))
      succeed
    }
  }

  @Test def testNestedSetArrayExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Set(Array(op1), Array(op2, op3)))
      succeed
    }
  }

  @Test def testNestedHeterogeneousSetExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      // TODO: [CLIENT] Maybe support this in the future?
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      assertDoesNotCompile("executable(Set(op1, Set(op2, op3))")
    }
  }

  @Test def testTupleExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable((op1, op2, op3))
      succeed
    }
  }

  @Test def testHeterogeneousNestedTupleExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable((op1, (op2, op3)))
      succeed
    }
  }

  @Test def testHeterogeneousNestedTupleSeqExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable((op1, Seq(op2, op3)))
      succeed
    }
  }
}
