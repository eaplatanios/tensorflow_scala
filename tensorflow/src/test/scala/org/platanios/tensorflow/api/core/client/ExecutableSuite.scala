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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.using
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Basic, ControlFlow, Op}
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

  @Test def testOpOutputLikeExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val output1 = Basic.constant(Tensor(Tensor(2L), Tensor(1L)), name = "Constant_1")
      val output2 = Basic.constant(Tensor(2L, 1L), name = "Constant_2")
      val output3 = Basic.constant(Tensor(3L), name = "Constant_3")
      executable(output1)
      executable(Op.OutputIndexedSlices(output1, output2, output3))
      executable(Op.SparseOutput(output1, output2, output3))
      succeed
    }
  }

  @Test def testOpSetExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Set(op1, op2, op3))
      succeed
    }
  }

  @Test def testOpSeqExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Seq(op1, op2, op3))
      succeed
    }
  }

  @Test def testOpListExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(List(op1, op2, op3))
      succeed
    }
  }

  @Test def testOpArrayExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Array(op1, op2, op3))
      succeed
    }
  }

  @Test def testOpNestedSetExecutable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val op1 = ControlFlow.noOp(name = "NoOp_1")
      val op2 = ControlFlow.noOp(name = "NoOp_2")
      val op3 = ControlFlow.noOp(name = "NoOp_3")
      executable(Set(Set(op1), Set(op2, op3)))
      succeed
    }
  }

  // @Test def testOpNestedHeterogeneousSetExecutable(): Unit = using(Graph()) { graph =>
  //   Op.createWith(graph) {
  //     val op1 = ControlFlow.noOp(name = "NoOp_1")
  //     val op2 = ControlFlow.noOp(name = "NoOp_2")
  //     val op3 = ControlFlow.noOp(name = "NoOp_3")
  //     executable(Set(op1, Set(op2, op3)))
  //     succeed
  //   }
  // }
}
