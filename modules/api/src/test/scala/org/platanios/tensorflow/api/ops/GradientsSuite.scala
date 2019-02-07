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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.types.{INT32, FLOAT64}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.using

import org.scalatest.junit.JUnitSuite
import org.junit.Test

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientsSuite extends JUnitSuite {
  @Test def testOpsBetweenSimple(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val t1 = Basic.constant(1.0)
      val t2 = Basic.constant(2.0)
      val t3 = Basic.stack(Seq(t1, t2))
      // Full graph
      assertOpSeqEqual(Seq(t3.op, t2.op, t1.op), opsBetween(Set(t1.op, t2.op), Set(t3.op)))
      // Only `t1` and `t3`
      assertOpSeqEqual(Seq(t3.op, t1.op), opsBetween(Set(t1.op), Set(t3.op)))
    }
  }

  @Test def testOpsBetweenUnreachable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val t1 = Basic.constant(1.0)
      val t2 = Basic.constant(2.0)
      Basic.stack(Seq(t1, t2))
      val t4 = Basic.constant(1.0)
      val t5 = Basic.constant(2.0)
      val t6 = Basic.stack(Seq(t4, t5))
      assertOpSeqEqual(Seq(t6.op), opsBetween(Set(t1.op), Set(t6.op)))
    }
  }

  @Test def testOpsBetweenCut(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val t1 = Basic.constant(1.0)
      val t2 = Basic.constant(2.0)
      val t3 = Basic.stack(Seq(t1, t2))
      val t4 = Basic.constant(Tensor(1.0))
      val t5 = Basic.concatenate(Seq(t4, t3), 0)
      val t6 = Basic.constant(Tensor(2.0))
      val t7 = Basic.concatenate(Seq(t5, t6), 0)
      assertOpSeqEqual(Seq(t4.op, t5.op, t7.op), opsBetween(Set(t4.op), Set(t7.op)))
    }
  }

  @Test def testOpsBetweenCycle(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val t1 = Basic.constant(1.0)
      val t2 = Basic.constant(2.0)
      val t3 = Basic.stack(Seq(t1, t2))
      val t4 = Basic.concatenate(Seq(t3, t3, t3), 0)
      val t5 = Basic.constant(Tensor(1.0))
      val t6 = Basic.concatenate(Seq(t4, t5), 0)
      val t7 = Basic.concatenate(Seq(t6, t3), 0)
      assertOpSeqEqual(Seq(t3.op, t4.op, t6.op), opsBetween(Set(t3.op), Set(t6.op)))
      assertOpSeqEqual(Seq(t1.op, t3.op, t4.op, t5.op, t6.op, t7.op), opsBetween(Set(t1.op, t5.op), Set(t7.op)))
      assertOpSeqEqual(Seq(t2.op, t3.op, t4.op, t5.op, t6.op), opsBetween(Set(t2.op, t5.op), Set(t6.op)))
    }
  }

  @Test def testGradientsSimple(): Unit = {
    val graph = Graph()
    val expectedGraph = Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Op.createWith(graph)(Gradients.gradients(Seq(output), inputs, FLOAT64))
    val expectedGradients = buildExpectedGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  @Test def testCCGradientsSimple(): Unit = {
    val graph = Graph()
    val expectedGraph = Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Op.createWith(graph)(Gradients.ccGradients(Array(output), inputs))
    val expectedGradients = buildExpectedCCGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  @Test def testGradients(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(1.0, Shape(32, 100), name = "Input")
      val w = Basic.constant(1.0, Shape(100, 10), name = "W")
      val b = Basic.constant(1.0, Shape(10), name = "b")
      val xw = Math.matmul(input, w, name = "xW")
      val h = NN.addBias(xw, b, name = "h")
      val gradient = Gradients.gradients(Seq(h), Seq(w), FLOAT64).head
      assert(gradient.op.opType === "MatMul")
      assert(gradient.op.booleanAttribute("transpose_a"))
      assert(!gradient.op.booleanAttribute("transpose_b"))
    }
  }

  @Test def testUnusedOutput[Any](): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val w = Basic.constant(1.0, Shape(2, 2))
      val x = Basic.constant(1.0, Shape(2, 2))
      val wx = Math.matmul(w, x)
      val wxSplit = Basic.splitEvenly(wx, 2, axis = 0)
      val c = Math.sum(wxSplit(1))
      val gradient = Gradients.gradients(Seq(c), Seq(w), FLOAT64).head
      assert(gradient.op.opType === "MatMul")
    }
  }

  @Test def testBoundaryStop(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      // Test that we do not differentiate `x`. The gradient function for `x` is set explicitly to `null` so we will get
      // an exception if the gradient code tries to differentiate `x`.
      val c = Basic.constant(1.0)
      val x = Basic.identity(c)
      val y = Math.add(x, 1.0)
      val z = Math.add(y, 1.0)
      val gradients = Gradients.gradients(Seq(z), Seq(x), FLOAT64)
      assert(!gradients.contains(null))
    }
  }

  @Test def testBoundaryContinue(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      // Test that we differentiate both `x` and `y` correctly when `x` is a predecessor of `y`.
      val x = Basic.constant(1.0)
      val y = Math.multiply(x, 2.0)
      val z = Math.multiply(y, 3.0)
      val gradients = Gradients.gradients(Seq(z), Seq(x, y), FLOAT64)
      assert(!gradients.contains(null))
      val session = Session()
      assert(session.run(fetches = gradients.head.toOutput).scalar.asInstanceOf[Double] === 6.0)
    }
  }

  @Test def testNonDifferentiableSwitchInWhileLoop(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val v = Basic.placeholder[Float](Shape.scalar())
      val lv = ControlFlow.whileLoop(
        (lv: (Output[Int], Output[Int], TensorArray[Int])) => Math.less(lv._1, 4),
        (lv: (Output[Int], Output[Int], TensorArray[Int])) => {
          val a = Math.add(lv._2, v.castTo[Int])
          (Math.add(lv._1, 1), a, lv._3.write(lv._1, a))
        },
        (Basic.constant(0), Basic.constant(0), TensorArray.create[Int](4)))
      val target = lv._3.read(lv._1 - 1)
      val gradient = Gradients.gradients(Seq(target), Seq(v), INT32).head
      assert(gradient == null)
    }
  }

  /** Returns the ops reached when going from `sourceOps` to `destinationOps`. */
  private[this] def opsBetween(sourceOps: Set[UntypedOp], destinationOps: Set[UntypedOp]): Seq[UntypedOp] = {
    val reached = mutable.Set[UntypedOp](destinationOps.toSeq: _*)
    val reachedQueue = mutable.Queue[UntypedOp](sourceOps.toSeq: _*)
    while (reachedQueue.nonEmpty) {
      val op = reachedQueue.dequeue()
      if (!reached.contains(op)) {
        reached += op
        op.outputsSeq.foreach(o => reachedQueue.enqueue(o.consumers.map(_.op): _*))
      }
    }
    // Collect all inputs of `destinationOps` that are in `reached`.
    val inputs = mutable.ArrayBuffer.empty[UntypedOp]
    reachedQueue.clear()
    reachedQueue.enqueue(destinationOps.toSeq: _*)
    while (reachedQueue.nonEmpty) {
      val op = reachedQueue.dequeue()
      if (reached.contains(op)) {
        inputs += op
        // Remove the op from `reached` so we won't add the inputs again.
        reached -= op
        op.inputsSeq.foreach(i => reachedQueue.enqueue(i.op))
      }
    }
    inputs
  }

  private[this] def assertOpSeqEqual(opSeq1: Seq[UntypedOp], opSeq2: Seq[UntypedOp]): Unit = {
    assert(opSeq1.map(_.name).toSet === opSeq2.map(_.name).toSet)
  }

  private[this] def noGradientOp(input: Output[Any], name: String = "NoGradientOp"): Output[Any] = {
    ???
  }

  private[this] def buildErrorGraph(graph: Graph): (Output[Any], Output[Any]) = {
    Op.createWith(graph) {
      val constant = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val noGradient = noGradientOp(constant)
      // TODO: Check for error or something.
      (constant, noGradient)
    }
  }

  /** Constructs the following graph:
    * {{{
    *               ^
    *               |
    *              z|
    *               |
    *             MatMul
    *         --------------
    *         ^            ^
    *         |            |
    *         |            |
    *        x|           y|
    *         |            |
    *         |            |
    *     Constant_0   Constant_1
    * }}}
    *
    * @param  graph Graph in which to place the newly constructed ops.
    * @return Tuple containing the input and output tensors, respectively.
    */
  private[this] def buildSuccessGraph(graph: Graph): (Array[Output[Double]], Output[Double]) = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      (Array[Output[Double]](constant0, constant1), matmul)
    }
  }

  /** Constructs the following graph:
    * {{{
    *     ^                    ^
    *     |                    |
    *   dy|                  dx|
    *     |                    |
    *  MatMul_2             MatMul_1
    *  --------            ---------
    *  ^      ^            ^       ^
    *  |      |            |       |
    *  |      --------------       |
    *  |            ^              |
    *  |            |              |
    *  |          dz|              |
    *  |            |              |
    *  |        Constant_2         |
    *  |            ^              |
    *  |            |              |
    *  |           z|              |
    *  |            |              |
    *  |          MatMul           |
    *  |      --------------       |
    *  |      ^            ^       |
    *  |      |            |       |
    *  |      |            |       |
    *  |     x|           y|       |
    *  |      |            |       |
    *  |      |            |       |
    *  -- Constant_0   Constant_1 --
    * }}}
    *
    * @param  graph                  Graph in which to place the newly constructed ops.
    * @param  gradientInputsProvided Boolean value indicating whether the output gradients are initialized with some
    *                                pre-existing values. If `false`, they are initialized with ones.
    * @return Array containing the gradient tensors.
    */
  private[this] def buildExpectedGraph(graph: Graph, gradientInputsProvided: Boolean): Array[Output[Any]] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      Op.nameScope("Gradients") {
        val constant2 = {
          if (gradientInputsProvided) {
            Basic.constant(Tensor(Tensor(1.0, 1.0), Tensor(1.0, 1.0)), name = "GradientInputs")
          } else {
            Op.nameScope("Gradients_0") {
              Basic.ones[Double](matmul.shape)
            }
          }
        }
        Op.nameScope("MatMulGradient") {
          val matmul1 = Math.matmul(constant2, constant1, transposeA = false, transposeB = true, name = "MatMul")
          val matmul2 = Math.matmul(constant0, constant2, transposeA = true, transposeB = false, name = "MatMul_1")
          Array[Output[Any]](matmul1, matmul2)
        }
      }
    }
  }

  /** Constructs the following graph:
    * {{{
    *     ^                    ^
    *     |                    |
    *   dy|                  dx|
    *     |                    |
    *  MatMul_2             MatMul_1
    *  --------            ---------
    *  ^      ^            ^       ^
    *  |      |            |       |
    *  |      --------------       |
    *  |            ^              |
    *  |            |              |
    *  |          dz|              |
    *  |            |              |
    *  |        Constant_2         |
    *  |            ^              |
    *  |            |              |
    *  |           z|              |
    *  |            |              |
    *  |          MatMul           |
    *  |      --------------       |
    *  |      ^            ^       |
    *  |      |            |       |
    *  |      |            |       |
    *  |     x|           y|       |
    *  |      |            |       |
    *  |      |            |       |
    *  -- Constant_0   Constant_1 --
    * }}}
    *
    * @param  graph                  Graph in which to place the newly constructed ops.
    * @param  gradientInputsProvided Boolean value indicating whether the output gradients are initialized with some
    *                                pre-existing values. If `false`, they are initialized with ones.
    * @return Array containing the gradient tensors.
    */
  private[this] def buildExpectedCCGraph(graph: Graph, gradientInputsProvided: Boolean): Array[Output[Any]] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      Op.nameScope("gradients") {
        val constant2 = {
          if (gradientInputsProvided)
            Basic.constant(Tensor(Tensor(1.0, 1.0), Tensor(1.0, 1.0)), name = "GradientInputs")
          else
            Basic.onesLike(matmul, optimize = false, name = "OnesLike")
        }
        val matmul1 = Math.matmul(constant2, constant1, transposeB = true, name = "MatMul")
        val matmul2 = Math.matmul(constant0, constant2, transposeA = true, name = "MatMul_1")
        Array[Output[Any]](matmul1, matmul2)
      }
    }
  }
}
