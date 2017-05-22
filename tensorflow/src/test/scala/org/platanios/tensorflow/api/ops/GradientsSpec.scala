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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.tf.{Graph, Tensor}

import org.scalatest._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientsSpec extends FlatSpec with Matchers {
  "'Op.gradients'" must "work when gradients are defined for the ops being used" in {
    val graph = Graph()
    val expectedGraph = Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Gradients.gradients(Array(output), inputs)
    val expectedGradients = buildExpectedGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  "'Op.cc_gradients'" must "work when gradients are defined for the ops being used" in {
    val graph = Graph()
    val expectedGraph = Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Gradients.cc_gradients(Array(output), inputs)
    val expectedGradients = buildExpectedCCGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  private[this] def noGradientOp(input: Op.Output, name: String = "NoGradientOp"): Op.Output = {
    ???
  }

  private[this] def buildErrorGraph(graph: Graph): (Op.Output, Op.Output) = {
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
  private[this] def buildSuccessGraph(graph: Graph): (Array[Op.Output], Op.Output) = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matMul = Math.matMul(constant0, constant1, name = "MatMul")
      (Array[Op.Output](constant0, constant1), matMul)
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
  private[this] def buildExpectedGraph(graph: Graph, gradientInputsProvided: Boolean): Array[Op.Output] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matMul = Math.matMul(constant0, constant1, name = "MatMul")
      Op.createWithNameScope("Gradients") {
        val constant2 = {
          if (gradientInputsProvided)
            Basic.constant(Tensor(Tensor(1.0, 1.0), Tensor(1.0, 1.0)), name = "GradientInputs")
          else
            Basic.ones(matMul.shape, matMul.dataType, name = "OnesLike")
        }
        Op.createWithNameScope("MatMulGradient") {
          val matMul1 = Math.matMul(constant2, constant1, transposeA = false, transposeB = true, name = "MatMul_1")
          val matMul2 = Math.matMul(constant0, constant2, transposeA = true, transposeB = false, name = "MatMul_2")
          Array[Op.Output](matMul1, matMul2)
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
  private[this] def buildExpectedCCGraph(graph: Graph, gradientInputsProvided: Boolean): Array[Op.Output] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(Tensor(Tensor(1.0, 2.0), Tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(Tensor(Tensor(1.0, 0.0), Tensor(0.0, 1.0)), name = "Constant_1")
      val matMul = Math.matMul(constant0, constant1, name = "MatMul")
      val constant2 = {
        if (gradientInputsProvided)
          Basic.constant(Tensor(Tensor(1.0, 1.0), Tensor(1.0, 1.0)), name = "GradientInputs")
        else
          Basic.onesLike(matMul, optimize = false, name = "OnesLike")
      }
      val matMul1 = Math.matMul(constant2, constant1, transposeA = false, transposeB = true, name = "MatMul_1")
      val matMul2 = Math.matMul(constant0, constant2, transposeA = true, transposeB = false, name = "MatMul_2")
      Array[Op.Output](matMul1, matMul2)
    }
  }
}

object GradientsSpec {
  /** Gathers and returns all inputs of `destinations` (recursively) that have been reached.
    *
    * @param  destinations Ops whose inputs are being gathered.
    * @param  reached      Reached ops.
    * @return Set of input ops to `destinations` (recursively) that have been reached.
    */
  private[this] def gatherInputs(destinations: Set[Op], reached: mutable.Set[Op]): Set[Op] = {
    val inputs = mutable.Set.empty[Op]
    val queue = mutable.Queue[Op](destinations.toSeq: _*)
    while (queue.nonEmpty) {
      val op = queue.dequeue()
      if (reached.contains(op)) {
        inputs += op
        reached -= op // Done so we don't go through the same ops twice
        op.inputs.foreach(i => queue.enqueue(i.op))
      }
    }
    inputs.toSet
  }
}
