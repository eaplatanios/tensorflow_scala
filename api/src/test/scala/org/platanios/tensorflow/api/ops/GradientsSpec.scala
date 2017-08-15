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

import org.platanios.tensorflow.api._

import org.scalatest._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientsSpec extends FlatSpec with Matchers {
  "'Op.gradients'" must "work when gradients are defined for the ops being used" in {
    val graph = tf.Graph()
    val expectedGraph = tf.Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Gradients.gradients(Array(output), inputs)
    val expectedGradients = buildExpectedGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = tf.Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  "'Op.cc_gradients'" must "work when gradients are defined for the ops being used" in {
    val graph = tf.Graph()
    val expectedGraph = tf.Graph()
    val (inputs, output) = buildSuccessGraph(graph)
    val gradients = Gradients.cc_gradients(Array(output), inputs)
    val expectedGradients = buildExpectedCCGraph(expectedGraph, gradientInputsProvided = false)
    val graphDef = graph.toProto
    val expectedGraphDef = expectedGraph.toProto
    val (equal, difference) = tf.Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  private[this] def noGradientOp(input: Output, name: String = "NoGradientOp"): Output = {
    ???
  }

  private[this] def buildErrorGraph(graph: tf.Graph): (Output, Output) = {
    Op.createWith(graph) {
      val constant = tf.constant(tf.tensor(tf.tensor(1.0, 2.0), tf.tensor(3.0, 4.0)), name = "Constant_0")
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
  private[this] def buildSuccessGraph(graph: tf.Graph): (Array[Output], Output) = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(tf.tensor(tf.tensor(1.0, 2.0), tf.tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(tf.tensor(tf.tensor(1.0, 0.0), tf.tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      (Array[Output](constant0, constant1), matmul)
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
  private[this] def buildExpectedGraph(graph: tf.Graph, gradientInputsProvided: Boolean): Array[Output] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(tf.tensor(tf.tensor(1.0, 2.0), tf.tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(tf.tensor(tf.tensor(1.0, 0.0), tf.tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      Op.createWithNameScope("Gradients") {
        val constant2 = {
          if (gradientInputsProvided)
            Basic.constant(tf.tensor(tf.tensor(1.0, 1.0), tf.tensor(1.0, 1.0)), name = "GradientInputs")
          else
            Basic.ones(matmul.shape, matmul.dataType, name = "OnesLike")
        }
        Op.createWithNameScope("MatMulGradient") {
          val matmul1 = Math.matmul(constant2, constant1, transposeA = false, transposeB = true, name = "MatMul")
          val matmul2 = Math.matmul(constant0, constant2, transposeA = true, transposeB = false, name = "MatMul_1")
          Array[Output](matmul1, matmul2)
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
  private[this] def buildExpectedCCGraph(graph: tf.Graph, gradientInputsProvided: Boolean): Array[Output] = {
    Op.createWith(graph) {
      val constant0 = Basic.constant(tf.tensor(tf.tensor(1.0, 2.0), tf.tensor(3.0, 4.0)), name = "Constant_0")
      val constant1 = Basic.constant(tf.tensor(tf.tensor(1.0, 0.0), tf.tensor(0.0, 1.0)), name = "Constant_1")
      val matmul = Math.matmul(constant0, constant1, name = "MatMul")
      Op.createWithNameScope("gradients") {
        val constant2 = {
          if (gradientInputsProvided)
            Basic.constant(tf.tensor(tf.tensor(1.0, 1.0), tf.tensor(1.0, 1.0)), name = "GradientInputs")
          else
            Basic.onesLike(matmul, optimize = false, name = "OnesLike")
        }
        val matmul1 = Math.matmul(constant2, constant1, transposeB = true, name = "MatMul")
        val matmul2 = Math.matmul(constant0, constant2, transposeA = true, name = "MatMul_1")
        Array[Output](matmul1, matmul2)
      }
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
