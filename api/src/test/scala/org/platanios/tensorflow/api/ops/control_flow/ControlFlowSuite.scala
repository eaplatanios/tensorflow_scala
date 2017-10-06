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

package org.platanios.tensorflow.api.ops.control_flow

import org.platanios.tensorflow.api.using
import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.ops.{Basic, Embedding, Logging, Math, Op, Output, OutputIndexedSlices}
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent
import org.platanios.tensorflow.api.ops.variables.{OnesInitializer, RandomNormalInitializer, Variable}
import org.platanios.tensorflow.api.tensors.Tensor
import org.scalatest.Matchers
import org.scalatest.junit.JUnitSuite
import org.junit.Test
import org.platanios.tensorflow.api.types.INT32

/**
  * @author Emmanouil Antonios Platanios
  */
class ControlFlowSuite extends JUnitSuite with Matchers {
  def withNewGraph[T](fn: => T): T = using(Graph())(graph => Op.createWith(graph)(fn))

  @Test def testSwitchWithOutput(): Unit = withNewGraph {
    val data = Basic.constant(Tensor(0, 1))
    val zero = Basic.constant(0)
    val one = Basic.constant(1)
    val less = Math.less(zero, one)
    val switch = ControlFlow.switch(data, less)
    val session = Session()
    val switchTrue = session.run(fetches = switch._2)
    session.close()
    assert(switchTrue === Tensor(0, 1))
  }

  @Test def testSwitchWithOutputIndexedSlicesWithDenseShape(): Unit = withNewGraph {
    val data = OutputIndexedSlices(Tensor(0, 1), Tensor(1, 2, 3), Tensor(3))
    val zero = Basic.constant(0)
    val one = Basic.constant(1)
    val less = Math.less(zero, one)
    val switch = ControlFlow.switch(data, less)
    val session = Session()
    val switchTrue = session.run(fetches = switch._2)
    session.close()
    assert(switchTrue.indices === Tensor(0, 1))
    assert(switchTrue.values === Tensor(1, 2, 3))
  }

  @Test def testCondWithSingleOutput(): Unit = withNewGraph {
    val p = Basic.constant(false)
    val t = () => Basic.constant(true)
    val f = () => Basic.constant(false)
    val r = ControlFlow.cond(p, t, f)
    val session = Session()
    val result = session.run(fetches = r)
    session.close()
    assert(result.scalar === false)
  }

  @Test def testCondWithOutputSequence(): Unit = withNewGraph {
    val p = Basic.constant(0) < 10
    val t = () => Seq(Basic.constant(true), Basic.constant(1))
    val f = () => Seq(Basic.constant(false), Basic.constant(0))
    val r = ControlFlow.cond(p, t, f)
    val session = Session()
    val result = session.run(fetches = r)
    session.close()
    assert(result(0).scalar === true)
    assert(result(1).scalar === 1)
  }

  @Test def testCondGradientWithSingleOutput(): Unit = withNewGraph {
    val x = Variable.getVariable("x", shape = Shape(5, 5), initializer = OnesInitializer)
    val p = Basic.constant(true)
    val t = () => (x.value * 2f).sum()
    val f = () => Basic.constant(0.0f)
    val loss = ControlFlow.cond(p, t, f)
    val optimizer = GradientDescent(0.1)
    val trainOp = optimizer.minimize(loss)
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
    val losses = (0 until 10).map(_ => session.run(fetches = loss, targets = trainOp).scalar)
    session.close()
    assert(losses.last.asInstanceOf[Float] === -40f +- 0.001f)
  }

  @Test def testWhileLoopWithSingleOutput(): Unit = withNewGraph {
    val i = Basic.constant(0)
    val p = (i: Output) => i < 10
    val b = (i: Output) => i + 1
    val r = ControlFlow.whileLoop(p, b, i, null.asInstanceOf[Shape], 1, enableBackPropagation = false)
    val session = Session()
    val result = session.run(fetches = r)
    session.close()
    assert(result.scalar === 10)
  }

  @Test def testWhileLoopGradientWithOutput(): Unit = withNewGraph {
    val x = Variable.getVariable("x", shape = Shape(5, 5), initializer = OnesInitializer)
    val p = (v: (Output, Output)) => v._1 < 5
    val b = (v: (Output, Output)) => (v._1 + 1, v._2 + (x.value * 2.0f).sum())
    val (_, loss) = ControlFlow.whileLoop(p, b, (Basic.constant(0, INT32), Basic.constant(0.0f)))
    val optimizer = GradientDescent(0.1)
    val trainOp = optimizer.minimize(loss)
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
    val losses = (0 until 10).map(_ => session.run(fetches = loss, targets = trainOp).scalar)
    session.close()
    assert(losses.last.asInstanceOf[Float] === -2000f +- 0.001f)
  }

  @Test def testWhileLoopWithOutputIndexedSlicesGradient(): Unit = withNewGraph {
    val embeddingMatrix = Variable.getVariable(
      "EmbeddingMatrix", shape = Shape(5, 5), initializer = OnesInitializer)
    val p = (v: (Output, Output)) => v._1 < 5
    val b = (v: (Output, Output)) => (v._1 + 1, v._2 + Embedding.embeddingLookup(embeddingMatrix.value * 2.0f, 0).sum())
    val (_, loss) = ControlFlow.whileLoop(p, b, (Basic.constant(0, INT32), Basic.constant(0.0f)))
    val optimizer = GradientDescent(0.1)
    val trainOp = optimizer.minimize(loss)
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
    val losses = (0 until 10).map(_ => session.run(fetches = loss, targets = trainOp).scalar)
    session.close()
    assert(losses.last.asInstanceOf[Float] === -400f +- 0.001f)
  }
}
