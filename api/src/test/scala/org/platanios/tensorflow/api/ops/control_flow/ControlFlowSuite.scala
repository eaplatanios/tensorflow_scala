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
import org.platanios.tensorflow.api.ops.{Basic, Embedding, Gradients, Logging, Math, Op, Output, OutputIndexedSlices}
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, OnesInitializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.INT32
import com.google.protobuf.TextFormat
import org.scalatest.Matchers
import org.scalatest.junit.JUnitSuite
import org.junit.Test
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.JavaConverters._

/**
  * @author Emmanouil Antonios Platanios
  */
class ControlFlowSuite extends JUnitSuite with Matchers {
  private[this] def withNewGraph[T](fn: => T): T = using(Graph())(graph => Op.createWith(graph)(fn))

  private[this] def stripNodeDef(nodeDef: NodeDef): NodeDef = {
    val nodeDefBuilder = NodeDef.newBuilder()
    nodeDefBuilder.setName(nodeDef.getName)
    nodeDefBuilder.setOp(nodeDef.getOp)
    nodeDefBuilder.addAllInput(nodeDef.getInputList)
    if (nodeDef.getDevice != null)
      nodeDefBuilder.setDevice(nodeDef.getDevice)
    nodeDefBuilder.build()
  }

  /** Copies the provided `GraphDef` keeping only the node names, ops, inputs, and devices. */
  private[this] def stripGraphDef(graphDef: GraphDef): GraphDef = {
    GraphDef.newBuilder().addAllNode(graphDef.getNodeList.asScala.map(stripNodeDef).asJava).build()
  }

  //region withDependencies

  @Test def testWithDependencies(): Unit = withNewGraph {
    val cnt = Variable.getVariable("cnt", INT32, shape = Shape(), initializer = ZerosInitializer)
    val incrementCnt = cnt.assignAdd(1)
    val constWithDependencies = ControlFlow.withControlDependencies(
      Set(incrementCnt.op, Basic.constant(42).op), Basic.constant(7))
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
    assert(session.run(fetches = cnt.value).scalar === 0)
    assert(session.run(fetches = constWithDependencies).scalar === 7)
    assert(session.run(fetches = cnt.value).scalar === 1)
  }

  @Test def testWithDependenciesShapeInference(): Unit = withNewGraph {
    val t = Basic.constant(Tensor(1.0, 2.0))
    assert(Shape(2) === t.shape)
    assert(Shape(2) === ControlFlow.withControlDependencies(Set(Basic.constant(1.0)), t).shape)
  }

  //endregion withDependencies

  //region group

  @Test def testGroupNoDevices(): Unit = withNewGraph {
    val a = Basic.constant(0, name = "a")
    val b = Basic.constant(0, name = "b")
    val c = Basic.constant(0, name = "c")
    val _ = ControlFlow.group(Set(a.op, b.op, c.op), name = "root")
    val graphDef = stripGraphDef(Op.currentGraph.toGraphDef)
    val expectedGraphDefBuilder = GraphDef.newBuilder()
    TextFormat.merge(
      """
        |node { name: "a" op: "Const" }
        |node { name: "b" op: "Const" }
        |node { name: "c" op: "Const" }
        |node { name: "root" op: "NoOp" input: "^a" input: "^b" input: "^c" }
      """.stripMargin, expectedGraphDefBuilder)
    val expectedGraphDef = expectedGraphDefBuilder.build()
    val (equal, _) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  @Test def testGroupOneDevice(): Unit = withNewGraph {
    val (a, b) = Op.createWith(device = "/task:0") {
      val a = Basic.constant(0, name = "a")
      val b = Basic.constant(0, name = "b")
      (a, b)
    }
    val _ = ControlFlow.group(Set(a.op, b.op), name = "root")
    val graphDef = stripGraphDef(Op.currentGraph.toGraphDef)
    val expectedGraphDefBuilder = GraphDef.newBuilder()
    TextFormat.merge(
      """
        |node { name: "a" op: "Const" device: "/task:0" }
        |node { name: "b" op: "Const" device: "/task:0" }
        |node { name: "root" op: "NoOp" input: "^a" input: "^b" device: "/task:0" }
      """.stripMargin, expectedGraphDefBuilder)
    val expectedGraphDef = expectedGraphDefBuilder.build()
    val (equal, _) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  @Test def testGroupMultipleDevices(): Unit = withNewGraph {
    val (a, b) = Op.createWith(device = "/task:0") {
      val a = Basic.constant(0, name = "a")
      val b = Basic.constant(0, name = "b")
      (a, b)
    }
    val (c, d) = Op.createWith(device = "/task:1") {
      val c = Basic.constant(0, name = "c")
      val d = Basic.constant(0, name = "d")
      (c, d)
    }
    val _ = Op.createWith(device = "/task:2") {
      ControlFlow.group(Set(a.op, b.op, c.op, d.op), name = "root")
    }
    val graphDef = stripGraphDef(Op.currentGraph.toGraphDef)
    val expectedGraphDefBuilder = GraphDef.newBuilder()
    TextFormat.merge(
      """
        |node { name: "a" op: "Const" device: "/task:0" }
        |node { name: "b" op: "Const" device: "/task:0" }
        |node { name: "c" op: "Const" device: "/task:1" }
        |node { name: "d" op: "Const" device: "/task:1" }
        |node { name: "root" op: "NoOp" input: "^a" input: "^b" device: "/task:0" }
        |node { name: "root_1" op: "NoOp" input: "^c" input: "^d" device: "/task:1" }
        |node { name: "root_2" op: "NoOp" input: "^root" input: "^root_1" device: "/task:2" }
      """.stripMargin, expectedGraphDefBuilder)
    val expectedGraphDef = expectedGraphDefBuilder.build()
    val (equal, _) = Graph.equalGraphDef(graphDef, expectedGraphDef)
    assert(equal)
  }

  //endregion group

  //region switch

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

  //endregion switch

  //region cond

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

  //endregion cond

  //region whileLoop

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

  @Test def testWhileLoopResourceRead(): Unit = withNewGraph {
    val embeddingMatrix = Variable.getVariable(
      "EmbeddingMatrix", initializer = ConstantInitializer(Tensor(Tensor(2.0), Tensor(3.0))))
    val p = (v: (Output, Output)) => v._1 < 5
    val b = (v: (Output, Output)) => (v._1 + 1, v._2 + Embedding.embeddingLookup(embeddingMatrix.value, 0).sum())
    val (_, r) = ControlFlow.whileLoop(p, b, (Basic.constant(0, INT32), Basic.constant(0.0f)))
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
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

  @Test def testWhileLoopWithNestedCondWithOutputIndexedSlicesGradient(): Unit = withNewGraph {
    val embeddingMatrix = Variable.getVariable(
      "EmbeddingMatrix", shape = Shape(5, 5), initializer = OnesInitializer)
    val p = (v: (Output, Output)) => v._1 < 5
    val b = (v: (Output, Output)) => v match {
      case (i, l) =>
        val nextI = i + 1
        val nextL = ControlFlow.cond(
          Math.equal(i, 3),
          () => Math.square(l),
          () => l + Embedding.embeddingLookup(embeddingMatrix.value, 0).sum())
        (nextI, nextL)
    }
    val (_, loss) = ControlFlow.whileLoop(p, b, (Basic.constant(0, INT32), Basic.constant(0.0f)))
    val dynamicGradients = Gradients.gradients(Seq(loss), Seq(embeddingMatrix.handle), gateGradients = true)(0).toOutput
    val embedding = Embedding.embeddingLookup(embeddingMatrix.value, 0)
    val embeddingSum = embedding.sum()
    val staticLoss = (3 * embeddingSum).square + embeddingSum
    val staticGradients = Gradients.gradients(Seq(staticLoss), Seq(embeddingMatrix.handle))(0).toOutput
    val session = Session()
    session.run(targets = Op.currentGraph.globalVariablesInitializer())
    val dG = session.run(fetches = dynamicGradients)
    val sG = session.run(fetches = staticGradients)
    session.close()
    assert(dG === sG)
  }

  //endregion whileLoop
}
