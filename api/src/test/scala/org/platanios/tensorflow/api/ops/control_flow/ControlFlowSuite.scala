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
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, OnesInitializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{FLOAT32, FLOAT64, INT32}

import com.google.protobuf.TextFormat
import org.scalatest.Matchers
import org.scalatest.junit.JUnitSuite
import org.junit.Test
import org.tensorflow.framework.{CondContextDef, GraphDef, NodeDef, WhileContextDef}

import scala.collection.JavaConverters._
import scala.collection.mutable

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
    ControlFlow.group(Set(a.op, b.op, c.op), name = "root")
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
    ControlFlow.group(Set(a.op, b.op), name = "root")
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
    Op.createWith(device = "/task:2") {
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

  @Test def testGradientThroughSingleBranchOutsideOfContext(): Unit = withNewGraph {
    val p = Basic.constant(true)
    val x = Basic.constant(2.0)
    val (xFalse, xTrue) = ControlFlow.switch(x, p)
    val xFalseGradient = Gradients.gradients(Seq(xFalse), Seq(x))(0).toOutput
    val xTrueGradient = Gradients.gradients(Seq(xTrue), Seq(x))(0).toOutput
    val session = Session()
    val (xFG, xTG) = session.run(fetches = (xFalseGradient, xTrueGradient))
    session.close()
    assert(xFG.scalar === 0.0)
    assert(xTG.scalar === 1.0)
  }

  //endregion switch

  //region Control Flow Context

  @Test def testControlFlowContextDef(): Unit = withNewGraph {
    Basic.constant(0, name = "A")
    Basic.constant(2, name = "TestScope/A")
    val b1 = Basic.constant(1, name = "B")
    val b2 = Basic.constant(3, name = "TestScope/B")
    val values = mutable.Set("A", "B")
    val externalValues = mutable.Map("A" -> b1)
    val toProto = Context.toValuesDef(values, externalValues)
    val importedWithScope = Context.fromValuesDef(toProto, importScope = "TestScope")
    assert(importedWithScope._1 == Set("TestScope/A", "TestScope/B"))
    assert(importedWithScope._2 == Map("TestScope/A" -> b2))
    val fromProto = Context.toValuesDef(importedWithScope._1, importedWithScope._2, "TestScope")
    assert(toProto.getValuesList.asScala.toSet === fromProto.getValuesList.asScala.toSet)
    assert(toProto.getExternalValuesMap.asScala === fromProto.getExternalValuesMap.asScala)
  }

  @Test def testCondContextDef(): Unit = withNewGraph {
    val x = Basic.constant(2)
    val y = Basic.constant(5)
    ControlFlow.cond(Math.less(x, y), () => Math.multiply(x, 17), () => Math.add(y, 23))
    Op.currentGraph.ops.foreach(op => {
      op.controlFlowContext.foreach(c => {
        val toProto = c.toProto
        val fromProto =
          CondContextDef
              .newBuilder(CondContext.fromCondContextDef(toProto.asInstanceOf[CondContextDef]).toCondContextDef())
              .setContextName(toProto.asInstanceOf[CondContextDef].getContextName)
              .build()
        assert(toProto.toString === fromProto.toString)
      })
    })
  }

  @Test def testWhileLoopContextDef(): Unit = withNewGraph {
    val i = Basic.constant(0)
    val c = (i: Output) => Math.less(i, 10)
    val b = (i: Output) => Math.add(i, 1)
    ControlFlow.whileLoop(c, b, i)
    Op.currentGraph.ops.foreach(op => {
      op.controlFlowContext.foreach(c => {
        val toProto = c.toProto.asInstanceOf[WhileContextDef]
        val fromProto = WhileLoopContext.fromWhileContextDef(toProto).toProto.asInstanceOf[WhileContextDef]
        assert(toProto.getBackProp === fromProto.getBackProp)
        assert(toProto.getSwapMemory === fromProto.getSwapMemory)
        assert(toProto.getParallelIterations === fromProto.getParallelIterations)
        assert(toProto.getPivotName === fromProto.getPivotName)
        assert(toProto.getPivotForBodyName === fromProto.getPivotForBodyName)
        assert(toProto.getPivotForPredName === fromProto.getPivotForPredName)
        assert(toProto.getLoopEnterNamesList.asScala.toSet === fromProto.getLoopEnterNamesList.asScala.toSet)
        assert(toProto.getLoopExitNamesList.asScala.toSet === fromProto.getLoopExitNamesList.asScala.toSet)
        val toProtoValuesDef = toProto.getValuesDef
        val fromProtoValuesDef = fromProto.getValuesDef
        assert(toProtoValuesDef.getValuesList.asScala.toSet === fromProtoValuesDef.getValuesList.asScala.toSet)
        assert(toProtoValuesDef.getExternalValuesMap.asScala === fromProtoValuesDef.getExternalValuesMap.asScala)
      })
    })
  }

  //endregion Control Flow Context

  //region cond

  @Test def testCondTrue(): Unit = withNewGraph {
    val x = Basic.constant(2)
    val y = Basic.constant(5)
    val z = ControlFlow.cond(Math.less(x, y), () => Math.multiply(x, 17), () => Math.add(y, 23))
    val session = Session()
    val result = session.run(fetches = z)
    session.close()
    assert(result.scalar === 34)
  }

  @Test def testCondFalse(): Unit = withNewGraph {
    val x = Basic.constant(2)
    val y = Basic.constant(1)
    val z = ControlFlow.cond(Math.less(x, y), () => Math.multiply(x, 17), () => Math.add(y, 23))
    val session = Session()
    val result = session.run(fetches = z)
    session.close()
    assert(result.scalar === 24)
  }

  @Test def testCondModifyPredicate(): Unit = withNewGraph {
    val x = Variable.getVariable("Predicate", initializer = ConstantInitializer(true))
    val c = ControlFlow.cond(x, () => x.assign(false), () => Basic.constant(true))
    val session = Session()
    session.run(targets = x.initializer)
    assert(session.run(fetches = c).scalar === false)
    assert(session.run(fetches = c).scalar === true)
    session.close()
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
      "EmbeddingMatrix", initializer = ConstantInitializer(Tensor(Tensor(2.0f), Tensor(3.0f))))
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
    val embeddingMatrix = Variable.getVariable("EmbeddingMatrix", shape = Shape(5, 5), initializer = OnesInitializer)
    val p = (v: (Output, Output)) => v._1 < 5
    val b = (v: (Output, Output)) => (v._1 + 1, v._2 + 2.0f * Embedding.embeddingLookup(embeddingMatrix, 0).sum())
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
          () => l + Embedding.embeddingLookup(embeddingMatrix, 0).sum())
        (nextI, nextL)
    }
    val (_, loss) = ControlFlow.whileLoop(p, b, (Basic.constant(0, INT32), Basic.constant(0.0f)))
    val dynamicGradients = Gradients.gradients(Seq(loss), Seq(embeddingMatrix.handle))(0).toOutput
    val embedding = Embedding.embeddingLookup(embeddingMatrix, 0)
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

  @Test def testWhileLoopWithOutputIndexedSlicesWithStaticShapeGradient(): Unit = {
    val numIterations = 9
    for (dataType <- Set(FLOAT32, FLOAT64)) {
      withNewGraph {
        val inputs = Basic.placeholder(dataType, Shape(numIterations))
        val initialI = Basic.constant(0, INT32)
        val initialOutputs = TensorArray.create(numIterations, dataType)
        val p = (v: (Output, TensorArray)) => v._1 < numIterations
        val b = (v: (Output, TensorArray)) => v match {
          case (i, o) => (i + 1, o.write(i, Basic.gather(inputs, i)))
        }
        val (_, outputs) = ControlFlow.whileLoop(p, b, (initialI, initialOutputs))
        val outputsSum = outputs.stack().sum()
        val gradients = Gradients.gradients(Seq(outputsSum), Seq(inputs))(0).toOutput
        val session = Session()
        val (os, g) = session.run(
          feeds = inputs -> Tensor(dataType, 4, 6, 0, 7, 0, 0, 1, 2, 0),
          fetches = (outputsSum, gradients))
        assert(os.scalar === 20)
        assert(g === Tensor.ones(dataType, Shape(numIterations)))
      }
    }
  }

  @Test def testWhileLoopWithOutputIndexedSlicesWithDynamicShapeGradient(): Unit = {
    for (dataType <- Set(FLOAT32, FLOAT64)) {
      withNewGraph {
        val inputs = Basic.placeholder(dataType)
        val initialI = Basic.constant(0, INT32)
        val initialOutputs = TensorArray.create(1, dataType, dynamicSize = true)
        val p = (v: (Output, TensorArray)) => v._1 < Basic.size(inputs)
        val b = (v: (Output, TensorArray)) => v match {
          case (i, o) => (i + 1, o.write(i, Basic.gather(inputs, i)))
        }
        val (_, outputs) = ControlFlow.whileLoop(p, b, (initialI, initialOutputs))
        val outputsSum = outputs.stack().sum()
        val gradients = Gradients.gradients(Seq(outputsSum), Seq(inputs))(0).toOutput
        val session = Session()
        val (os, g) = session.run(
          feeds = inputs -> Tensor(dataType, 1, 2, 3),
          fetches = (outputsSum, gradients))
        assert(os.scalar === 6)
        assert(g === Tensor.ones(dataType, Shape(3)))
      }
    }
  }

  //endregion whileLoop
}
