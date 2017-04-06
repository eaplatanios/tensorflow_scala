package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.Exception.InvalidGraphElementException
import org.platanios.tensorflow.api.ops.ArrayOps.constant
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.Op.createWith
import org.scalatest.{FlatSpec, Matchers}

/**
  * @author Emmanouil Antonios Platanios
  */
class GraphSpec extends FlatSpec with Matchers {
  private[this] def prepareGraph(): (Graph, Array[Op]) = {
    val graph = Graph()
    val ops = createWith(graph = graph) {
      val c1 = constant(1.0, name = "C_1")
      val c2 = constant(2.0, name = "C_2")
      val c3 = createWith(nameScope = "Nested") {
        constant(3.0, name = "C_3")
      }
      val c4 = constant(4.0, name = "C_4")
      Array(c1.op, c2.op, c3.op, c4.op)
    }
    (graph, ops)
  }

  "'findOp'" must "return an existing op in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.findOp("C_2").get === ops(1))
  }

  it must "return 'None' if an op name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(graph.findOp("A") === None)
  }

  "'ops'" must "return all the ops in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.ops === ops)
  }

  "'opByName'" must "return an existing op in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.opByName("C_2") === ops(1))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an op name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](graph.opByName("A")).getMessage
               === "Name 'A' refers to an op which does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](graph.opByName("A:0")).getMessage
               === "Name 'A:0' appears to refer to an op output, but 'allowOpOutput' was set to 'false'.")
  }

  "'opOutputByName'" must "return an existing op output in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.opOutputByName("C_2:0") === ops(1).output(0))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an op output name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](graph.opOutputByName("A:0:3")).getMessage
               === "Name 'A:0:3' looks a like an op output name, but it is not a valid one. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](graph.opOutputByName("A:0")).getMessage
               === "Name 'A:0' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'A', does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](graph.opOutputByName("C_2:5")).getMessage
               === "Name 'C_2:5' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'C_2', does exist in the graph, but it only has 1 output(s).")
    assert(intercept[InvalidGraphElementException](graph.opOutputByName("A")).getMessage
               === "Name 'A' looks like an (invalid) op name, and not an op output name. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](graph.opOutputByName("C_2")).getMessage
               === "Name 'C_2' appears to refer to an op, but 'allowOp' was set to 'false'.")
  }

  "'graphElementByName'" must "return an existing element in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.graphElementByName("C_2").left.get === ops(1))
    assert(graph.graphElementByName("C_2:0").right.get === ops(1).output(0))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an element name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](
      graph.graphElementByName("A", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A' refers to an op which does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](
      graph.graphElementByName("A:0:3", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A:0:3' looks a like an op output name, but it is not a valid one. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](
      graph.graphElementByName("A:0", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A:0' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'A', does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](
      graph.graphElementByName("C_2:5", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'C_2:5' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'C_2', does exist in the graph, but it only has 1 output(s).")
    assert(intercept[IllegalArgumentException](
      graph.graphElementByName("A", allowOp = false, allowOpOutput = false)).getMessage()
               === "'allowOpOutput' and 'allowOp' cannot both be set to 'false'.")
  }
}
