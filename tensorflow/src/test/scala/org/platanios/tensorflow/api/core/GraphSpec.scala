package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.exception.{GraphMismatchException, InvalidGraphElementException}
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.Op.createWith
import org.platanios.tensorflow.api.ops.Basic.{constant, placeholder}
import org.platanios.tensorflow.api.ops.Math.add
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.FLOAT32

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class GraphSpec extends FlatSpec with Matchers {
  private[this] def prepareGraph(): (Graph, Array[Op]) = {
    val graph = Graph()
    val ops = createWith(graph = graph) {
      val c1 = constant(Tensor(1.0), name = "C_1")
      val c2 = constant(Tensor(2.0), name = "C_2")
      val c3 = createWith(nameScope = "Nested") {
        constant(Tensor(3.0), name = "C_3")
      }
      val c4 = constant(Tensor(4.0), name = "C_4")
      Array(c1.op, c2.op, c3.op, c4.op)
    }
    (graph, ops)
  }

  // TODO: Add collections specification.

  "'preventFeeding'" must "prevent valid ops from being fetched" in {
    val (graph, ops) = prepareGraph()
    assert(graph.isFeedable(ops(0).outputs(0)))
    assert(graph.isFeedable(ops(1).outputs(0)))
    assert(graph.isFeedable(ops(2).outputs(0)))
    assert(graph.isFeedable(ops(3).outputs(0)))
    graph.preventFeeding(ops(0).outputs(0))
    assert(!graph.isFeedable(ops(0).outputs(0)))
    assert(graph.isFeedable(ops(1).outputs(0)))
    assert(graph.isFeedable(ops(2).outputs(0)))
    assert(graph.isFeedable(ops(3).outputs(0)))
    graph.preventFeeding(ops(2).outputs(0))
    assert(!graph.isFeedable(ops(0).outputs(0)))
    assert(graph.isFeedable(ops(1).outputs(0)))
    assert(!graph.isFeedable(ops(2).outputs(0)))
    assert(graph.isFeedable(ops(3).outputs(0)))
  }

  it must "throw a 'GraphMismatchException' when provided ops from other graphs" in {
    val (graph, ops) = prepareGraph()
    createWith(graph = Graph()) {
      assert(intercept[GraphMismatchException](graph.isFeedable(constant(1.0))).getMessage ===
                 "The provided op output does not belong to this graph.")
      assert(intercept[GraphMismatchException](graph.preventFeeding(constant(1.0))).getMessage ===
                 "The provided op output does not belong to this graph.")
    }
  }

  "'preventFetching'" must "prevent valid ops from being fetched" in {
    val (graph, ops) = prepareGraph()
    assert(graph.isFetchable(ops(0).outputs(0)))
    assert(graph.isFetchable(ops(1).outputs(0)))
    assert(graph.isFetchable(ops(2).outputs(0)))
    assert(graph.isFetchable(ops(3).outputs(0)))
    graph.preventFetching(ops(0).outputs(0))
    assert(!graph.isFetchable(ops(0).outputs(0)))
    assert(graph.isFetchable(ops(1).outputs(0)))
    assert(graph.isFetchable(ops(2).outputs(0)))
    assert(graph.isFetchable(ops(3).outputs(0)))
    graph.preventFetching(ops(2).outputs(0))
    assert(!graph.isFetchable(ops(0).outputs(0)))
    assert(graph.isFetchable(ops(1).outputs(0)))
    assert(!graph.isFetchable(ops(2).outputs(0)))
    assert(graph.isFetchable(ops(3).outputs(0)))
  }

  it must "throw a 'GraphMismatchException' when provided ops from other graphs" in {
    val (graph, ops) = prepareGraph()
    createWith(graph = Graph()) {
      assert(intercept[GraphMismatchException](graph.isFetchable(constant(1.0))).getMessage ===
                 "The provided op output does not belong to this graph.")
      assert(intercept[GraphMismatchException](graph.preventFetching(constant(1.0))).getMessage ===
                 "The provided op output does not belong to this graph.")
    }
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
    assert(graph.getOpByName("C_2") === ops(1))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an op name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](graph.getOpByName("A")).getMessage
               === "Name 'A' refers to an op which does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](graph.getOpByName("A:0")).getMessage
               === "Name 'A:0' appears to refer to an op output, but 'allowOpOutput' was set to 'false'.")
  }

  "'opOutputByName'" must "return an existing op output in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.getOpOutputByName("C_2:0") === ops(1).outputs(0))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an op output name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](graph.getOpOutputByName("A:0:3")).getMessage
               === "Name 'A:0:3' looks a like an op output name, but it is not a valid one. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](graph.getOpOutputByName("A:0")).getMessage
               === "Name 'A:0' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'A', does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](graph.getOpOutputByName("C_2:5")).getMessage
               === "Name 'C_2:5' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'C_2', does exist in the graph, but it only has 1 output(s).")
    assert(intercept[InvalidGraphElementException](graph.getOpOutputByName("A")).getMessage
               === "Name 'A' looks like an (invalid) op name, and not an op output name. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](graph.getOpOutputByName("C_2")).getMessage
               === "Name 'C_2' appears to refer to an op, but 'allowOp' was set to 'false'.")
  }

  "'graphElementByName'" must "return an existing element in a graph" in {
    val (graph, ops) = prepareGraph()
    assert(graph.getByName("C_2").left.get === ops(1))
    assert(graph.getByName("C_2:0").right.get === ops(1).outputs(0))
  }

  it must "throw an 'InvalidGraphElementException' exception with an informative message " +
      "if an element name does not exist in the graph" in {
    val (graph, _) = prepareGraph()
    assert(intercept[InvalidGraphElementException](
      graph.getByName("A", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A' refers to an op which does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](
      graph.getByName("A:0:3", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A:0:3' looks a like an op output name, but it is not a valid one. " +
        "Op output names must be of the form \"<op_name>:<output_index>\".")
    assert(intercept[InvalidGraphElementException](
      graph.getByName("A:0", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'A:0' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'A', does not exist in the graph.")
    assert(intercept[InvalidGraphElementException](
      graph.getByName("C_2:5", allowOp = true, allowOpOutput = true)).getMessage
               === "Name 'C_2:5' refers to an op output which does not exist in the graph. " +
        "More specifically, op, 'C_2', does exist in the graph, but it only has 1 output(s).")
    assert(intercept[IllegalArgumentException](
      graph.getByName("A", allowOp = false, allowOpOutput = false)).getMessage
               === "'allowOpOutput' and 'allowOp' cannot both be set to 'false'.")
  }

  object INPUTS extends Graph.Keys.OpOutputCollectionKey {
    override def name: String = "inputs"
  }

  object OUTPUTS extends Graph.Keys.OpOutputCollectionKey {
    override def name: String = "outputs"
  }

  "'Graph.toMetaGraphDef'" must "work when no scope is provided" in {
    val graph = Graph()
    val session = Session(graph)

    Op.createWith(graph) {
      // Create a minimal graph with zero variables.
      val input = placeholder(FLOAT32, Shape(), name = "Input")
      val offset = constant(42, FLOAT32, name = "Offset")
      val output = add(input, offset, name = "AddOffset")

      // Add input and output tensors to graph collections.
      graph.addToCollection(input, INPUTS)
      graph.addToCollection(output, OUTPUTS)

      val outputValue = session.run(fetches = Array(output), feeds = Map(input -> -10f))(0)
      assert(outputValue.scalar === 32)
    }

    // Generate the 'MetaGraphDef' object.
    val metaGraphDef = graph.toMetaGraphDef(collections = Set(INPUTS, OUTPUTS))
    assert(metaGraphDef.hasMetaInfoDef)
    assert(metaGraphDef.getMetaInfoDef.getTensorflowVersion !== "")
    // assert(metaGraphDef.getMetaInfoDef.getTensorflowGitVersion !== "")

    session.close()

    // Create a clean graph and import the 'MetaGraphDef' object.
    val newGraph = Graph()
    val newSession = Session(newGraph)

    newGraph.importMetaGraphDef(metaGraphDef)

    // Re-exports the current graph state for comparison to the original.
    val newMetaGraphDef = newGraph.toMetaGraphDef()
    // TODO: [PROTO] Utility functions for ProtoBuf comparisons.
    // assert(newMetaGraphDef.equals(metaGraphDef))

    // Ensure that we can still get a reference to our graph collections.
    val newInput = newGraph.getCollection(INPUTS).head
    val newOutput = newGraph.getCollection(OUTPUTS).head

    // Verify that the new graph computes the same result as the original.
    val newOutputValue = newSession.run(fetches = Array(newOutput), feeds = Map(newInput -> -10f))(0)
    assert(newOutputValue.scalar === 32)

    newSession.close()
  }
}
