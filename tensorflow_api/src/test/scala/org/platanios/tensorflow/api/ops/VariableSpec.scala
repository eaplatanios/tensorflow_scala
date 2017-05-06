package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Basic.{constant, placeholder}
import org.platanios.tensorflow.api.ops.Op.createWith

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class VariableSpec extends FlatSpec with Matchers {
  "Variable creation" must "work" in {
    val graph = Graph()
    val variable = createWith(graph = graph) {
      val initializer = Variable.ConstantInitializer(Tensor(Tensor(2, 3)))
      Variable(initializer, shape = Shape(1, 2), dataType = DataType.Int64)
    }
    assert(variable.dataType === DataType.Int64)
    assert(graph.getCollection(Graph.Keys.GLOBAL_VARIABLES).contains(variable))
    assert(graph.getCollection(Graph.Keys.TRAINABLE_VARIABLES).contains(variable))
    val session = Session(graph = graph)
    session.run(Map.empty, Array.empty, Array(variable.initializer))
    val outputs = session.run(Map.empty, Array(variable.value))
    assert(outputs.length == 1)
    val expectedResult = Tensor(DataType.Int64, Tensor(2, 3))
    assert(outputs.head(0, 0).scalar === expectedResult(0, 0).scalar)
    assert(outputs.head(0, 1).scalar === expectedResult(0, 1).scalar)
    session.close()
    graph.close()
  }

  "Variable assignment" must "work" in {
    val graph = Graph()
    val (variable, variableAssignment) = createWith(graph = graph) {
      val a = constant(Tensor(Tensor(5, 7)), DataType.Int64, name = "A")
      val initializer = Variable.ConstantInitializer(Tensor(Tensor(2, 3)))
      val variable = Variable(initializer, shape = Shape(1, 2), dataType = DataType.Int64)
      val variableAssignment = variable.assign(a)
      (variable, variableAssignment)
    }
    assert(variable.dataType === DataType.Int64)
    assert(graph.getCollection(Graph.Keys.GLOBAL_VARIABLES).contains(variable))
    assert(graph.getCollection(Graph.Keys.TRAINABLE_VARIABLES).contains(variable))
    val session = Session(graph = graph)
    session.run(Map.empty, Array.empty, Array(variable.initializer))
    val outputs = session.run(Map.empty, Array(variableAssignment, variable.value))
    assert(outputs.length == 2)
    val expectedResult = Tensor(DataType.Int64, Tensor(5, 7))
    assert(outputs(1)(0, 0).scalar === expectedResult(0, 0).scalar)
    assert(outputs(1)(0, 1).scalar === expectedResult(0, 1).scalar)
    session.close()
    graph.close()
  }
}
