package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.{DataType, Shape, Tensor, using}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
object ArrayOps {
  def constant(value: Any, dataType: DataType[_] = null, name: String = "Const")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    val opName: String = Op.name(context = context, providedName = name)
    using(Tensor.create(value = value)) { tensor =>
      val opBuilder: Op.Builder = Op.Builder(graph = context.graph, opType = "Const", name = opName)
      opBuilder.setAttribute(name = "value", value = tensor)
      if (dataType != null)
        opBuilder.setAttribute(name = "dtype", value = dataType)
      else
        opBuilder.setAttribute(name = "dtype", value = tensor.dataType)
      opBuilder.build().output(0)
    }
  }

  def placeholder(dataType: DataType[_], shape: Option[Shape] = None, name: String = "Placeholder")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    val opName: String = Op.name(context = context, providedName = name)
    shape match {
      case Some(shapeValue) =>
        Op.Builder(graph = context.graph, opType = "PlaceholderV2", name = opName)
            .setAttribute(name = "dtype", value = dataType)
            .setAttribute(name = "shape", value = shapeValue)
            .build()
            .output(index = 0)
      case None =>
        Op.Builder(graph = context.graph, opType = "Placeholder", name = opName)
            .setAttribute(name = "dtype", value = dataType)
            .build()
            .output(index = 0)
    }
  }

  def placeholderWithDefault(value: Any, shape: Shape, name: String = "PlaceholderWithDefault")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    val opName: String = Op.name(context = context, providedName = name)
    Op.Builder(graph = context.graph, opType = "PlaceholderWithDefault", name = opName)
        .addInput(constant(value = value, name = s"$opName/Const"))
        .setAttribute(name = "shape", value = shape)
        .build()
        .output(index = 0)
  }
}
