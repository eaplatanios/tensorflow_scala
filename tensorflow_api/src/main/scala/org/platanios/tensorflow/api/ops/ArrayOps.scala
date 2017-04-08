package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.{DataType, Shape, Tensor, using}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
object ArrayOps {
  def constant(value: Any, dataType: DataType[_] = null, name: String = "Constant")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    using(Tensor.create(value = value)) { tensor =>
      val opBuilder = Op.Builder(context = context, opType = "Const", name = name)
      opBuilder.setAttribute(name = "value", value = tensor)
      if (dataType != null)
        opBuilder.setAttribute(name = "dtype", value = dataType)
      else
        opBuilder.setAttribute(name = "dtype", value = tensor.dataType)
      opBuilder.build().outputs(0)
    }
  }

  def placeholder(dataType: DataType[_], shape: Option[Shape] = None, name: String = "Placeholder")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    shape match {
      case Some(shapeValue) =>
        Op.Builder(context = context, opType = "PlaceholderV2", name = name)
            .setAttribute(name = "dtype", value = dataType)
            .setAttribute(name = "shape", value = shapeValue)
            .build()
            .outputs(0)
      case None =>
        Op.Builder(context = context, opType = "Placeholder", name = name)
            .setAttribute(name = "dtype", value = dataType)
            .build()
            .outputs(0)
    }
  }

  def placeholderWithDefault(value: Any, shape: Shape, name: String = "PlaceholderWithDefault")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    val default: Op.Output = constant(value = value, name = s"$name/DefaultValue")
    Op.Builder(context = context, opType = "PlaceholderWithDefault", name = name)
        .addInput(default)
        .setAttribute(name = "shape", value = shape)
        .build()
        .outputs(0)
  }
}
