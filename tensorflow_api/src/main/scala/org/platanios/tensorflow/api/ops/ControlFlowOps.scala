package org.platanios.tensorflow.api.ops

import scala.reflect.runtime.universe._

/**
  * @author Emmanouil Antonios Platanios
  */
object ControlFlowOps {
  /** Creates an op that makes its input available to the next iteration.
    *
    * @param  input Tensor to make available to the next iteration.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def nextIteration[T <: Op.OutputLike](input: T, name: String = "NextIteration"): T = {
    Op.createWithNameScope(nameScope = name, Array(input.op)) {
      // @formatter:off
      input match {
        case i: Op.Output =>
          Op.Builder(opType = "NextIteration", name = name)
              .addInput(i)
              .build().outputs(0)
        case i: Op.OutputIndexedSlices =>
          val values = nextIteration(i.values, name = "ValuesNextIteration")
          val indices = nextIteration(i.indices, name = "IndicesNextIteration")
          val denseShape = {
            if (i.denseShape ne null)
              nextIteration(i.denseShape, name = "DenseShapeNextIteration")
            else
              null
          }
          Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: Op.SparseOutput =>
          val values = nextIteration(i.values, name = "ValuesNextIteration")
          val indices = nextIteration(i.indices, name = "IndicesNextIteration")
          val denseShape = nextIteration(i.denseShape, name = "DenseShapeNextIteration")
          Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
      // @formatter:on
    }.asInstanceOf[T]
  }

  /** Creates an op that creates or finds a child frame, and makes `input` available to that child frame.
    *
    * The op is used together with `exit` to create loops in the graph. The unique `frameName` is used by the `Executor`
    * to identify frames. If `isConstant` is `true`, then the output is a constant in the child frame. Otherwise, it may
    * be changed in the child frame. At most `parallelIterations` iterations are run in parallel in the child frame.
    *
    * @param  input              Tensor to be made available to the child frame.
    * @param  frameName          Name of the child frame.
    * @param  isConstant         If `true`, the output is constant within the child frame.
    * @param  parallelIterations Number of iterations allowed to run in parallel.
    * @param  useInputShape      If `true`, the output tensor's shape is manually set to the input tensor's shape.
    * @param  name               Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def enter[T <: Op.OutputLike](
      input: T, frameName: String, isConstant: Boolean = false, parallelIterations: Int = 10,
      useInputShape: Boolean = true, name: String = "Enter"): T = {
    Op.createWithNameScope(nameScope = name, Array(input.op)) {
      // @formatter:off
      input match {
        case i: Op.Output  =>
          val result = Op.Builder(opType = "NextIteration", name = name)
              .addInput(i)
              .build().outputs(0)
          if (useInputShape)
            result.setShape(i.shape)
          result
        case i: Op.OutputIndexedSlices =>
          val values = enter(i.values, frameName, isConstant, parallelIterations, useInputShape, "ValuesEnter")
          val indices = enter(i.indices, frameName, isConstant, parallelIterations, useInputShape, "IndicesEnter")
          val denseShape = {
            if (i.denseShape ne null)
              enter(i.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShapeEnter")
            else
              null
          }
          Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: Op.SparseOutput =>
          val values = enter(i.values, frameName, isConstant, parallelIterations, useInputShape, "ValuesEnter")
          val indices = enter(i.indices, frameName, isConstant, parallelIterations, useInputShape, "IndicesEnter")
          val denseShape = enter(
            i.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShapeEnter")
          Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
      // @formatter:on
    }.asInstanceOf[T]
  }

  /** Creates an op that exits from the current frame to its parent frame.
    *
    * The op makes `input` available to the parent frame.
    *
    * @param  input Tensor to be made available to the parent frame.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def exit[T <: Op.OutputLike](input: T, name: String = "Exit"): T = {
    Op.createWithNameScope(nameScope = name, Array(input.op)) {
      // @formatter:off
      input match {
        case i: Op.Output =>
          Op.Builder(opType = "Exit", name = name)
              .addInput(i)
              .build().outputs(0)
        case i: Op.OutputIndexedSlices =>
          val values = exit(i.values, name = "ValuesExit")
          val indices = exit(i.indices, name = "IndicesExit")
          val denseShape = {
            if (i.denseShape ne null)
              exit(i.denseShape, name = "DenseShapeExit")
            else
              null
          }
          Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: Op.SparseOutput =>
          val values = exit(i.values, name = "ValuesExit")
          val indices = exit(i.indices, name = "IndicesExit")
          val denseShape = exit(i.denseShape, name = "DenseShapeExit")
          Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
      // @formatter:on
    }.asInstanceOf[T]
  }

  /** Creates an op that forwards `input` to the output port determined by `predicate`.
    *
    * If `predicate` is `true`, then `input` is forwarded to `outputTrue`. Otherwise, it goes to `outputFalse`.
    *
    * @param  input     Tensor to be forwarded to the appropriate output.
    * @param  predicate Scalar boolean tensor that specifies which output port will receive `input`.
    * @param  name      Name for the created op.
    * @return Tuple containing `outputFalse` and `outputTrue`, in that order.
    */
  private[this] def switch[T <: Op.OutputLike](input: T, predicate: Op.Output, name: String = "Switch"): (T, T) = {
    Op.createWithNameScope(nameScope = name, Array(input.op, predicate.op)) {
      // @formatter:off
      input match {
        case i: Op.Output =>
          val outputs = Op.Builder(opType = "Switch", name = name)
              .addInput(i)
              .addInput(predicate)
              .build().outputs
          (outputs(0), outputs(1))
        case i: Op.OutputIndexedSlices =>
          val (valuesFalse, valuesTrue) = switch(i.values, predicate, name = "ValuesSwitch")
          val (indicesFalse, indicesTrue) = switch(i.indices, predicate, name = "IndicesSwitch")
          val (denseShapeFalse, denseShapeTrue) = {
            if (i.denseShape ne null)
              switch(i.denseShape, predicate, name = "DenseShapeSwitch")
            else
              (null, null)
          }
          (Op.OutputIndexedSlices(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
              Op.OutputIndexedSlices(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
        case i: Op.SparseOutput =>
          val (valuesFalse, valuesTrue) = switch(i.values, predicate, name = "ValuesSwitch")
          val (indicesFalse, indicesTrue) = switch(i.indices, predicate, name = "IndicesSwitch")
          val (denseShapeFalse, denseShapeTrue) = switch(i.denseShape, predicate, name = "DenseShapeSwitch")
          (Op.SparseOutput(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
              Op.SparseOutput(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
      }
      // @formatter:on
    }.asInstanceOf[(T, T)]
  }

  /** Creates an op that forwards the value of an available tensor from `inputs` to `output`.
    *
    * The op tests each of the tensors in `inputs` in turn to determine if any of them is available. If it finds an
    * available tensor, it returns it and its index, `outputIndex`, in `inputs`.
    *
    * No more than one tensor in `inputs` should be available. If no tensor in `inputs` is available, the returned
    * tensor and index are not set.
    *
    * This op is usually combined with `switch` to implement branching.
    *
    * IMPORTANT NOTE: The input tensors can either all be of type [[Op.SparseOutput]] or of mixed types that extend
    * [[Op.OutputIndexedSlicesConvertible]]. If they are all of type [[Op.Output]], then that is also the return op
    * type. Otherwise, they will all be converted to [[Op.OutputIndexedSlices]] first.
    *
    * @param  inputs Input tensors.
    * @param  name   Name for the created op.
    * @return Tuple containing `output` and `outputIndex`, in that order.
    */
  private[this] def merge[T <: Op.OutputLike : TypeTag](
      inputs: Array[T], name: String = "Merge"): (Op.OutputLike, Op.Output) = {
    Op.createWithNameScope(nameScope = name, inputs.map(_.op)) {
      // @formatter:off
      inputs match {
        case i if typeOf[T] =:= typeOf[Op.Output] =>
          val outputs = Op.Builder(opType = "Merge", name = name)
              .addInputs(i.asInstanceOf[Array[Op.Output]])
              .build().outputs
          (outputs(0), outputs(1))
        case i if typeOf[T] =:= typeOf[Op.SparseOutput] =>
          val (values, _) = merge(i.map(_.asInstanceOf[Op.SparseOutput].values), "ValuesMerge")
          val (indices, chosenIndex) = merge(i.map(_.asInstanceOf[Op.SparseOutput].indices), "IndicesMerge")
          val (denseShape, _) = merge(i.map(_.asInstanceOf[Op.SparseOutput].denseShape), "DenseShapeMerge")
          (Op.SparseOutput(
            indices = indices.asInstanceOf[Op.Output],
            values = values.asInstanceOf[Op.Output],
            denseShape = denseShape.asInstanceOf[Op.Output]), chosenIndex)
        case i if typeOf[T] =:= typeOf[Op.OutputIndexedSlicesConvertible] =>
          val ii = i.map(_.asInstanceOf[Op.OutputIndexedSlicesConvertible].toOpOutputIndexedSlices(optimize = true))
          val (values, _) = merge(ii.map(_.values), "ValuesMerge")
          val (indices, chosenIndex) = merge(ii.map(_.indices), "IndicesMerge")
          val denseShape = if (ii.map(_.denseShape).exists(_ ne null)) {
            if (ii.map(_.denseShape).exists(_ eq null))
              throw new IllegalArgumentException(
                "Either all merged 'Op.OutputIndexedSlices' must have a known dense shape, or none of them.")
            merge(ii.map(_.denseShape), "DenseShapeMerge")
          } else {
            null
          }
          (Op.OutputIndexedSlices(
            indices = indices.asInstanceOf[Op.Output],
            values = values.asInstanceOf[Op.Output],
            denseShape = denseShape.asInstanceOf[Op.Output]), chosenIndex)
        case _ => throw new IllegalArgumentException("Invalid inputs passed to 'merge'.")
      }
      // @formatter:on
    }.asInstanceOf[(T, Op.Output)]
  }
}
