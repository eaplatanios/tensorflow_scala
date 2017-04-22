package org.platanios.tensorflow.api.ops

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
  private[this] def nextIteration(input: Op.Output, name: String = "NextIteration"): Op.Output = {
    Op.Builder(opType = "NextIteration", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an op that makes its input available to the next iteration.
    *
    * @param  input Tensor to make available to the next iteration.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def nextIteration(
      input: Op.OutputIndexedSlices, name: String = "NextIteration"): Op.OutputIndexedSlices = {
    Op.createWith(nameScope = name) {
      val values = nextIteration(input.values, name = "ValuesNextIteration")
      val indices = nextIteration(input.indices, name = "IndicesNextIteration")
      val denseShape = {
        if (input.denseShape ne null)
          nextIteration(input.denseShape, name = "DenseShapeNextIteration")
        else
          null
      }
      Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
    }
  }

  /** Creates an op that makes its input available to the next iteration.
    *
    * @param  input Tensor to make available to the next iteration.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def nextIteration(input: Op.SparseOutput, name: String = "NextIteration"): Op.SparseOutput = {
    Op.createWith(nameScope = name) {
      val values = nextIteration(input.values, name = "ValuesNextIteration")
      val indices = nextIteration(input.indices, name = "IndicesNextIteration")
      val denseShape = nextIteration(input.denseShape, name = "DenseShapeNextIteration")
      Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
    }
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
  private[this] def enter(
      input: Op.Output, frameName: String, isConstant: Boolean = false, parallelIterations: Int = 10,
      useInputShape: Boolean = true, name: String = "Enter"): Op.Output = {
    val result = Op.Builder(opType = "NextIteration", name = name)
        .addInput(input)
        .build().outputs(0)
    if (useInputShape)
      result.setShape(input.shape)
    result
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
  private[this] def enter(
      input: Op.OutputIndexedSlices, frameName: String, isConstant: Boolean = false, parallelIterations: Int = 10,
      useInputShape: Boolean = true, name: String = "Enter"): Op.OutputIndexedSlices = {
    Op.createWith(nameScope = name) {
      val values = enter(
        input.values, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
        useInputShape = useInputShape, name = "ValuesEnter")
      val indices = enter(
        input.indices, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
        useInputShape = useInputShape, name = "IndicesEnter")
      val denseShape = {
        if (input.denseShape ne null)
          enter(
            input.denseShape, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
            useInputShape = useInputShape, name = "DenseShapeEnter")
        else
          null
      }
      Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
    }
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
  private[this] def enter(
      input: Op.SparseOutput, frameName: String, isConstant: Boolean = false, parallelIterations: Int = 10,
      useInputShape: Boolean = true, name: String = "Enter"): Op.SparseOutput = {
    Op.createWith(nameScope = name) {
      val values = enter(
        input.values, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
        useInputShape = useInputShape, name = "ValuesEnter")
      val indices = enter(
        input.indices, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
        useInputShape = useInputShape, name = "IndicesEnter")
      val denseShape = enter(
        input.denseShape, frameName = frameName, isConstant = isConstant, parallelIterations = parallelIterations,
        useInputShape = useInputShape, name = "DenseShapeEnter")
      Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
    }
  }

  /** Creates an op that exits from the current frame to its parent frame.
    *
    * The op makes `input` available to the parent frame.
    *
    * @param  input Tensor to be made available to the parent frame.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def exit(input: Op.Output, name: String = "Exit"): Op.Output = {
    Op.Builder(opType = "Exit", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an op that exits from the current frame to its parent frame.
    *
    * The op makes `input` available to the parent frame.
    *
    * @param  input Tensor to be made available to the parent frame.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def exit(input: Op.OutputIndexedSlices, name: String = "Exit"): Op.OutputIndexedSlices = {
    Op.createWith(nameScope = name) {
      val values = exit(input.values, name = "ValuesExit")
      val indices = exit(input.indices, name = "IndicesExit")
      val denseShape = {
        if (input.denseShape ne null)
          exit(input.denseShape, name = "DenseShapeExit")
        else
          null
      }
      Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
    }
  }

  /** Creates an op that exits from the current frame to its parent frame.
    *
    * The op makes `input` available to the parent frame.
    *
    * @param  input Tensor to be made available to the parent frame.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[this] def exit(input: Op.SparseOutput, name: String = "Exit"): Op.SparseOutput = {
    Op.createWith(nameScope = name) {
      val values = exit(input.values, name = "ValuesExit")
      val indices = exit(input.indices, name = "IndicesExit")
      val denseShape = exit(input.denseShape, name = "DenseShapeExit")
      Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
    }
  }
}
