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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.{TypeTag, typeOf}

/**
  * @author Emmanouil Antonios Platanios
  */
trait ControlFlow {
  /** Creates an op that produces the content of `input` only after all ops in `dependencies` have finished executing.
    *
    * In some cases, a user may want the output of an op to be consumed externally only after some other dependencies
    * have run first. This function ensures returns `input`, but only after all ops in `dependencies` have run. Note
    * that this means that there is no guarantee that `input` will be evaluated after any `dependencies` have run.
    *
    * @param  dependencies Set of ops to be executed before `input`.
    * @param  input        Op output to be computed after all ops in `dependencies` have finished executing.
    * @param  name         Name for the created op (used mainly as a name scope).
    * @return Created op output.
    */
  private[api] def withControlDependencies[T <: OutputLike](
      dependencies: Set[Op], input: T, name: String = "WithControlDependencies"): T = {
    Op.createWithNameScope(name, dependencies + input.op) {
      Op.colocateWith(Set[Op](input.op)) {
        Op.createWith(controlDependencies = dependencies) {
          Basic.identity(input)
        }
      }
    }
  }

  /** Creates an op that groups multiple ops together.
    *
    * When this op finishes, all ops in `inputs` have finished. This op has no output.
    *
    * @param  inputs Ops to group.
    * @param  name   Name for the created op (used mainly as a name scope).
    * @return Created op output, which in this case is the result of a `noOp`.
    */
  def group(inputs: Set[Op], name: String = "GroupDependencies"): Op = {
    Op.createWithNameScope(name, inputs) {
      val inputsByDevice = inputs.groupBy(_.device)
      if (inputsByDevice.size == 1) {
        // 1-level tree. The root node is the returned no-op node.
        val (device, ops) = inputsByDevice.head
        if (device != null && device != "")
          Op.createWith(device = device, controlDependencies = ops)(noOp(name))
        else
          Op.createWith(controlDependencies = ops)(noOp(name))
      } else {
        // 2-level tree. The root node is the returned no-op node. `dependencies` contains 1 NoOp node for each device.
        val dependencies = inputsByDevice.toSeq.sortBy(_._1).map {
          case (device, ops) =>
            if (device != null && device != "")
              Op.createWith(device = device, controlDependencies = ops)(noOp(name))
            else
              Op.createWith(controlDependencies = ops)(noOp(name))
        }
        Op.createWith(controlDependencies = dependencies.toSet)(noOp(name))
      }
    }
  }

  /** Creates an op that groups op outputs together.
    *
    * The op creates a tuple of op outputs with the same values as `inputs`, except that the value of each output is
    * only returned after the values of all outputs in `inputs` have been computed.
    *
    * This op can be used as a "join" mechanism for parallel computations: all the argument tensors can be computed in
    * parallel, but the values of any tensor returned by `tuple` are only available after all the parallel computations
    * are done.
    *
    * @param  inputs        Op outputs being grouped.
    * @param  controlInputs Set of additional ops that have to finish before this op finishes, but whose outputs are not
    *                       returned.
    * @param  name          Name for the created ops (used mainly as a name scope).
    * @return Created op outputs, which in this case are the values of `inputs`.
    */
  def tuple[T <: OutputLike](
      inputs: Array[T], controlInputs: Set[Op] = Set.empty, name: String = "Tuple")
      (implicit tag: ClassTag[T]): Array[T] = {
    val gatingOps = inputs.map(_.op).toSet
    Op.createWithNameScope(name, gatingOps) {
      val gate = group(gatingOps ++ controlInputs)
      inputs.map(withControlDependencies(Set[Op](gate), _))
    }
  }

  /** Creates an op that does nothing. The created op is only useful as a placeholder for control edges.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  private[api] def noOp(name: String = "NoOp"): Op = {
    Op.Builder(opType = "NoOp", name = name).build()
  }

  /** Creates an op that raises an exception to abort the process when called.
    *
    * @param  errorMessage     Error message associated with the exception.
    * @param  exitWithoutError If `true`, the process will exit normally. Otherwise, it will exit with a `SIGABORT`
    *                          signal.
    * @param  name             Name for the created op.
    * @return Created op output.
    */
  private[api] def abort(
      errorMessage: String = "", exitWithoutError: Boolean = false, name: String = "Abort"): Output = {
    Op.Builder(opType = "Abort", name = name)
        .setAttribute("error_message", errorMessage)
        .setAttribute("exit_without_error", exitWithoutError)
        .build().outputs(0)
  }
}

object ControlFlow extends ControlFlow {
  /** Creates an op that does nothing and serves as a control trigger for scheduling. The created op is only useful as
    * a placeholder for control edges.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  private[ControlFlow] def controlTrigger(name: String = "ControlTrigger"): Output = {
    Op.Builder(opType = "ControlTrigger", name = name).build().outputs(0)
  }

  /** Creates an op that forwards its input to the output.
    *
    * The op represents the loop termination condition used by the "pivot" switches of a loop.
    *
    * @param  input Boolean scalar tensor, representing the branch predicate of the switch op.
    * @param  name  Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  private[ControlFlow] def loopCond(input: Output, name: String = "LoopCond"): Output = {
    Op.Builder(opType = "LoopCond", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Returns `true` if and only if the provided op is a switch op. */
  private[ControlFlow] def isSwitch(op: Op): Boolean = op.opType == "Switch" && op.opType == "RefSwitch"

  /** Returns `true` if and only if the provided op is a loop invariant. */
  private[ControlFlow] def isLoopConstantEnter(op: Op): Boolean = {
    op.opType == "Enter" && op.booleanAttribute("is_constant")
  }

  /** Returns `true` if and only if the provided op is a loop exit op. */
  private[ControlFlow] def isLoopExit(op: Op): Boolean = op.opType == "Exit" && op.opType == "RefExit"

  /** Returns `true` if and only if the provided op is a switch op for a while loop. */
  private[ControlFlow] def isLoopSwitch(op: Op): Boolean = ???

  /** Returns the enter op if we can infer `value` to be a loop invariant. Otherwise, returns [[None]]. */
  private[ControlFlow] def getLoopConstantEnter(value: Output): Option[Op] = {
    val identityOpTypes = Set("Identity", "Switch")
    var op = value.op
    while (identityOpTypes.contains(op.opType))
      op = op.inputs(0).op
    Some(op).filter(isLoopConstantEnter)
  }

//  /** Returns the control flow context for the outputs of an op. */
//  private[ControlFlow] def getOutputContext(op: Op): Option[ControlFlowContext] = {
//    val context = op.controlFlowContext
//    if (isLoopExit(op))
//      context.flatMap(_.outerContext)
//    else
//      context
//  }

  /** Creates an op that forwards `input` to the output port determined by `predicate`, while making sure the new op is
    * colocated with `input`.
    *
    * If `predicate` is `true`, then `input` is forwarded to `outputTrue`. Otherwise, it goes to `outputFalse`.
    *
    * @param  input     Tensor to be forwarded to the appropriate output.
    * @param  predicate Scalar boolean tensor that specifies which output port will receive `input`.
    * @param  name      Name for the created op.
    * @return Tuple containing `outputFalse` and `outputTrue`, in that order.
    */
  private[ControlFlow] def colocatedSwitch[T <: OutputLike](
      input: T, predicate: Output, name: String = "Switch"): (T, T) = {
    // The device colocation below addresses the following scenario:
    //
    // Assume you execute Optimizer.applyGradients() in a branch of a cond() and:
    //   1. The update op is created inside a `Op.colocateWith(Set(var.op)) { }` block.
    //   2. Some tensor `data` is captured and a switch is created in a `Op.colocateWith(Set(data.op)) { }` block.
    //
    // Op.colocateWith(Set(var.op)) {
    //   Op.colocateWith(Set(data.op)) {
    //     op = ...
    //   }
    // }
    //
    // `var` and `data` may be pinned to different devices and so we want the ops created within the
    // `Op.colocateWith(Set(data.op)) { }` block to ignore the existing stack.
    Op.colocateWith(Set(input.op), ignoreExisting = true)(switch(input = input, predicate = predicate, name = name))
  }

  //  /** Creates a next iteration op for `v` and adds a back edge from `v` to `m`. */
  //  @throws[IllegalArgumentException]
  //  private[ControlFlow] def addNextIterationAndBackEdge[T <: OutputLike](m: T, v: T): Unit = {
  //    (m, v) match {
  //      case (mm: Output, vv: Output) => updateInput(mm.op, 1, nextIteration(vv))
  //      case (mm: OutputIndexedSlices, vv: OutputIndexedSlices) =>
  //        val nextVV = nextIteration(vv)
  //        updateInput(mm.values.op, 1, nextVV.values)
  //        updateInput(mm.indices.op, 1, nextVV.indices)
  //        if (mm.denseShape != null) {
  //          if (nextVV.denseShape == null)
  //            throw new IllegalArgumentException(s"Output indexed slices '$nextVV' must have dense shape information.")
  //          else
  //            updateInput(mm.denseShape.op, 1, nextVV.denseShape)
  //        }
  //      case (mm: SparseOutput, vv: SparseOutput) =>
  //        val nextVV = nextIteration(vv)
  //        updateInput(mm.values.op, 1, nextVV.values)
  //        updateInput(mm.indices.op, 1, nextVV.indices)
  //        updateInput(mm.denseShape.op, 1, nextVV.denseShape)
  //      case (_, _) =>
  //        throw new IllegalArgumentException(
  //          "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the tensor types must match.")
  //    }
  //  }

  //region Shape Invariants

  /** Returns `true` if `shape2` is a less strict shape than `shape1`, while being compatible with `shape1`. */
  private[this] def shapeLessThenOrEqual(shape1: Shape, shape2: Shape): Boolean = {
    shape2.rank == -1 ||
        shape1.rank == shape2.rank ||
        shape1.asArray.zip(shape2.asArray).forall(pair => pair._2 == -1 || pair._1 == pair._2)
  }

  /** Sets the shapes of the tensors in `enterTensors` to `shapes` and makes sure that the shape invariants apply.
    *
    * @param  inputTensors Tensors that are inputs to `enterTensors`.
    * @param  enterTensors Tensors whose shapes will be set.
    * @param  shapes       Shapes to use for `enterTensors`.
    * @throws ShapeMismatchException   If any tensor in `inputTensors` has a less specific shape than its corresponding
    *                                  shape in `shapes`.
    * @throws IllegalArgumentException If the types of the input tensors do not match the types of the enter tensors or
    *                                  if the type of either is not supported.
    */
  @throws[ShapeMismatchException]
  @throws[IllegalArgumentException]
  private[this] def setShapeInvariants(
      inputTensors: Array[OutputLike], enterTensors: Array[OutputLike], shapes: Array[Shape]): Unit = {
    // Check that the shapes of the inputs are less than the shape invariants, and set the shapes of the enter tensors
    // to the shape invariants.
    for ((input, enter, shape) <- (inputTensors, enterTensors, shapes).zipped) {
      (input, enter) match {
        case (i: Output, e: Output) =>
          if (!shapeLessThenOrEqual(i.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.name}' is not compatible with the initial shape of the " +
                  s"loop variable. It enters the loop with shape '${i.shape}', but the specified shape invariant " +
                  s"is '$shape'.")
          e.setShape(shape)
        case (i: OutputIndexedSlices, e: OutputIndexedSlices) =>
          if (!shapeLessThenOrEqual(i.values.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.values.name}' is not compatible the initial shape of the " +
                  s"values tensor of these indexed slices. It enters the loop with shape '${i.values.shape}', but " +
                  s"the specified shape invariant is '$shape'.")
          e.values.setShape(shape)
          e.indices.setShape(Shape(shape(0)))
          if (e.denseShape != null)
            e.denseShape.setShape(Shape(shape.rank))
        case (i: SparseOutput, e: SparseOutput) =>
          if (!shapeLessThenOrEqual(i.denseShape.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.denseShape.name}' is not compatible the initial shape of the " +
                  s"dense shape tensor of this sparse tensor. It enters the loop with shape '${i.denseShape.shape}', " +
                  s" but the specified shape invariant is '$shape'.")
          e.values.setShape(Shape(-1))
          e.indices.setShape(Shape(-1, shape.rank))
          e.denseShape.setShape(shape)
        case (_, _) =>
          throw new IllegalArgumentException(
            "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the input tensor " +
                "and the enter tensor types must match.")
      }
    }
  }

  /** Checks if the shapes of a loop variable satisfy the shape invariants.
    *
    * @param  mergeTensor Tensor representing the initial value of the loop variable.
    * @param  nextTensor  Tensor representing the value of the loop variable after one loop iteration.
    * @throws ShapeMismatchException   If `mergeTensor` has a less specific shape than its corresponding shape in
    *                                  `nextTensor`.
    * @throws IllegalArgumentException If the type of the merge tensor does not match the type of the next tensor or if
    *                                  the type of either is not supported.
    */
  @throws[ShapeMismatchException]
  @throws[IllegalArgumentException]
  private[this] def enforceShapeInvariant(mergeTensor: OutputLike, nextTensor: OutputLike): Unit = {
    (mergeTensor, nextTensor) match {
      case (merge: Output, next: Output) =>
        if (!shapeLessThenOrEqual(next.shape, merge.shape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'${merge.shape}', but has shape '${next.shape}' after one iteration. Please provide shape " +
                s"invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' method of " +
                s"the loop variables.")
      case (merge: OutputIndexedSlices, next: OutputIndexedSlices) =>
        val mergeValuesShape = merge.values.shape
        val mergeIndicesShape = merge.indices.shape
        val mergeDenseShapeShape = if (merge.denseShape != null) merge.denseShape.shape else Shape.unknown()
        val nextValuesShape = next.values.shape
        val nextIndicesShape = next.indices.shape
        val nextDenseShapeShape = if (next.denseShape != null) next.denseShape.shape else Shape.unknown()
        if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
            !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
            !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                s"'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                s"shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                s"method of the loop variables.")
      case (merge: SparseOutput, next: SparseOutput) =>
        val mergeValuesShape = merge.values.shape
        val mergeIndicesShape = merge.indices.shape
        val mergeDenseShapeShape = merge.denseShape.shape
        val nextValuesShape = next.values.shape
        val nextIndicesShape = next.indices.shape
        val nextDenseShapeShape = next.denseShape.shape
        if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
            !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
            !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                s"'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                s"shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                s"method of the loop variables.")
      case (_, _) =>
        throw new IllegalArgumentException(
          "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the merge tensor " +
              "and the next tensor types must match>")
    }
  }

  //endregion Shape Invariants

  //region Low Level Ops

  /** Creates an op that makes its input available to the next iteration.
    *
    * @param  input Tensor to make available to the next iteration.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[ControlFlow] def nextIteration[T <: OutputLike](input: T, name: String = "NextIteration"): T = {
    Op.createWithNameScope(nameScope = name, Set(input.op)) {
      input match {
        case i: Output =>
          Op.Builder(opType = "NextIteration", name = name)
              .addInput(i)
              .build().outputs(0)
        case i: OutputIndexedSlices =>
          val values = nextIteration(i.values, name = "ValuesNextIteration")
          val indices = nextIteration(i.indices, name = "IndicesNextIteration")
          val denseShape = {
            if (i.denseShape != null)
              nextIteration(i.denseShape, name = "DenseShapeNextIteration")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: SparseOutput =>
          val values = nextIteration(i.values, name = "ValuesNextIteration")
          val indices = nextIteration(i.indices, name = "IndicesNextIteration")
          val denseShape = nextIteration(i.denseShape, name = "DenseShapeNextIteration")
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
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
  private[ControlFlow] def enter[T <: OutputLike](
      input: T, frameName: String, isConstant: Boolean = false, parallelIterations: Int = 10,
      useInputShape: Boolean = true, name: String = "Enter"): T = {
    Op.createWithNameScope(nameScope = name, Set(input.op)) {
      input match {
        case i: Output =>
          val result = Op.Builder(opType = "NextIteration", name = name)
              .addInput(i)
              .build().outputs(0)
          if (useInputShape)
            result.setShape(i.shape)
          result
        case i: OutputIndexedSlices =>
          val values = enter(i.values, frameName, isConstant, parallelIterations, useInputShape, "ValuesEnter")
          val indices = enter(i.indices, frameName, isConstant, parallelIterations, useInputShape, "IndicesEnter")
          val denseShape = {
            if (i.denseShape != null)
              enter(i.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShapeEnter")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: SparseOutput =>
          val values = enter(i.values, frameName, isConstant, parallelIterations, useInputShape, "ValuesEnter")
          val indices = enter(i.indices, frameName, isConstant, parallelIterations, useInputShape, "IndicesEnter")
          val denseShape = enter(
            i.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShapeEnter")
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
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
  private[ControlFlow] def exit[T <: OutputLike](input: T, name: String = "Exit"): T = {
    Op.createWithNameScope(nameScope = name, Set(input.op)) {
      input match {
        case i: Output =>
          Op.Builder(opType = "Exit", name = name)
              .addInput(i)
              .build().outputs(0)
        case i: OutputIndexedSlices =>
          val values = exit(i.values, name = "ValuesExit")
          val indices = exit(i.indices, name = "IndicesExit")
          val denseShape = {
            if (i.denseShape != null)
              exit(i.denseShape, name = "DenseShapeExit")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: SparseOutput =>
          val values = exit(i.values, name = "ValuesExit")
          val indices = exit(i.indices, name = "IndicesExit")
          val denseShape = exit(i.denseShape, name = "DenseShapeExit")
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
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
  private[ControlFlow] def switch[T <: OutputLike](input: T, predicate: Output, name: String = "Switch"): (T, T) = {
    Op.createWithNameScope(nameScope = name, Set(input.op, predicate.op)) {
      input match {
        case i: Output =>
          val outputs = Op.Builder(opType = "Switch", name = name)
              .addInput(i)
              .addInput(predicate)
              .build().outputs
          (outputs(0), outputs(1))
        case i: OutputIndexedSlices =>
          val (valuesFalse, valuesTrue) = switch(i.values, predicate, name = "ValuesSwitch")
          val (indicesFalse, indicesTrue) = switch(i.indices, predicate, name = "IndicesSwitch")
          val (denseShapeFalse, denseShapeTrue) = {
            if (i.denseShape != null)
              switch(i.denseShape, predicate, name = "DenseShapeSwitch")
            else
              (null, null)
          }
          (OutputIndexedSlices(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
              OutputIndexedSlices(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
        case i: SparseOutput =>
          val (valuesFalse, valuesTrue) = switch(i.values, predicate, name = "ValuesSwitch")
          val (indicesFalse, indicesTrue) = switch(i.indices, predicate, name = "IndicesSwitch")
          val (denseShapeFalse, denseShapeTrue) = switch(i.denseShape, predicate, name = "DenseShapeSwitch")
          (SparseOutput(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
              SparseOutput(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
      }
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
    * IMPORTANT NOTE: The input tensors can either all be of type [[SparseOutput]] or of mixed types that extend
    * [[OutputLike]]. If they are all of type [[Output]] or [[SparseOutput]], then that is also the return op type.
    * Otherwise, they will all be converted to [[OutputIndexedSlices]] first.
    *
    * @param  inputs Input tensors.
    * @param  name   Name for the created op.
    * @return Tuple containing `output` and `outputIndex`, in that order.
    */
  @throws[IllegalArgumentException]
  private[ControlFlow] def merge[T <: OutputLike : TypeTag](
      inputs: Array[T], name: String = "Merge"): (OutputLike, Output) = {
    Op.createWithNameScope(nameScope = name, inputs.map(_.op).toSet) {
      inputs match {
        case i if typeOf[T] =:= typeOf[Output] =>
          val outputs = Op.Builder(opType = "Merge", name = name)
              .addInputList(i.asInstanceOf[Array[Output]])
              .build().outputs
          (outputs(0), outputs(1))
        case i if typeOf[T] =:= typeOf[SparseOutput] =>
          val (values, _) = merge(i.map(_.asInstanceOf[SparseOutput].values), "ValuesMerge")
          val (indices, chosenIndex) = merge(i.map(_.asInstanceOf[SparseOutput].indices), "IndicesMerge")
          val (denseShape, _) = merge(i.map(_.asInstanceOf[SparseOutput].denseShape), "DenseShapeMerge")
          (SparseOutput(
            indices = indices.asInstanceOf[Output],
            values = values.asInstanceOf[Output],
            denseShape = denseShape.asInstanceOf[Output]), chosenIndex)
        case i =>
          val ii = i.map(_.toOutputIndexedSlices(optimize = true))
          val (values, _) = merge(ii.map(_.values), "ValuesMerge")
          val (indices, chosenIndex) = merge(ii.map(_.indices), "IndicesMerge")
          val denseShape = if (ii.map(_.denseShape).exists(_ != null)) {
            if (ii.map(_.denseShape).contains(null))
              throw new IllegalArgumentException(
                "Either all merged 'OutputIndexedSlices' must have a known dense shape, or none of them.")
            merge(ii.map(_.denseShape), "DenseShapeMerge")
          } else {
            null
          }
          (OutputIndexedSlices(
            indices = indices.asInstanceOf[Output],
            values = values.asInstanceOf[Output],
            denseShape = denseShape.asInstanceOf[Output]), chosenIndex)
      }
    }.asInstanceOf[(T, Output)]
  }

  //endregion Low Level Ops

  //  //region Native Library Functions
  //
  //  /** Replaces the `index`th input of `op` with `newInput`. */
  //  private[ControlFlow] def updateInput(op: Op, index: Int, newInput: Output): Unit = {
  //    using(op.graph.reference)(r => {
  //      NativeLibrary.updateInput(r.nativeHandle, op.nativeHandle, index, newInput.op.nativeHandle, newInput.index)
  //    })
  //  }
  //
  //  /** Adds `inputOp` as a control input of `op`. */
  //  private[ControlFlow] def addControlInput(op: Op, inputOp: Op): Unit = {
  //    using(op.graph.reference)(r => {
  //      NativeLibrary.addControlInput(r.nativeHandle, op.nativeHandle, inputOp.nativeHandle)
  //    })
  //  }
  //
  //  /** Clears the control inputs of `op` (i.e., removes all of them). */
  //  private[ControlFlow] def clearControlInputs(op: Op): Unit = {
  //    using(op.graph.reference)(r => {
  //      NativeLibrary.clearControlInputs(r.nativeHandle, op.nativeHandle)
  //    })
  //  }
  //
  //  //endregion Native Library Functions
}
