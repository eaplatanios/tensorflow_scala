/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.api.utilities.Collections

import com.google.protobuf.{ByteString, GeneratedMessageV3}
import org.tensorflow.framework.{AttrValue, CollectionDef, WhileContextDef}
import org.tensorflow.framework.CollectionDef.BytesList
import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.language.postfixOps
import scala.reflect.ClassTag

/** Control flow context for the while-loop construct.
  *
  * @param  maximumIterations     Optional scalar specifying the maximum number of iterations to loop for. If
  *                               `null` (the default), no iteration limit is enforced.
  * @param  parallelIterations    Number of iterations allowed to run in parallel.
  * @param  enableBackPropagation If `true`, back-propagation support is enabled for this while-loop context.
  * @param  swapMemory            If `true`, GPU-CPU memory swapping support is enabled for this while-loop context.
  * @param  gradientLoopState     Gradient loop state.
  * @param  pivot                 Tensor used for the loop termination condition. Used in code generation for the
  *                               gradient computation.
  * @param  pivotForPredicate     We use this node to control constants created by the predicate function.
  * @param  pivotForBody          We use this node to control constants created by the body function.
  * @param  loopEnters            Enter tensors for loop variables.
  * @param  loopExits             Exit tensors for loop variables.
  * @param  requestedName         Requested name prefix for this while-loop context. Note that this will be made unique
  *                               and thus the actual name of the created while loop context may differ from the
  *                               requested one.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class WhileLoopContext private[control_flow] (
    maximumIterations: Option[Output[Int]] = None,
    parallelIterations: Int = 10,
    enableBackPropagation: Boolean = true,
    swapMemory: Boolean = false,
    gradientLoopState: Option[GradientLoopState] = None,
    private[ops] var pivot: Output[Boolean] = null,
    private var pivotForPredicate: UntypedOp = null,
    private var pivotForBody: UntypedOp = null,
    private[control_flow] val loopEnters: mutable.Set[Output[Any]] = mutable.Set.empty[Output[Any]],
    private[control_flow] val loopExits: mutable.Set[Output[Any]] = mutable.Set.empty[Output[Any]],
    private val requestedName: String = "WhileLoopContext"
) extends Context() with ProtoSerializable {
  require(parallelIterations > 0, "'parallelIterations' must be a positive integer.")

  override val name: String = {
    Op.currentGraph.uniqueName(requestedName)
  }

  override def controlPivot: Option[UntypedOp] = {
    Option(pivotForBody).orElse(Option(pivotForPredicate))
  }

  override def whileLoopContext(stopContext: Option[Context] = None): Option[WhileLoopContext] = {
    Some(this)
  }

  override def add(op: UntypedOp): Unit = {
    // For a reduction op, if the op is in a gradient context and its input is from its forward context, moving the op
    // to the forward context means we would store the tensor after the reduction as opposed to the tensor before the
    // reduction, and therefore we could significantly reduce memory consumption. For now, we do this only for a few
    // ops.
    var added = false
    if (Set("Shape", "Size", "Rank").contains(op.opType)) {
      val gradientContext = Op.currentControlFlowContext
      if (gradientContext.isDefined) {
        gradientContext.flatMap(_.whileLoopContext().flatMap(_.gradientLoopState)).foreach(gradientLoopState => {
          WhileLoopContext.getWhileLoopContext(op.inputsSeq(0).op).foreach(opInputForwardContext => {
            if (opInputForwardContext == gradientLoopState.forwardContext) {
              val opInputContext = op.inputsSeq(0).op.controlFlowContext
              op.controlFlowContext = opInputContext
              opInputContext.foreach(_.addInternal(op))
              added = true
            }
          })
        })
      }
    }
    if (!added)
      addInternal(op)
  }

  //  override private[control_flow] def addInternal(op: Op): Unit = {
  //    if (op.numInputs == 0) {
  //      // Remove any external control dependencies on this op.
  //      val controlInputs = removeExternalControlEdges(op)
  //      // Add a control edge from the control pivot to this op.
  //      if (controlInputs.isEmpty)
  //        controlPivot.foreach(ControlFlow.addControlInput(op, _))
  //      op.outputs.foreach(values += _.name)
  //    } else {
  //      op.inputs.zipWithIndex.foreach({
  //        case (input, index) =>
  //          val realInput = add(input)
  //          if (realInput != input)
  //            ControlFlow.updateInput(op, index, realInput)
  //      })
  //      // Remove any external control dependencies on this op.
  //      removeExternalControlEdges(op)
  //      // Add a control dependency to prevent loop invariants from enabling ops that should not be executed.
  //      if (op.controlInputs.isEmpty &&
  //          ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient") ||
  //              op.inputs.forall(o => ControlFlow.isLoopConstantEnter(o.op))))
  //        controlPivot.foreach(ControlFlow.addControlInput(op, _))
  //      op.outputs.foreach(values += _.name)
  //    }
  //    if (outerContext.isDefined || !ControlFlow.isLoopExit(op)) {
  //      op.graph.preventFetching(op)
  //      op.outputs.foreach(op.graph.preventFeeding)
  //    }
  //    outerContext.foreach(_.addInnerOp(op))
  //  }

  override def add[T: TF](output: Output[T]): Output[T] = {
    if (values.contains(output.name)) {
      // Use the real value if it comes from an outer context. This is needed in particular for nested conditionals.
      externalValues.getOrElse(output.name, output).asInstanceOf[Output[T]]
    } else {
      var value: Output[T] = null
      values += output.name
      // If we are in a gradient context and `output` is from its forward context, we use `getRealValue()`, which adds
      // the logic to save the history of `output` in the forward pass.
      Op.currentControlFlowContext.foreach(gradientContext => {
        gradientContext.whileLoopContext().flatMap(_.gradientLoopState).foreach(gradientLoopState => {
          WhileLoopContext.getWhileLoopContext(output.op).flatMap(forwardContext => {
            if (ControlFlow.isLoopExit(output.op))
              forwardContext.outerContext.flatMap(_.whileLoopContext())
            else
              Some(forwardContext)
          }).foreach(forwardContext => {
            if (forwardContext == gradientLoopState.forwardContext) {
              val realValue = gradientLoopState.getRealValue(output)
              externalValues += output.name -> realValue
              value = realValue.asInstanceOf[Output[T]]
            }
          })
        })
      })
      if (value != null) {
        value
      } else {
        val result = outerContext.map(_.add(output)).getOrElse(output)
        // Create an enter op to make `result` known to this loop context.
        val enter = Op.createWith(controlDependencies = Set.empty) {
          val enter = ControlFlow.enter(
            input = result,
            frameName = name,
            isConstant = true,
            parallelIterations = parallelIterations
          )(TF.fromDataType(result.dataType))
          enter.graph.preventFeeding(enter)
          enter
        }
        // Fix the control inputs and control flow context of these enter ops.
        fixControlInputsAndContext(Seq(enter))
        // Add `enter` in this context.
        values += enter.name
        externalValues += output.name -> enter
        enter
      }
    }
  }

  def backPropagate: Boolean = {
    enableBackPropagation
  }

  private[this] def isInOuterContext(op: Op[_, _]): Boolean = {
    val opContext = ControlFlow.getOutputContext(op)
    var outerContext = this.outerContext
    while (outerContext != opContext && outerContext.isDefined)
      outerContext = outerContext.flatMap(_.outerContext)
    outerContext == opContext
  }

  /** Adds the loop termination condition and the loop body to the graph. */
  private[control_flow] def buildLoop[T, TS](
      predicateFn: T => Output[Boolean],
      bodyFn: T => T,
      loopVariables: T,
      shapeInvariants: Option[TS]
  )(implicit ev: WhileLoopVariable.Aux[T, TS]): T = {
    try {
      // Enter the frame for this loop.
      enter()

      val flattenedLoopVariables = ev.outputs(loopVariables)
      val flattenedShapeInvariants = shapeInvariants.map(ev.shapes)

      // Let the context know the loop variables so the loop variables would be added to the outer contexts properly.
      initializeValues(flattenedLoopVariables)
      val realVariables = outerContext.map(c => {
        flattenedLoopVariables.map(v => {
          c.add(v)(TF.fromDataType(v.dataType))
        })
      }).getOrElse(flattenedLoopVariables)
      val enterVariables = Op.createWith(controlDependencies = Set.empty) {
        val enterVariables = realVariables.map(v => {
          ControlFlow.enter(
            input = v,
            frameName = name,
            isConstant = false,
            parallelIterations = parallelIterations,
            useInputShape = flattenedShapeInvariants.isEmpty
          )(TF.fromDataType(v.dataType))
        })
        enterVariables.foreach(v => v.graph.preventFeeding(v))
        enterVariables
      }

      // Find the closest enclosing non-None control pivot.
      var outerCtx: Option[Context] = outerContext
      var controlPivot: Option[UntypedOp] = None
      while (outerCtx.isDefined && controlPivot.isEmpty) {
        controlPivot = outerContext.get.controlPivot
        outerCtx = outerCtx.get.outerContext
      }

      controlPivot.foreach(p => {
        enterVariables.filter(v => ControlFlow.isLoopConstantEnter(v.op.inputsSeq(0).op)).foreach(v => {
          ControlFlow.addControlInput(v.op, p)
        })
      })
      flattenedShapeInvariants.foreach(WhileLoopContext.setShapeInvariants(realVariables, enterVariables, _))

      // Fix the control inputs and control flow context of these enter ops.
      fixControlInputsAndContext(enterVariables)
      initializeValues(enterVariables)
      loopEnters.clear()
      loopEnters ++= enterVariables

      val mergeVariables = enterVariables.map(v => {
        ControlFlow.merge(Seq(v, v))(TF.fromDataType(v.dataType))._1
      })
      pivotForPredicate = mergeVariables(0).op

      // Build the graph for the predicate.
      val packedPredicateVariables = ev.fromOutputs(loopVariables, mergeVariables)
      val predicateResult = predicateFn(packedPredicateVariables)
      pivot = ControlFlow.loopCond(predicateResult, name = "LoopCond")
      val switchVariables = mergeVariables.map(v => {
        ControlFlow.colocatedSwitch(v, pivot)(TF.fromDataType(v.dataType))
      })

      // Build the graph for the body.
      val bodyVariables = switchVariables.map(v => {
        Basic.identity(v._2)(TF.fromDataType(v._2.dataType))
      })
      pivotForBody = bodyVariables(0).op
      val packedBodyVariables = ev.fromOutputs(loopVariables, bodyVariables)
      val bodyResult = bodyFn(packedBodyVariables)

      // Convert the tensor arrays returned by the body function into their flow variables.
      val flattenedBodyResult = ev.outputs(bodyResult)

      // Add the `NextIteration` op and the back edges to complete the loop.
      mergeVariables.zip(flattenedBodyResult).map(p => {
        WhileLoopContext.addNextIterationAndBackEdge(p._1, p._2)(TF.fromDataType(p._1.dataType))
      })

      // Add the exit ops.
      val exitVariables = switchVariables.map(v => {
        ControlFlow.exit(v._1)(TF.fromDataType(v._1.dataType))
      })
      loopExits.clear()
      loopExits ++= exitVariables

      // Exit the loop.
      exitResult(exitVariables)

      // Convert any tensor array flow variables outside the context back into their associated tensor arrays for
      // returning to the caller.
      ev.fromOutputs(bodyResult, exitVariables)
    } catch {
      case t: Throwable =>
        exit()
        throw t
    }
  }

  private[this] def fixControlInputsAndContext(values: Seq[OutputLike[_]]): Unit = {
    values.foreach(value => {
      val outputs = value match {
        case o: Output[_] => Set(o)
        case o: OutputIndexedSlices[_] =>
          if (o.denseShape != null)
            Set(o.indices, o.values, o.denseShape)
          else
            Set(o.indices, o.values)
        case o: SparseOutput[_] =>
          if (o.denseShape != null)
            Set(o.indices, o.values, o.denseShape)
          else
            Set(o.indices, o.values)
      }
      outputs.foreach(output => {
        val input = output.op.inputsSeq(0)
        val outerControlInputs = Op.controlDependencies(Set(input)).filter(isInOuterContext)
        output.op.controlFlowContext = Some(this)
        outerControlInputs.foreach(i => ControlFlow.addControlInput(output.op, i))
      })
    })
  }

  /** Makes the provided values known to this context. */
  private[this] def initializeValues(providedValues: Seq[OutputLike[_]]): Unit = {
    values.clear()
    providedValues.foreach {
      case v: Output[_] => values += v.name
      case v: OutputIndexedSlices[_] =>
        values += v.indices.name
        values += v.values.name
        if (v.denseShape != null)
          values += v.denseShape.name
      case v: SparseOutput[_] =>
        values += v.indices.name
        values += v.values.name
        if (v.denseShape != null)
          values += v.denseShape.name
    }
  }

  /** Adds a loop that counts the number of iterations.
    *
    * This is added to the forward loop at the time when we start to create the loop for the back-propagation gradient
    * computation. It is called in the outer context of this forward context.
    *
    * The pseudocode is: `n = 0; while (pivot) { n++; }`
    *
    * Note that a control dependency is added to `n` to ensure the correct execution order of stack push ops.
    *
    * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
    * @return Tuple containing the number of iterations taken by the forward loop and the loop index.
    */
  private[control_flow] def addForwardLoopCounter(
      outerGradientLoopState: Option[GradientLoopState]
  ): (Output[Int], Output[Int]) = {
    Op.nameScope("ForwardLoopCounter") {
      val n = Basic.zeros[Int](Shape())
      outerGradientLoopState.foreach(state => {
        // Force the stack pushes of the i-th execution of an inner loop to be ordered before the pushes of the (i+1)-th
        // execution of the same inner loop.
        ControlFlow.addControlInput(n.op, state.forwardIndex.op.inputsSeq(0).op)
      })
      enter()
      values += n.name
      val enterN = ControlFlow.enter(n, name, isConstant = false, parallelIterations)
      loopEnters += enterN
      val mergeN = ControlFlow.merge(Seq(enterN, enterN))._1
      val switchN = ControlFlow.switch(mergeN, pivot)
      val index = Math.add(switchN._2, Basic.ones[Int](Shape()))
      val nextN = ControlFlow.nextIteration(index)
      ControlFlow.updateInput(mergeN.op, 1, nextN)
      val exitN = ControlFlow.exit(switchN._1).toOutput
      loopExits += exitN
      exitResult(Seq(exitN))
      exit()
      (exitN, nextN)
    }
  }

  /** Adds the back-propagation loop that counts the number of iterations.
    *
    * This is added to the back-propagation loop. It is used to control the loop termination of the back-propagation
    * loop. It is called in the outer context of this gradient context.
    *
    * The pseudocode is: `n = count; while (n >= 1) { n--; }`
    *
    * Note that a control dependency is added to the final exit op to ensure the correct execution order of stack pop
    * ops.
    *
    * @param  count              Number of iterations for the back-propagation loop.
    * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
    * @return Loop index.
    */
  private[control_flow] def addBackwardLoopCounter(
      count: Output[Int],
      outerGradientLoopState: Option[GradientLoopState]
  ): Output[Int] = {
    Op.nameScope("BackwardLoopCounter") {
      val one = Basic.ones[Int](Shape())
      enter()
      values += count.name
      val enterC = ControlFlow.enter(count, name, isConstant = false, parallelIterations)
      loopEnters += enterC
      val mergeC = ControlFlow.merge(Seq(enterC, enterC))._1
      pivotForPredicate = mergeC.op
      pivot = ControlFlow.loopCond(Math.greaterEqual(mergeC, one))
      val switchC = ControlFlow.switch(mergeC, pivot)
      val indexC = Math.subtract(switchC._2, one)
      pivotForBody = indexC.op
      val nextC = ControlFlow.nextIteration(indexC)
      ControlFlow.updateInput(mergeC.op, 1, nextC)
      val exitC = ControlFlow.exit(switchC._1)
      loopExits += exitC
      outerGradientLoopState.foreach(state => {
        // Force the stack pops of the i-th execution of an inner loop to be ordered before the pops of the (i+1)-th
        // execution of the same inner loop.
        ControlFlow.addControlInput(state.backwardSync, exitC.op)
      })
      exitResult(Seq(exitC))
      exit()
      nextC
    }
  }

  /** Adds an accumulation loop for every loop invariant.
    *
    * This is added to the back-propagation loop. It is used to accumulate partial gradients within each loop iteration.
    * It is called when in the gradient while context.
    *
    * The pseudocode is: `acc = 0.0; while (pivot) { acc += grad; }`
    *
    * @param  op       Enter op for a loop invariant.
    * @param  gradient Partial gradient of an iteration for a loop invariant.
    * @return Gradient for a loop invariant.
    */
  private[control_flow] def addBackwardAccumulator[T: TF, OL[A] <: OutputLike[A]](
      op: Op[_, _],
      gradient: OL[T]
  ): OL[T] = {
    val result = gradient match {
      case g: Output[T] =>
        exit()
        // We create a zeros tensor with the right shape for the accumulator. If we don't know the full shape
        // statically, we will have to get the shape dynamically from the forward inference. Getting the shape right for
        // the zeros is only needed for the base case when the loop exits without running any iterations.
        val shape = g.shape
        val acc: Output[T] = {
          if (shape.isFullyDefined) {
            outerContext.foreach(_.enter())
            val acc = Basic.zerosLike(g, name = "BackwardAccumulator")(TF.fromDataType(g.dataType))
            outerContext.foreach(_.exit())
            acc
          } else {
            val value = op.inputsSeq(0)
            // TODO: !!! [CONTROL_FLOW] Is this even necessary for obtaining the shape?
            outerContext match {
              case Some(context: WhileLoopContext) if context.gradientLoopState.isDefined =>
                // We are in a nested while loop.
                val forwardContext = context.gradientLoopState.get.forwardContext
                forwardContext.outerContext.foreach(_.enter())
                val zerosShape = resourceSafeShape(value)(TF.fromDataType(value.dataType))
                forwardContext.outerContext.foreach(_.exit())
                val outerGradientLoopState = context.gradientLoopState.get.outerGradientLoopState.get
                val historyZerosShape = outerGradientLoopState.addForwardAccumulator(zerosShape)
                context.enter()
                val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyZerosShape, zerosShape)
                val acc = Basic.zeros(g.dataType, realShape)
                context.exit()
                acc.setShape(g.shape)
                acc
              case _ =>
                outerContext.foreach(_.enter())
                val zerosShape = resourceSafeShape(value)(TF.fromDataType(value.dataType))
                val acc = Basic.zeros(g.dataType, zerosShape)
                outerContext.foreach(_.exit())
                // TODO: [CONTROL_FLOW] Figure out if this is necessary.
                // acc.setShape(g.shape)
                acc
            }
          }
        }
        Op.nameScope("BackwardAccumulator") {
          enter()
          values += acc.name
          val enterAcc = ControlFlow.enter(acc, name, isConstant = false, parallelIterations)
          loopEnters += enterAcc
          val mergeAcc = ControlFlow.merge(Seq(enterAcc, enterAcc))._1.toOutput
          val switchAcc = ControlFlow.switch(mergeAcc, pivot)

          // TODO: [TYPES] !!! Super hacky. Remove in the future.
          implicit val ev: IsNotQuantized[T] = new IsNotQuantized[T] {}

          val addAcc = Math.add(switchAcc._2, g)
          val nextAcc = ControlFlow.nextIteration(addAcc)
          ControlFlow.updateInput(mergeAcc.op, 1, nextAcc)
          val exitAcc = ControlFlow.exit(switchAcc._1)
          loopExits += exitAcc
          exitResult(Seq(exitAcc))
          exitAcc
        }
      case g: OutputIndexedSlices[T] => Op.nameScope("BackwardAccumulator") {
        exit()
        // We create a zeros tensor with the right shape for the accumulator. If we don't know the full shape
        // statically, we will have to get the shape dynamically from the forward inference. Getting the shape right for
        // the zeros is only needed for the base case when the loop exits without running any iterations.
        outerContext.foreach(_.enter())
        val indicesAcc = Basic.zeros(g.dataType, Shape(1))
        val valuesAcc = {
          if (g.values.shape.isFullyDefined) {
            val zerosShape = Shape(1 +: g.values.shape.asArray.tail: _*)
            Basic.zeros(g.dataType, zerosShape)
          } else {
            val value = op.inputsSeq(0)
            // TODO: !!! [CONTROL_FLOW] Is this even necessary for obtaining the shape?
            outerContext match {
              case Some(context: WhileLoopContext) if context.gradientLoopState.isDefined =>
                // We are in a nested while loop.
                val forwardContext = context.gradientLoopState.get.forwardContext
                forwardContext.outerContext.foreach(_.enter())
                val zerosShape = Basic.concatenate[Long](
                  Seq(
                    Tensor(1L),
                    resourceSafeShape(value)(TF.fromDataType(value.dataType)).slice(1 ::)
                  ), axis = 0)
                forwardContext.outerContext.foreach(_.exit())
                val outerGradientLoopState = context.gradientLoopState.get.outerGradientLoopState.get
                val historyZerosShape = outerGradientLoopState.addForwardAccumulator(zerosShape)
                context.enter()
                val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyZerosShape, zerosShape)
                val acc = Basic.zeros(g.dataType, realShape)
                context.exit()
                // TODO: [CONTROL_FLOW] Figure out if this is necessary.
                // acc.setShape(g.values.shape)
                acc
              case _ =>
                val dataType = op.inputsSeq(0).dataType
                val zerosShape = Basic.concatenate[Long](
                  Seq(
                    Tensor(1L),
                    resourceSafeShape(op.inputsSeq(0))(TF.fromDataType(dataType)).slice(1 ::)
                  ), axis = 0)
                Basic.zeros[T](zerosShape)
            }
          }
        }
        val denseShapeAcc = Option(g.denseShape).map(shape => {
          if (shape.shape.isFullyDefined) {
            Basic.zeros[Long](shape.shape)
          } else {
            val dataType = op.inputsSeq(0).dataType
            Basic.zerosLike(
              Basic.shape(op.inputsSeq(0), optimize = false)(TF.fromDataType(dataType)),
              optimize = false)
          }
        })
        outerContext.foreach(_.exit())
        enter()
        values += indicesAcc.name
        values += valuesAcc.name
        denseShapeAcc.foreach(values += _.name)
        val initAcc = Seq[Output[Any]](indicesAcc, valuesAcc) ++
            denseShapeAcc.map(Seq[Output[Any]](_)).getOrElse(Seq.empty)
        // Set `useInputShape` to `false` since the accumulator tensors will grow in size. If `useInputShape` is `true`,
        // the `updateInput` call below will result in incompatible shapes.
        val enterAcc = initAcc.map(a => {
          ControlFlow.enter(
            input = a,
            frameName = name,
            isConstant = false,
            parallelIterations = parallelIterations,
            useInputShape = false
          )(TF.fromDataType(a.dataType))
        })
        // Manually set appropriate partial shapes.
        enterAcc.head.setShape(Shape(-1))
        if (valuesAcc.rank != -1 && valuesAcc.rank > 1)
          enterAcc(1).setShape(Shape(-1) ++ valuesAcc.shape(1 ::))
        else if (valuesAcc.rank != -1)
          enterAcc(1).setShape(Shape(-1))
        loopEnters ++= enterAcc
        val mergeAcc = enterAcc.map(a => {
          ControlFlow.merge(Seq(a, a))(TF.fromDataType(a.dataType))._1
        })
        val switchAcc = mergeAcc.map(a => {
          ControlFlow.switch(a, pivot)(TF.fromDataType(a.dataType))
        })

        // The actual accumulation.
        var addAcc = mutable.ListBuffer[Output[Any]](
          Basic.concatenate[Long](Seq(switchAcc(0)._2.asInstanceOf[Output[Long]], g.indices), 0),
          Basic.concatenate[T](Seq(switchAcc(1)._2.asInstanceOf[Output[T]], g.values), 0))
        denseShapeAcc.foreach(_ => {
          // TODO: [TYPES] Can handle types better here by not using sequences of tensors.
          // For the shape we just keep the maximum.
          addAcc += Math.maximum(
            g.denseShape,
            switchAcc(2)._2.asInstanceOf[Output[Long]]
          ).asInstanceOf[Output[Any]]
        })
        val nextAcc = addAcc.map(a => {
          ControlFlow.nextIteration(a)(TF.fromDataType(a.dataType))
        })
        mergeAcc.zip(nextAcc).foreach(a => ControlFlow.updateInput(a._1.op, 1, a._2))
        val exitAcc = switchAcc.map(a => {
          ControlFlow.exit(a._1)(TF.fromDataType(a._1.dataType))
        })
        loopExits ++= exitAcc
        exitResult(exitAcc)
        OutputIndexedSlices(
          indices = exitAcc(0).asInstanceOf[Output[Long]],
          values = exitAcc(1).asInstanceOf[Output[T]],
          denseShape = denseShapeAcc.map(_ => exitAcc(2).asInstanceOf[Output[Long]]).orNull)
      }
      case g: SparseOutput[T] => ???
    }
    result.asInstanceOf[OL[T]]
  }

  /** Returns the shape of `value` of the shape of the variable it points to. */
  private def resourceSafeShape[T: TF](value: Output[T]): Output[Long] = {
    if (value.dataType == RESOURCE) {
      var v = value
      while (v.op.inputsSeq.nonEmpty)
        v = v.op.inputsSeq(0).asInstanceOf[Output[T]]
      v.op.shapeAttribute("shape")
    } else {
      Basic.shape(value, optimize = false)
    }
  }

  override def toProto: GeneratedMessageV3 = {
    toProto(null)
  }

  /** Alias for `toWhileContextDef`. */
  override def toProto(exportScope: String = null): GeneratedMessageV3 = {
    toWhileContextDef(exportScope)
  }

  /** Constructs and returns a [[WhileContextDef]] object that represents this while-loop context.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[WhileContextDef]].
    */
  def toWhileContextDef(exportScope: String = null): WhileContextDef = {
    if (exportScope == null || name.startsWith(exportScope)) {
      // TODO: !!! Add `maximumIterations` when possible.
      WhileContextDef.newBuilder()
          .setContextName(Op.stripNameScope(exportScope, name))
          .setParallelIterations(parallelIterations)
          .setBackProp(enableBackPropagation)
          .setSwapMemory(swapMemory)
          .setPivotName(Op.stripNameScope(exportScope, pivot.name))
          .setPivotForPredName(Op.stripNameScope(exportScope, pivotForPredicate.name))
          .setPivotForBodyName(Op.stripNameScope(exportScope, pivotForBody.name))
          .addAllLoopEnterNames(loopEnters.map(e => Op.stripNameScope(exportScope, e.name)).asJava)
          .addAllLoopExitNames(loopExits.map(e => Op.stripNameScope(exportScope, e.name)).asJava)
          .setValuesDef(super.toValuesDef(exportScope))
          .build()
    } else {
      null
    }
  }
}

object WhileLoopContext {
  /** Returns the while-loop context to which `op` belongs. */
  private[control_flow] def getWhileLoopContext(op: Op[_, _]): Option[WhileLoopContext] = {
    op.controlFlowContext.flatMap(_.whileLoopContext())
  }

  /** Creates a next iteration op for `v` and adds a back edge from `v` to `m`. */
  @throws[IllegalArgumentException]
  private[ops] def addNextIterationAndBackEdge[T: TF, OL[A] <: OutputLike[A]](
      m: OL[T],
      v: OL[T],
      enforceShapeInvariant: Boolean = true
  ): OL[T] = {
    val result = (m, v) match {
      case (mm: Output[T], vv: Output[T]) =>
        val nextVV = ControlFlow.nextIteration(vv: Output[T])
        if (enforceShapeInvariant) {
          // Make sure the shapes of the loop outputs are correct. We do this before calling `updateInput`, which will
          // raise a less helpful error message if the types do not match.
          // TODO: Apply the same checks for the other cases, below.
          WhileLoopContext.enforceShapeInvariant(mm, nextVV)
        }
        ControlFlow.updateInput(mm.op, 1, nextVV)
        nextVV
      case (mm: OutputIndexedSlices[T], vv: OutputIndexedSlices[T]) =>
        val nextVV = ControlFlow.nextIteration(vv: OutputIndexedSlices[T])
        ControlFlow.updateInput(mm.values.op, 1, nextVV.values)
        ControlFlow.updateInput(mm.indices.op, 1, nextVV.indices)
        if (mm.denseShape != null) {
          if (nextVV.denseShape == null)
            throw new IllegalArgumentException(s"Output indexed slices '$nextVV' must have dense shape information.")
          else
            ControlFlow.updateInput(mm.denseShape.op, 1, nextVV.denseShape)
        }
        nextVV
      case (mm: SparseOutput[T], vv: SparseOutput[T]) =>
        val nextVV = ControlFlow.nextIteration(vv: SparseOutput[T])
        ControlFlow.updateInput(mm.values.op, 1, nextVV.values)
        ControlFlow.updateInput(mm.indices.op, 1, nextVV.indices)
        if (mm.denseShape != null) {
          if (nextVV.denseShape == null)
            throw new IllegalArgumentException(s"Sparse output '$nextVV' must have dense shape information.")
          else
            ControlFlow.updateInput(mm.denseShape.op, 1, nextVV.denseShape)
        }
        nextVV
      case (_, _) =>
        throw new IllegalArgumentException(
          "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the tensor types must match.")
    }
    result.asInstanceOf[OL[T]]
  }

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
  private[WhileLoopContext] def setShapeInvariants(
      inputTensors: Seq[OutputLike[_]],
      enterTensors: Seq[OutputLike[_]],
      shapes: Seq[Shape]
  ): Unit = {
    // Check that the shapes of the inputs are less than the shape invariants, and set the shapes of the enter tensors
    // to the shape invariants.
    for ((input, enter, shape) <- (inputTensors, enterTensors, shapes).zipped) {
      (input, enter) match {
        case (i: Output[_], e: Output[_]) =>
          if (!shapeLessThenOrEqual(i.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.name}' is not compatible with the initial shape of the " +
                  s"loop variable. It enters the loop with shape '${i.shape}', but the specified shape invariant " +
                  s"is '$shape'.")
          e.setShape(shape)
        case (i: OutputIndexedSlices[_], e: OutputIndexedSlices[_]) =>
          if (!shapeLessThenOrEqual(i.values.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.values.name}' is not compatible the initial shape of the " +
                  s"values tensor of these indexed slices. It enters the loop with shape '${i.values.shape}', but " +
                  s"the specified shape invariant is '$shape'.")
          e.values.setShape(shape)
          e.indices.setShape(Shape(shape(0)))
          if (e.denseShape != null)
            e.denseShape.setShape(Shape(shape.rank))
        case (i: SparseOutput[_], e: SparseOutput[_]) =>
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
  private[WhileLoopContext] def enforceShapeInvariant(
      mergeTensor: OutputLike[_],
      nextTensor: OutputLike[_]
  ): Unit = {
    (mergeTensor, nextTensor) match {
      case (merge: Output[_], next: Output[_]) =>
        if (!shapeLessThenOrEqual(next.shape, merge.shape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'${merge.shape}', but has shape '${next.shape}' after one iteration. Please provide shape " +
                s"invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' method of " +
                s"the loop variables.")
      case (merge: OutputIndexedSlices[_], next: OutputIndexedSlices[_]) =>
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
      case (merge: SparseOutput[_], next: SparseOutput[_]) =>
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
              "and the next tensor types must match.")
    }
  }

  //endregion Shape Invariants

  /** Creates a [[WhileLoopContext]] from the provided [[WhileContextDef]] object.
    *
    * @param  whileContextDef Serialized while-loop context object.
    * @param  importScope     Optional prefix that will be prepended to all op names in the cond context that is being
    *                         loaded from the provided [[WhileContextDef]].
    * @return Constructed [[WhileLoopContext]].
    */
  def fromWhileContextDef(
      whileContextDef: WhileContextDef,
      importScope: String = null
  ): WhileLoopContext = {
    // TODO: !!! Add `maximumIterations` when possible.
    val graph = Op.currentGraph
    val name = Op.prependNameScope(importScope, whileContextDef.getContextName)
    val parallelIterations = whileContextDef.getParallelIterations
    val enableBackPropagation = whileContextDef.getBackProp
    val swapMemory = whileContextDef.getSwapMemory
    val pivot = graph.getOutputByName(Op.prependNameScope(importScope, whileContextDef.getPivotName))
        .asInstanceOf[Output[Boolean]]
    val pivotForPredicate = graph.getOpByName(Op.prependNameScope(importScope, whileContextDef.getPivotForPredName))
    val pivotForBody = graph.getOpByName(Op.prependNameScope(importScope, whileContextDef.getPivotForBodyName))
    val loopEnters = mutable.Set(whileContextDef.getLoopEnterNamesList.asScala.map(name => {
      graph.getOutputByName(Op.prependNameScope(importScope, name))
    }): _*)
    val loopExits = mutable.Set(whileContextDef.getLoopExitNamesList.asScala.map(name => {
      graph.getOutputByName(Op.prependNameScope(importScope, name))
    }): _*)
    val (values, externalValues) = Context.fromValuesDef(whileContextDef.getValuesDef, importScope)
    val whileLoopContext = WhileLoopContext(
      None, parallelIterations, enableBackPropagation, swapMemory, None, pivot, pivotForPredicate, pivotForBody,
      loopEnters, loopExits, name)
    whileLoopContext.values ++= values
    whileLoopContext.externalValues ++= externalValues
    if (importScope != null) {
      val frameName = AttrValue.newBuilder().setS(ByteString.copyFromUtf8(whileLoopContext.name)).build()
      values.map(Op.currentGraph.findOp).filter(_.exists(ControlFlow.isLoopEnter)).foreach(_.foreach(op => {
        ControlFlow.setAttribute(op, "frame_name", frameName)
      }))
    }
    whileLoopContext
  }

  /** Key for collections of [[WhileLoopContext]]s. */
  trait CollectionKey extends Graph.Key[WhileLoopContext] {
    override def createCollectionDef(values: Set[WhileLoopContext], exportScope: String = null): CollectionDef = {
      val bytesListBuilder = BytesList.newBuilder()
      values
          .map(_.toProto(exportScope))
          .filter(_ != null)
          .foreach(s => bytesListBuilder.addValue(s.toByteString))
      CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
    }

    override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
      val kind = collectionDef.getKindCase
      if (kind != CollectionDef.KindCase.BYTES_LIST)
        throw new IllegalArgumentException(s"The '$name' collection should be stored as a byte list.")
      collectionDef.getBytesList.getValueList.asScala
          .foreach(s => graph.addToCollection(this)(
            WhileLoopContext.fromWhileContextDef(WhileContextDef.parseFrom(s), importScope)))
    }
  }

  /** Key to collect the [[WhileLoopContext]]s that have been created in the graph. */
  object WHILE_LOOP_CONTEXTS extends CollectionKey {
    override def name: String = "while_context"
  }
}

/** Type trait used for representing supported while-loop construct loop variable types. */
trait WhileLoopVariable[T] {
  type S

  /** Helper used by the RNN construction code to create default states. */
  def zero(
      batchSize: Output[Int],
      shape: S,
      name: String = "Zero"
  ): T

  def size(output: T): Int
  def outputs(output: T): Seq[Output[Any]]
  def shapes(shape: S): Seq[Shape]
  def fromOutputs(output: T, values: Seq[Output[Any]]): T = segmentOutputs(output, values)._1
  def segmentOutputs(output: T, values: Seq[Output[Any]]): (T, Seq[Output[Any]])
  def fromShapes(output: T, values: Seq[Shape]): S = segmentShapes(output, values)._1
  def segmentShapes(output: T, values: Seq[Shape]): (S, Seq[Shape])

  // TODO: [OPS] These "map" functions involve some runtime checking for the "Symbol" type that would be good to work around.
  def map(
      value: T,
      mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
  ): T

  def mapWithShape(
      value: T,
      shape: S,
      mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
  ): T
}

object WhileLoopVariable {
  type Aux[T, TS] = WhileLoopVariable[T] {
    type S = TS
  }

  implicit def fromOutput[T: TF]: Aux[Output[T], Shape] = {
    new WhileLoopVariable[Output[T]] {
      override type S = Shape

      // TODO: [TYPES] !!! The following is quite hacky.

      override def zero(
          batchSize: Output[Int],
          shape: Shape,
          name: String = "Zero"
      ): Output[T] = {
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        Op.nameScope(name) {
          val fullShape = Basic.concatenate(Seq(
            batchSize.expandDims(0).castTo[Long],
            shape.toOutput
          ), axis = 0)
          val zero = Basic.zeros[T](fullShape)
          zero.setShape(Shape(staticBatchSize) ++ shape)
          zero
        }
      }

      override def size(output: Output[T]): Int = {
        1
      }

      override def outputs(output: Output[T]): Seq[Output[Any]] = {
        Seq(output)
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def segmentOutputs(
          output: Output[T],
          values: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (values.head.asInstanceOf[Output[T]], values.tail)
      }

      override def segmentShapes(
          output: Output[T],
          values: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (values.head, values.tail)
      }

      override def map(
          value: Output[T],
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): Output[T] = {
        mapFn(value).asInstanceOf[Output[T]]
      }

      override def mapWithShape(
          value: Output[T],
          shape: S,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): Output[T] = {
        mapFn(value, shape).asInstanceOf[Output[T]]
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], Shape] = {
    new WhileLoopVariable[TensorArray[T]] {
      override type S = Shape

      override def zero(
          batchSize: Output[Int],
          shape: Shape,
          name: String = "Zero"
      ): TensorArray[T] = {
        ???
      }

      override def size(output: TensorArray[T]): Int = {
        1
      }

      override def outputs(output: TensorArray[T]): Seq[Output[Any]] = {
        Seq(output.flow)
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def segmentOutputs(
          output: TensorArray[T],
          values: Seq[Output[Any]]
      ): (TensorArray[T], Seq[Output[Any]]) = {
        val newTensorArray = output.copy(flow = values.head.asInstanceOf[Output[Float]])
        // TODO: !!! [TENSOR_ARRAY] What about colocate with?
        (newTensorArray, values.tail)
      }

      override def segmentShapes(
          output: TensorArray[T],
          values: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (values.head, values.tail)
      }

      override def map(
          value: TensorArray[T],
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): TensorArray[T] = {
        mapFn(value.asInstanceOf[TensorArray[Any]]).asInstanceOf[TensorArray[T]]
      }

      override def mapWithShape(
          value: TensorArray[T],
          shape: S,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): TensorArray[T] = {
        mapFn(value.asInstanceOf[TensorArray[Any]], shape).asInstanceOf[TensorArray[T]]
      }
    }
  }

  implicit def fromArray[T: ClassTag, TS: ClassTag](implicit
      ev: Aux[T, TS]
  ): Aux[Array[T], Array[TS]] = {
    new WhileLoopVariable[Array[T]] {
      override type S = Array[TS]

      override def zero(
          batchSize: Output[Int],
          shape: Array[TS],
          name: String
      ): Array[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }

      override def size(output: Array[T]): Int = {
        output.map(ev.size).sum
      }

      override def outputs(output: Array[T]): Seq[Output[Any]] = {
        output.toSeq.flatMap(ev.outputs)
      }

      override def shapes(shape: Array[TS]): Seq[Shape] = {
        shape.toSeq.flatMap(ev.shapes)
      }

      override def segmentOutputs(
          output: Array[T],
          values: Seq[Output[Any]]
      ): (Array[T], Seq[Output[Any]]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromOutputs(f._1, f._2)), values.drop(n))
      }

      override def segmentShapes(
          output: Array[T],
          values: Seq[Shape]
      ): (Array[TS], Seq[Shape]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromShapes(f._1, f._2)), values.drop(n))
      }

      override def map(
          value: Array[T],
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): Array[T] = {
        value.map(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: Array[T],
          shape: Array[TS],
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): Array[T] = {
        value.zip(shape).map(p => ev.mapWithShape(p._1, p._2, mapFn))
      }
    }
  }

  implicit def fromSeq[T, TS](implicit
      ev: Aux[T, TS]
  ): Aux[Seq[T], Seq[TS]] = {
    new WhileLoopVariable[Seq[T]] {
      override type S = Seq[TS]

      override def zero(
          batchSize: Output[Int],
          shape: Seq[TS],
          name: String
      ): Seq[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }

      override def size(output: Seq[T]): Int = {
        output.map(ev.size).sum
      }

      override def outputs(output: Seq[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs)
      }

      override def shapes(shape: Seq[TS]): Seq[Shape] = {
        shape.flatMap(ev.shapes)
      }

      override def segmentOutputs(
          output: Seq[T],
          values: Seq[Output[Any]]
      ): (Seq[T], Seq[Output[Any]]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromOutputs(f._1, f._2)), values.drop(n))
      }

      override def segmentShapes(
          output: Seq[T],
          values: Seq[Shape]
      ): (Seq[TS], Seq[Shape]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromShapes(f._1, f._2)), values.drop(n))
      }

      override def map(
          value: Seq[T],
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): Seq[T] = {
        value.map(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: Seq[T],
          shape: Seq[TS],
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): Seq[T] = {
        value.zip(shape.toSeq).map(p => ev.mapWithShape(p._1, p._2, mapFn))
      }
    }
  }

  implicit def fromMap[T, TS, MK](implicit
      ev: Aux[T, TS]
  ): Aux[Map[MK, T], Map[MK, TS]] = {
    new WhileLoopVariable[Map[MK, T]] {
      override type S = Map[MK, TS]

      override def zero(
          batchSize: Output[Int],
          shape: Map[MK, TS],
          name: String
      ): Map[MK, T] = {
        Op.nameScope(name) {
          shape.mapValues(ev.zero(batchSize, _))
        }
      }

      override def size(output: Map[MK, T]): Int = {
        output.values.map(ev.size).sum
      }

      override def outputs(output: Map[MK, T]): Seq[Output[Any]] = {
        output.values.flatMap(ev.outputs).toSeq
      }

      override def shapes(shape: Map[MK, TS]): Seq[Shape] = {
        shape.values.flatMap(ev.shapes).toSeq
      }

      override def segmentOutputs(
          output: Map[MK, T],
          values: Seq[Output[Any]]
      ): (Map[MK, T], Seq[Output[Any]]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(values.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.fromOutputs(f._1, f._2))).toMap, values.drop(n))
      }

      override def segmentShapes(
          output: Map[MK, T],
          values: Seq[Shape]
      ): (Map[MK, TS], Seq[Shape]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(values.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.fromShapes(f._1, f._2))).toMap, values.drop(n))
      }

      override def map(
          value: Map[MK, T],
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): Map[MK, T] = {
        value.mapValues(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: Map[MK, T],
          shape: Map[MK, TS],
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): Map[MK, T] = {
        value.map(p => p._1 -> ev.mapWithShape(p._2, shape(p._1), mapFn))
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil] = {
    new WhileLoopVariable[HNil] {
      override type S = HNil

      override def zero(
          batchSize: Output[Int],
          shape: HNil,
          name: String = "Zero"
      ): HNil = {
        HNil
      }

      override def size(output: HNil): Int = {
        0
      }

      override def outputs(output: HNil): Seq[Output[Any]] = {
        Seq.empty
      }

      override def shapes(shape: HNil): Seq[Shape] = {
        Seq.empty
      }

      override def segmentOutputs(
          output: HNil,
          values: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, values)
      }

      override def segmentShapes(
          output: HNil,
          values: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, values)
      }

      override def map(
          value: HNil,
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): HNil = {
        HNil
      }

      override def mapWithShape(
          value: HNil,
          shape: HNil,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[H, HS, T <: HList, TS <: HList](implicit
      evH: Strict[Aux[H, HS]],
      evT: Aux[T, TS]
  ): Aux[H :: T, HS :: TS] = {
    new WhileLoopVariable[H :: T] {
      override type S = HS :: TS

      override def zero(
          batchSize: Output[Int],
          shape: HS :: TS,
          name: String = "Zero"
      ): H :: T = {
        Op.nameScope(name) {
          evH.value.zero(batchSize, shape.head) ::
              evT.zero(batchSize, shape.tail)
        }
      }

      override def size(output: H :: T): Int = {
        evH.value.size(output.head) +
            evT.size(output.tail)
      }

      override def outputs(output: H :: T): Seq[Output[Any]] = {
        evH.value.outputs(output.head) ++
            evT.outputs(output.tail)
      }

      override def shapes(shape: HS :: TS): Seq[Shape] = {
        evH.value.shapes(shape.head) ++
            evT.shapes(shape.tail)
      }

      override def segmentOutputs(
          output: H :: T,
          values: Seq[Output[Any]]
      ): (H :: T, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = evH.value.segmentOutputs(output.head, values)
        val (tailOut, tailRemaining) = evT.segmentOutputs(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def segmentShapes(
          output: H :: T,
          values: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.segmentShapes(output.head, values)
        val (tailOut, tailRemaining) = evT.segmentShapes(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def map(
          value: H :: T,
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): H :: T = {
        evH.value.map(value.head, mapFn) ::
            evT.map(value.tail, mapFn)
      }

      override def mapWithShape(
          value: H :: T,
          shape: HS :: TS,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): H :: T = {
        evH.value.mapWithShape(value.head, shape.head, mapFn) ::
            evT.mapWithShape(value.tail, shape.tail, mapFn)
      }
    }
  }

  implicit def fromCoproduct[H, HS, T <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[Aux[H, HS]],
      evT: Aux[T, TS]
  ): Aux[H :+: T, HS :+: TS] = {
    new WhileLoopVariable[H :+: T] {
      override type S = HS :+: TS

      override def zero(
          batchSize: Output[Int],
          shape: HS :+: TS,
          name: String
      ): H :+: T = {
        shape match {
          case Inl(h) => Inl(evH.value.zero(batchSize, h, name))
          case Inr(t) => Inr(evT.zero(batchSize, t, name))
        }
      }

      override def size(output: H :+: T): Int = {
        output match {
          case Inl(h) => evH.value.size(h)
          case Inr(t) => evT.size(t)
        }
      }

      override def outputs(output: H :+: T): Seq[Output[Any]] = {
        output match {
          case Inl(h) => evH.value.outputs(h)
          case Inr(t) => evT.outputs(t)
        }
      }

      override def shapes(shape: HS :+: TS): Seq[Shape] = {
        shape match {
          case Inl(h) => evH.value.shapes(h)
          case Inr(t) => evT.shapes(t)
        }
      }

      override def segmentOutputs(
          output: H :+: T,
          values: Seq[Output[Any]]
      ): (H :+: T, Seq[Output[Any]]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.segmentOutputs(h, values)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.segmentOutputs(t, values)
            (Inr(result), remaining)
        }
      }

      override def segmentShapes(
          output: H :+: T,
          values: Seq[Shape]
      ): (HS :+: TS, Seq[Shape]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.segmentShapes(h, values)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.segmentShapes(t, values)
            (Inr(result), remaining)
        }
      }

      override def map(
          value: H :+: T,
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): H :+: T = {
        value match {
          case Inl(h) => Inl(evH.value.map(h, mapFn))
          case Inr(t) => Inr(evT.map(t, mapFn))
        }
      }

      override def mapWithShape(
          value: H :+: T,
          shape: HS :+: TS,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): H :+: T = {
        (value, shape) match {
          case (Inl(hv), Inl(hs)) => Inl(evH.value.mapWithShape(hv, hs, mapFn))
          case (Inr(tv), Inr(ts)) => Inr(evT.mapWithShape(tv, ts, mapFn))
          case _ => throw new IllegalStateException("Something went wrong while deriving implicit evidence.")
        }
      }
    }
  }

  implicit def fromProduct[P <: Product, PS <: Product, L <: HList, LS <: HList](implicit
      genP: Generic.Aux[P, L],
      evL: Strict[Aux[L, LS]],
      tuplerS: Tupler.Aux[LS, PS],
      genS: Generic.Aux[PS, LS]
  ): Aux[P, PS] = {
    new WhileLoopVariable[P] {
      override type S = PS

      override def zero(
          batchSize: Output[Int],
          shape: PS,
          name: String = "Zero"
      ): P = {
        genP.from(evL.value.zero(batchSize, genS.to(shape), name))
      }

      override def size(output: P): Int = {
        evL.value.size(genP.to(output))
      }

      override def outputs(output: P): Seq[Output[Any]] = {
        evL.value.outputs(genP.to(output))
      }

      override def shapes(shape: PS): Seq[Shape] = {
        evL.value.shapes(genS.to(shape))
      }

      override def segmentOutputs(
          output: P,
          values: Seq[Output[Any]]
      ): (P, Seq[Output[Any]]) = {
        val (out, remaining) = evL.value.segmentOutputs(genP.to(output), values)
        (genP.from(out), remaining)
      }

      override def segmentShapes(
          output: P,
          values: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evL.value.segmentShapes(genP.to(output), values)
        (tuplerS(out), remaining)
      }

      override def map(
          value: P,
          mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
      ): P = {
        genP.from(evL.value.map(genP.to(value), mapFn))
      }

      override def mapWithShape(
          value: P,
          shape: PS,
          mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
      ): P = {
        genP.from(evL.value.mapWithShape(genP.to(value), genS.to(shape), mapFn))
      }
    }
  }
}
