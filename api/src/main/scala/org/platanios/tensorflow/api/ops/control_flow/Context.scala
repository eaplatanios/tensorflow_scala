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

import org.platanios.tensorflow.api.ProtoSerializable
import org.platanios.tensorflow.api.ops._

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.framework.ValuesDef

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.util.DynamicVariable

/** Base class for control flow contexts.
  *
  * The usage pattern is a sequence of (Enter, Exit) ops followed by a final ExitResult op.
  *
  * We maintain the following state for control flow contexts during graph construction:
  *   1. [[OpCreationContext]] has a `controlFlowContext` field, which represents the current control flow context used
  *      to construct new ops. It can be changed by `context.enter()` and `context.exit()`.
  *   2. Each op has a `controlFlowContext` field, which is the control flow context to which the op belongs. It is set
  *      at the time the op is created and it is immutable.
  *   3. Each [[Context]] has an `outerContext` field, which is the control flow context in which this context is
  *      created. It is set at the time a context is created and it is immutable.
  *   4. Each [[Context]] has a `contextStack` field, which contains the control flow contexts pushed and popped by
  *      `context.enter()` and `context.exit()`.
  *
  * @param  values         Set of values that have already been seen in this context.
  * @param  externalValues Set of values referenced by but external to this context.
  * @author Emmanouil Antonios Platanios
  */
abstract class Context protected (
    private[control_flow] val values: mutable.Set[String] = mutable.Set.empty,
    private[control_flow] val externalValues: mutable.Map[String, Output] = mutable.Map.empty)
    (implicit context: DynamicVariable[OpCreationContext]) extends ProtoSerializable {
  (values -- externalValues.keySet)
      .map(_.split(":")(0))
      .map(Op.currentGraph.getOpByName(_))
      .foreach(_.controlFlowContext = Some(this))

  /** Name of this control flow context. */
  val name: String

  /** Contains the stack of control flow contexts that have been entered so far. */
  private[this] val contextStack: mutable.ListBuffer[Option[Context]] = mutable.ListBuffer.empty[Option[Context]]

  /** Control flow context containing this context. */
  val outerContext: Option[Context] = context.value.controlFlowContext

  /** Returns the control pivot op output for this context. */
  def controlPivot: Option[Op] = None

  /** Returns the while context containing this context. */
  def whileLoopContext: Option[WhileLoopContext] = outerContext.flatMap(_.whileLoopContext)

  /** Adds `op` to the current context. */
  def add(op: Op): Unit = addInternal(op)

  /** Adds `op` to the current context. */
  private[control_flow] def addInternal(op: Op): Unit = {
    if (op.numInputs == 0) {
      // Remove any external control dependencies on this op.
      val controlInputs = removeExternalControlEdges(op)
      // Add a control edge from the control pivot to this op.
      if (controlInputs.isEmpty)
        controlPivot.foreach(ControlFlow.addControlInput(op, _))
      op.outputs.foreach(values += _.name)
    } else {
      op.inputs.zipWithIndex.foreach({
        case (input, index) =>
          val realInput = add(input)
          if (realInput != input)
            ControlFlow.updateInput(op, index, realInput)
      })
      // Remove any external control dependencies on this op.
      removeExternalControlEdges(op)
      // Add a control dependency to the op if it only depends on loop invariants. That is to prevent loop invariants
      // from enabling ops that should not be executed.
      if (op.controlInputs.isEmpty &&
          ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient") ||
              op.inputs.forall(o => ControlFlow.isLoopConstantEnter(o.op))))
        controlPivot.foreach(ControlFlow.addControlInput(op, _))
      op.outputs.foreach(values += _.name)
    }
    if (outerContext.isDefined || !ControlFlow.isLoopExit(op)) {
      op.graph.preventFetching(op)
      op.outputs.foreach(op.graph.preventFeeding)
    }
  }

  /** Adds `output` to the current context and its outer context recursively. */
  def add(output: Output): Output

  /** Returns `true` if back-propagation is supported for this control flow context. */
  def backPropagate: Boolean

  /** Gradient loop state for this context, used for back-propagation. */
  def gradientLoopState: Option[GradientLoopState]

  /** Enters this control flow context. */
  def enter()(implicit context: DynamicVariable[OpCreationContext]): Unit = {
    contextStack.append(context.value.controlFlowContext)
    context.value = context.value.copy(controlFlowContext = Some(this))
  }

  /** Exits this control flow context. */
  def exit()(implicit context: DynamicVariable[OpCreationContext]): Unit = {
    context.value = context.value.copy(controlFlowContext = contextStack.remove(contextStack.size - 1))
  }

  /** Makes a sequence of tensors available in the outer context. */
  def exitResult(result: Seq[OutputLike]): Unit = outerContext.foreach(c => result.foreach(r => c.values += r.name))

  /** Remove any external control dependency on this op. */
  private[control_flow] def removeExternalControlEdges(op: Op): Set[Op] = {
    // A control input of 'op' is internal if it is in the same while loop context as the enclosing while loop context
    // of this context.
    val internalControlInputs = this.whileLoopContext match {
      case None => op.controlInputs
      case Some(wC) =>
        op.controlInputs.filter(i => ControlFlow.getOutputContext(i).exists(_.whileLoopContext.contains(wC)))
    }
    if (internalControlInputs.size != op.controlInputs.size) {
      ControlFlow.clearControlInputs(op)
      internalControlInputs.foreach(ControlFlow.addControlInput(op, _))
    }
    internalControlInputs
  }

  override def toProto: GeneratedMessageV3 = toProto(null)

  /** Alias for `toValuesDef`. */
  def toProto(exportScope: String): GeneratedMessageV3 = toValuesDef(exportScope)

  /** Constructs and returns a [[ValuesDef]] object that represents this control flow context.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[ValuesDef]].
    */
  def toValuesDef(exportScope: String = null): ValuesDef = Context.toValuesDef(values, externalValues, exportScope)
}

object Context {
  /** Constructs and returns a [[ValuesDef]] object that represents the provided values.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[ValuesDef]].
    */
  def toValuesDef(
      values: mutable.Set[String], externalValues: mutable.Map[String, Output],
      exportScope: String = null): ValuesDef = {
    val valuesDefBuilder = ValuesDef.newBuilder()
    values.foreach(v => valuesDefBuilder.addValues(Op.stripNameScope(exportScope, v)))
    externalValues.foreach(p => valuesDefBuilder.putExternalValues(
      Op.stripNameScope(exportScope, p._1), Op.stripNameScope(exportScope, p._2.name)))
    valuesDefBuilder.build()
  }

  /** Returns a set of values and a map of external values loaded from the provided [[ValuesDef]] object.
    *
    * @param  valuesDef   Serialized control flow context object.
    * @param  importScope Optional prefix that will be prepended to all op names in the objects that are being loaded
    *                     from the provided [[ValuesDef]].
    * @return Tuple containing the loaded values and external values that can be used to create control flow contexts.
    */
  def fromValuesDef(
      valuesDef: ValuesDef, importScope: String = null): (mutable.Set[String], mutable.Map[String, Output]) = {
    val values = valuesDef.getValuesList.asScala.toSet[String].map(v => Op.prependNameScope(importScope, v))
    val externalValues = valuesDef.getExternalValuesMap.asScala.map {
      case (k, v) =>
        val key = Op.prependNameScope(importScope, k)
        val value = Op.currentGraph.getOutputByName(Op.prependNameScope(importScope, v))
        (key, value)
    }
    (mutable.Set[String](values.toSeq: _*), externalValues)
  }

  /** Create a `zerosLike` op for the specified op output, while taking into account control flow contexts. */
  private[ops] def zerosLikeOutsideLoop(op: Op, index: Int): Output = {
    if (ControlFlow.isSwitch(op)) {
      op.controlFlowContext.filter(_.isInstanceOf[CondContext]).map(c => {
        val condContext = c.asInstanceOf[CondContext]
        // We are in a conditional context and so we use a switch to create zeros only when needed.
        val switch = {
          val switchOutput = ControlFlow.switch(op.inputs(0), condContext.predicate)
          condContext.branch.other.selectSwitchResult(switchOutput)
        }
        val shape = Basic.shape(switch, optimize = false)
        Basic.fill(op.outputs(index).dataType, shape)(0)
      }).getOrElse(Basic.zerosLike(op.outputs(index), optimize = false))
    } else {
      Basic.zerosLike(op.outputs(index), optimize = false)
    }
  }
}
