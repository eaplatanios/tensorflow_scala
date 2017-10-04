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
import org.platanios.tensorflow.api.ops.{Op, OpCreationContext, Output, OutputLike}

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

  private[this] val contextStack = mutable.Stack[Context]()

  /** Control flow context containing this context. */
  val outerContext: Option[Context] = context.value.controlFlowContext

  /** Returns the control pivot op output for this context. */
  def controlPivot: Option[Op] = None

  /** Returns the while context containing this context. */
  def whileLoopContext: Option[WhileLoopContext] = outerContext.flatMap(_.whileLoopContext)

  /** Adds `op` to the current context. */
  def add(op: Op): Unit

  /** Adds `op` to the current context. */
  private[control_flow] def addInternal(op: Op): Unit

  /** Adds `output` to the current context and its outer context recursively. */
  def add(output: Output): Output

  /** Notifies this context about an operator added to an inner context. */
  def addInnerOp(op: Op): Unit = ()

  /** Returns `true` if back-propagation is supported for this control flow context. */
  def backPropagate: Boolean

  /** Gradient loop state for this context, used for back-propagation. */
  def gradientLoopState: Option[GradientLoopState]

  /** Enters this control flow context. */
  def enter()(implicit context: DynamicVariable[OpCreationContext]): Unit = {
    context.value.controlFlowContext.foreach(contextStack.push)
    context.value = context.value.copy(controlFlowContext = Some(this))
  }

  /** Exits this control flow context. */
  def exit()(implicit context: DynamicVariable[OpCreationContext]): Unit = {
    context.value = context.value.copy(
      controlFlowContext = if (contextStack.nonEmpty) Some(contextStack.pop()) else None)
  }

  /** Makes a sequence of tensors available in the outer context. */
  def exitResult(result: Seq[OutputLike]): Unit = outerContext.foreach(c => result.foreach(r => c.values += r.name))

  /** Remove any external control dependency on this op. */
  private[control_flow] def removeExternalControlEdges(op: Op): Set[Op] = {
    // A control input of 'op' is internal if it is in the same while loop context as the enclosing while loop context
    // of this context.
    val internalControlInputs = this.whileLoopContext match {
      case Some(wC) => op.controlInputs.filter(i => ControlFlow.getOutputContext(i).exists(_.whileLoopContext.contains(wC)))
      case None => op.controlInputs
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
  def toValuesDef(exportScope: String): ValuesDef = {
    val valuesDefBuilder = ValuesDef.newBuilder()
    values.foreach(v => valuesDefBuilder.addValues(Op.stripNameScope(exportScope, v)))
    externalValues.foreach(p => valuesDefBuilder.putExternalValues(p._1, Op.stripNameScope(exportScope, p._2.name)))
    valuesDefBuilder.build()
  }
}

object Context {
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
}
