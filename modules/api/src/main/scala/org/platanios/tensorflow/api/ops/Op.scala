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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{DeviceSpecification, Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.control_flow.{Context, ControlFlow}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.utilities.using
import org.platanios.tensorflow.jni.{Op => NativeOp, Tensor => NativeTensor, TensorFlow => NativeLibrary}

import com.google.protobuf.ByteString
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.{AttrValue, NodeDef, OpDef}

import java.nio.charset.{Charset, StandardCharsets}

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.reflect.runtime.universe._
import scala.util.Try

/** Represents a graph node, or as we shall call it, an operation, that performs computation on tensors.
  *
  * An `Op` is a symbolic representation of the computation it performs. It is a node in a TensorFlow [[Graph]] that
  * takes zero or more `Op.Output` objects as input, and produces zero or more `Op.Output` objects as output. `Op`
  * objects are constructed by calling op creation functions, such as `tf.constant` or `tf.matmul`.
  *
  * For example, `val c = MathOps.matmul(a, b)` creates an `Op` of type `"MatMul"` that takes `Op.Output`s `a` and
  * `b` as input, and produces `Op.Output` `c` as output.
  *
  * @note The `Op.Input` class is simply a wrapper around an `Op` meant to represent one of its inputs. Actual op inputs
  *       have type `Op.Output` since they represent outputs of other ops. Currently, `Op.Input` is only useful for
  *       representing consumers of an `Op`'s outputs.
  *
  *       After the graph has been launched in a [[Session]], an `Op` can be executed by using `Session.run`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class Op[+I: Op.OpInput, +O: Op.OpOutput] private (
    graph: Graph,
    private[ops] val originalInput: Option[I],
    private[api] val nativeHandle: Long
) {
  // Update the ops cache of the graph with the current op.
  graph.opsCache.update(nativeHandle, this.asUntyped)

  /** Name of the op. */
  lazy val name: String = {
    using(graph.reference) { _ =>
      NativeOp.name(nativeHandle)
    }
  }

  /** Type of the op (i.e., the name of the computation performed by the operation). */
  lazy val opType: String = {
    using(graph.reference) { _ =>
      NativeOp.opType(nativeHandle)
    }
  }

  /** Device in which the op tensors are stored and where all computations for this op are performed. */
  def device: String = {
    using(graph.reference) { _ =>
      val nativeDevice = NativeOp.device(nativeHandle)
      if (nativeDevice == null)
        ""
      else
        nativeDevice
    }
  }

  /** Colocation ops for this op (i.e., ops guaranteed to be placed on the same device). */
  def colocationOps: Set[UntypedOp] = {
    using(graph.reference) { _ =>
      Try(NativeOp.getAttrStringList(nativeHandle, COLOCATION_OPS_ATTRIBUTE_NAME))
          .map(_.toSet[String]
              .filter(_.startsWith(COLOCATION_OPS_ATTRIBUTE_PREFIX))
              .map(opName => graph.findOp(opName.substring(COLOCATION_OPS_ATTRIBUTE_PREFIX.length)).get))
          .getOrElse(Set.empty)
    }
  }

  private def updateColocationOps(colocationOpNames: Set[String]): Unit = {
    val builder = AttrValue.newBuilder()
    builder.setList(builder.getListBuilder.addAllS(
      colocationOps.toSeq.map(opName => COLOCATION_OPS_ATTRIBUTE_PREFIX + opName)
          .sorted.map(ByteString.copyFrom(_, StandardCharsets.ISO_8859_1)).asJava))
    using(graph.reference) { r =>
      NativeLibrary.setAttributeProto(
        r.nativeHandle, nativeHandle, COLOCATION_OPS_ATTRIBUTE_NAME,
        builder.build().toByteArray)
    }
  }

  private[ops] var gradientFn: Option[Gradients.UntypedGradientFn] = None

  def setGradient[GI >: I : Op.OpInput, GO >: O : Op.OpOutput](
      gradientFn: Gradients.GradientFn[I, O, GI, GO]
  ): Unit = {
    this.gradientFn = Some(Gradients.convertGradientFn(gradientFn))
  }

  def hasGradient: Boolean = {
    gradientFn.isDefined
  }

  private[ops] var controlFlowContext: Option[Context] = {
    None
  }

  // The following caching of inputs and control inputs is done so that we can improve performance by avoiding redundant
  // JNI calls, while at the same time allowing the control flow package to modify inputs and control inputs of ops.

  private[Op] var _numInputs       : Int                = _loadNumInputs()
  private[Op] var _inputs          : Array[Output[Any]] = _loadInputs()
  private[Op] var _numControlInputs: Int                = _loadNumControlInputs()
  private[Op] var _controlInputs   : Set[UntypedOp]     = _loadControlInputs()

  private[ops] def _reloadNumInputs(): Unit = _numInputs = _loadNumInputs()
  private[ops] def _reloadInputs(): Unit = _inputs = _loadInputs()
  private[ops] def _reloadNumControlInputs(): Unit = _numControlInputs = _loadNumControlInputs()
  private[ops] def _reloadControlInputs(): Unit = _controlInputs = _loadControlInputs()

  private[Op] def _loadNumInputs(): Int = {
    using(graph.reference) { _ =>
      NativeOp.numInputs(nativeHandle)
    }
  }

  private[Op] def _loadInputs(): Array[Output[Any]] = {
    using(graph.reference) { _ =>
      NativeOp.inputs(nativeHandle).map(i => {
        val op = graph.opsCache.getOrElseUpdate(
          i.opHandle,
          Op[Seq[Output[Any]], Seq[Output[Any]]](
            graph = graph,
            originalInput = None,
            nativeHandle = i.opHandle))
        op._outputs(i.outputIndex)
      })
    }
  }

  private[Op] def _loadNumControlInputs(): Int = {
    using(graph.reference) { _ =>
      NativeOp.numControlInputs(nativeHandle)
    }
  }

  private[Op] def _loadControlInputs(): Set[UntypedOp] = {
    using(graph.reference) { _ =>
      NativeOp.controlInputs(nativeHandle).map(handle => {
        graph.opsCache.getOrElseUpdate(
          handle,
          Op[Seq[Output[Any]], Seq[Output[Any]]](
            graph = graph,
            originalInput = None,
            nativeHandle = handle))
      }).toSet
    }
  }

  private[Op] def _outputs: Array[Output[Any]] = {
    (0 until numOutputs).map(i => {
      Output[Any](op = this.asUntyped, index = i)
    }).toArray
  }

  /** Number of inputs to this op (i.e., number of tensors fed as input to this op). */
  def numInputs: Int = {
    _numInputs
  }

  /** Input of this op. */
  def input: I = {
    implicitly[Op.OpInput[I]].fromOutputs(_inputs, originalInput)
  }

  /** Inputs of this op. */
  def inputsSeq: Seq[Output[Any]] = {
    _inputs
  }

  /** Number of control inputs to this op. These are ops that are guaranteed to execute before this op. */
  def numControlInputs: Int = {
    _numControlInputs
  }

  /** Control inputs of this op. These are ops that are guaranteed to execute before this op. */
  def controlInputs: Set[UntypedOp] = {
    _controlInputs
  }

  /** Number of tensors produced by this operation. */
  def numOutputs: Int = {
    using(graph.reference) { _ =>
      NativeOp.numOutputs(nativeHandle)
    }
  }

  /** Output of this op. */
  def output: O = {
    implicitly[Op.OpOutput[O]].fromOutputs(_outputs)
  }

  /** Inputs of this op. */
  def outputsSeq: Seq[Output[Any]] = {
    _outputs
  }

  /** Gets the (current) number of control outputs of this op. These are ops that are guaranteed to start executing
    * after this op finishes executing.
    *
    * @note A concurrent modification of the graph can change the number of control outputs of this op.
    * @return Current number of control outputs of this op.
    */
  def numControlOutputs: Int = {
    using(graph.reference) { _ =>
      NativeOp.numControlOutputs(nativeHandle)
    }
  }

  /** Gets the (current) control outputs of this op. These are ops that are guaranteed to start executing after this op
    * finishes executing.
    *
    * @note A concurrent modification of the graph can change the number of control outputs of this op.
    * @return Current control outputs of this op.
    */
  def controlOutputs: Set[UntypedOp] = {
    val controlOutputHandles = using(graph.reference) { _ =>
      NativeOp.controlOutputs(nativeHandle)
    }
    controlOutputHandles.map(handle => {
      graph.opsCache.getOrElseUpdate(
        handle,
        Op[Seq[Output[Any]], Seq[Output[Any]]](
          graph = graph,
          originalInput = None,
          nativeHandle = handle))
    }).toSet
  }

  //region Attributes

  /** Gets the value of a string-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no string attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringAttribute(name: String): String = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrString(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no string attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a string-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no string array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringArrayAttribute(name: String): Array[String] = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrStringList(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no string array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a long-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no long attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def longAttribute(name: String): Long = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrInt(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no long attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a long-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no long array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def longArrayAttribute(name: String): Array[Long] = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrIntList(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no long array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a float-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no float attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def floatAttribute(name: String): Float = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrFloat(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no float attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a float-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no float array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def floatArrayAttribute(name: String): Array[Float] = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrFloatList(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no float array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a boolean-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no boolean attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def booleanAttribute(name: String): Boolean = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrBool(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no boolean attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a boolean-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no boolean array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def booleanArrayAttribute(name: String): Array[Boolean] = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrBoolList(nativeHandle, name)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no boolean array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a data type-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no data type attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def dataTypeAttribute[T](name: String): DataType[T] = {
    using(graph.reference) { _ =>
      try {
        DataType.fromCValue[T](NativeOp.getAttrType(nativeHandle, name))
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no data type attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a data type-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no data type array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def dataTypeArrayAttribute(name: String): Array[DataType[Any]] = {
    using(graph.reference) { _ =>
      try {
        NativeOp.getAttrTypeList(nativeHandle, name).map(DataType.fromCValue)
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no data type array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a tensor-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no tensor attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def tensorAttribute[T](name: String): Tensor[T] = {
    using(graph.reference) { _ =>
      try {
        Tensor.fromHostNativeHandle[T](NativeOp.getAttrTensor(nativeHandle, name))
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no tensor attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  /** Gets the value of a shape-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no shape attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def shapeAttribute(name: String): Shape = {
    using(graph.reference) { _ =>
      try {
        Shape.fromSeq(NativeOp.getAttrShape(nativeHandle, name).map(_.toInt))
      } catch {
        case e: Exception => throw new IllegalArgumentException(
          s"Op has no shape attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
      }
    }
  }

  //endregion Attributes

  //region Serialization

  /** Constructs and returns a [[OpDef]] object, which is a serialized version of this op. */
  def toOpDef: OpDef = {
    OpDef.parseFrom(NativeOp.toOpDef(graph.nativeHandle, opType))
  }

  /** Constructs and returns a [[NodeDef]] object, which is a serialized version of this op. */
  def toNodeDef: NodeDef = {
    NodeDef.parseFrom(NativeOp.toNodeDef(nativeHandle))
  }

  //endregion Serialization

  def asUntyped: UntypedOp = {
    this.asInstanceOf[Op[Seq[Output[Any]], Seq[Output[Any]]]]
  }

  override def toString: String = {
    name
  }

  // TODO: [OP] Better implementations for equals and hashCode.

  override def equals(that: Any): Boolean = that match {
    case that: Op[_, _] => this.graph == that.graph && this.nativeHandle == that.nativeHandle
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    result = prime * result + graph.hashCode
    result = prime * result + nativeHandle.hashCode
    result
  }
}

object Op {
  private[ops] val logger = Logger(LoggerFactory.getLogger("Op"))

  private[ops] trait API {
    type Op[+I, +O] = ops.Op[I, O]

    val Op: ops.Op.type = ops.Op

    type OpCreationContext = ops.GraphConstructionScope
    type OpSpecification = ops.OpSpecification

    def currentGraph: Graph = Op.currentGraph
    def currentNameScope: String = Op.currentNameScope
    def currentDevice: String = Op.currentDevice
    def currentDeviceFunction: OpSpecification => String = Op.currentDeviceFunction
    def currentColocationOps: Set[UntypedOp] = Op.currentColocationOps
    def currentControlDependencies: Set[UntypedOp] = Op.currentControlDependencies
    def currentAttributes: Map[String, Any] = Op.currentAttributes
    def currentContainer: String = Op.currentContainer

    def currentGraphRandomSeed(opSeed: Option[Int] = None): (Option[Int], Option[Int]) = {
      Op.currentGraphRandomSeed(opSeed)
    }

    def setCurrentGraphRandomSeed(value: Int): Unit = {
      Op.setCurrentGraphRandomSeed(value)
    }

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: String = "",
        deviceFunction: OpSpecification => String = _.device, colocationOps: Set[UntypedOp] = null,
        controlDependencies: Set[UntypedOp] = null, attributes: Map[String, Any] = null, container: String = null
    )(block: => R): R = {
      Op.createWith(
        graph, nameScope, device, deviceFunction, colocationOps, controlDependencies, attributes, container)(block)
    }

    def nameScope[R](nameScope: String)(block: => R): R = {
      Op.nameScope(nameScope)(block)
    }

    def device[R](
        device: String = "",
        deviceFunction: OpSpecification => String = _.device
    )(block: => R): R = {
      Op.device(device, deviceFunction)(block)
    }

    def colocateWith[R](
        colocationOps: Set[UntypedOp],
        ignoreExisting: Boolean = false
    )(block: => R): R = {
      Op.colocateWith(colocationOps, ignoreExisting)(block)
    }

    def initializationScope[R](block: => R): R = {
      Op.initializationScope(block)
    }

    def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): UntypedOp = {
      Op.currentGraph.globalVariablesInitializer(name)
    }

    def localVariablesInitializer(name: String = "LocalVariablesInitializer"): UntypedOp = {
      Op.currentGraph.localVariablesInitializer(name)
    }

    def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): UntypedOp = {
      Op.currentGraph.modelVariablesInitializer(name)
    }

    def metricVariablesInitializer(name: String = "MetricVariablesInitializer"): UntypedOp = {
      Op.currentGraph.metricVariablesInitializer(name)
    }

    def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): UntypedOp = {
      Op.currentGraph.trainableVariablesInitializer(name)
    }
  }

  //region Type Traits

  sealed trait OpInput[T] {
    @inline def fromOutputs(outputs: Seq[Output[Any]], reference: Option[T]): T
    @inline def toBuilderInputs(value: T): Seq[Builder.Input]
    @inline def toOutputLikes(value: T): Seq[OutputLike[Any]]
  }

  object OpInput {
    implicit val unitEvidence: OpInput[Unit] = new OpInput[Unit] {
      @inline override def fromOutputs(outputs: Seq[Output[Any]], reference: Option[Unit]): Unit = {
        ()
      }

      @inline override def toBuilderInputs(value: Unit): Seq[Builder.Input] = {
        Seq.empty
      }

      @inline override def toOutputLikes(value: Unit): Seq[OutputLike[Any]] = {
        Seq.empty
      }
    }

    implicit def opInputPrimitiveEvidence[T: OpInputPrimitive]: OpInput[T] = {
      new OpInput[T] {
        @inline override def fromOutputs(outputs: Seq[Output[Any]], reference: Option[T]): T = {
          implicitly[OpInputPrimitive[T]].fromOutputs(outputs, reference)._1
        }

        @inline override def toBuilderInputs(value: T): Seq[Builder.Input] = {
          implicitly[OpInputPrimitive[T]].toBuilderInput(value)
        }

        @inline override def toOutputLikes(value: T): Seq[OutputLike[Any]] = {
          implicitly[OpInputPrimitive[T]].toOutputLikes(value)
        }
      }
    }

    implicit def opInputPrimitiveTuple2Evidence[T1, T2](implicit
        evT1: OpInputPrimitive[T1],
        evT2: OpInputPrimitive[T2]
    ): OpInput[(T1, T2)] = {
      new OpInput[(T1, T2)] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[(T1, T2)]
        ): (T1, T2) = {
          val (value1, remaining1) = evT1.fromOutputs(outputs, reference.map(_._1))
          val (value2, _) = evT2.fromOutputs(remaining1, reference.map(_._2))
          (value1, value2)
        }

        @inline override def toBuilderInputs(value: (T1, T2)): Seq[Builder.Input] = {
          evT1.toBuilderInput(value._1) ++ evT2.toBuilderInput(value._2)
        }

        @inline override def toOutputLikes(value: (T1, T2)): Seq[OutputLike[Any]] = {
          evT1.toOutputLikes(value._1) ++ evT2.toOutputLikes(value._2)
        }
      }
    }

    implicit def opInputPrimitiveTuple3Evidence[T1, T2, T3](implicit
        evT1: OpInputPrimitive[T1],
        evT2: OpInputPrimitive[T2],
        evT3: OpInputPrimitive[T3]
    ): OpInput[(T1, T2, T3)] = {
      new OpInput[(T1, T2, T3)] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[(T1, T2, T3)]
        ): (T1, T2, T3) = {
          val (value1, remaining1) = evT1.fromOutputs(outputs, reference.map(_._1))
          val (value2, remaining2) = evT2.fromOutputs(remaining1, reference.map(_._2))
          val (value3, _) = evT3.fromOutputs(remaining2, reference.map(_._3))
          (value1, value2, value3)
        }

        @inline override def toBuilderInputs(value: (T1, T2, T3)): Seq[Builder.Input] = {
          evT1.toBuilderInput(value._1) ++
              evT2.toBuilderInput(value._2) ++
              evT3.toBuilderInput(value._3)
        }

        @inline override def toOutputLikes(value: (T1, T2, T3)): Seq[OutputLike[Any]] = {
          evT1.toOutputLikes(value._1) ++
              evT2.toOutputLikes(value._2) ++
              evT3.toOutputLikes(value._3)
        }
      }
    }

    implicit def opInputPrimitiveTuple4Evidence[T1, T2, T3, T4](implicit
        evT1: OpInputPrimitive[T1],
        evT2: OpInputPrimitive[T2],
        evT3: OpInputPrimitive[T3],
        evT4: OpInputPrimitive[T4]
    ): OpInput[(T1, T2, T3, T4)] = {
      new OpInput[(T1, T2, T3, T4)] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[(T1, T2, T3, T4)]
        ): (T1, T2, T3, T4) = {
          val (value1, remaining1) = evT1.fromOutputs(outputs, reference.map(_._1))
          val (value2, remaining2) = evT2.fromOutputs(remaining1, reference.map(_._2))
          val (value3, remaining3) = evT3.fromOutputs(remaining2, reference.map(_._3))
          val (value4, _) = evT4.fromOutputs(remaining3, reference.map(_._4))
          (value1, value2, value3, value4)
        }

        @inline override def toBuilderInputs(value: (T1, T2, T3, T4)): Seq[Builder.Input] = {
          evT1.toBuilderInput(value._1) ++
              evT2.toBuilderInput(value._2) ++
              evT3.toBuilderInput(value._3) ++
              evT4.toBuilderInput(value._4)
        }

        @inline override def toOutputLikes(value: (T1, T2, T3, T4)): Seq[OutputLike[Any]] = {
          evT1.toOutputLikes(value._1) ++
              evT2.toOutputLikes(value._2) ++
              evT3.toOutputLikes(value._3) ++
              evT4.toOutputLikes(value._4)
        }
      }
    }

    implicit def opInputPrimitiveTuple5Evidence[T1, T2, T3, T4, T5](implicit
        evT1: OpInputPrimitive[T1],
        evT2: OpInputPrimitive[T2],
        evT3: OpInputPrimitive[T3],
        evT4: OpInputPrimitive[T4],
        evT5: OpInputPrimitive[T5]
    ): OpInput[(T1, T2, T3, T4, T5)] = {
      new OpInput[(T1, T2, T3, T4, T5)] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[(T1, T2, T3, T4, T5)]
        ): (T1, T2, T3, T4, T5) = {
          val (value1, remaining1) = evT1.fromOutputs(outputs, reference.map(_._1))
          val (value2, remaining2) = evT2.fromOutputs(remaining1, reference.map(_._2))
          val (value3, remaining3) = evT3.fromOutputs(remaining2, reference.map(_._3))
          val (value4, remaining4) = evT4.fromOutputs(remaining3, reference.map(_._4))
          val (value5, _) = evT5.fromOutputs(remaining4, reference.map(_._5))
          (value1, value2, value3, value4, value5)
        }

        @inline override def toBuilderInputs(value: (T1, T2, T3, T4, T5)): Seq[Builder.Input] = {
          evT1.toBuilderInput(value._1) ++
              evT2.toBuilderInput(value._2) ++
              evT3.toBuilderInput(value._3) ++
              evT4.toBuilderInput(value._4) ++
              evT5.toBuilderInput(value._5)
        }

        @inline override def toOutputLikes(value: (T1, T2, T3, T4, T5)): Seq[OutputLike[Any]] = {
          evT1.toOutputLikes(value._1) ++
              evT2.toOutputLikes(value._2) ++
              evT3.toOutputLikes(value._3) ++
              evT4.toOutputLikes(value._4) ++
              evT5.toOutputLikes(value._5)
        }
      }
    }
  }

  sealed trait OpInputPrimitive[T] {
    @inline def fromOutputs(outputs: Seq[Output[Any]], reference: Option[T]): (T, Seq[Output[Any]])
    @inline def toBuilderInput(value: T): Seq[Builder.Input]
    @inline def toOutputLikes(value: T): Seq[OutputLike[Any]]
  }

  object OpInputPrimitive {

    // TODO: [OPS] Make this a bit prettier.

    implicit def outputEvidence[T]: OpInputPrimitive[Output[T]] = {
      new OpInputPrimitive[Output[T]] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[Output[T]]
        ): (Output[T], Seq[Output[Any]]) = {
          (outputs.head.asInstanceOf[Output[T]], outputs.tail)
        }

        @inline override def toBuilderInput(value: Output[T]): Seq[Builder.Input] = {
          Seq(Builder.InputTensor(value))
        }

        @inline override def toOutputLikes(value: Output[T]): Seq[OutputLike[Any]] = {
          Seq(value)
        }
      }
    }

    implicit def outputLikeEvidence[T, O <: OutputLike[T] : TypeTag]: OpInputPrimitive[O] = {
      new OpInputPrimitive[O] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[O]
        ): (O, Seq[Output[Any]]) = {
          val (value, remaining) = reference match {
            case Some(_: Output[Any]) =>
              (outputs.head, outputs.tail)
            case Some(_: OutputIndexedSlices[Any]) =>
              (OutputIndexedSlices(
                indices = outputs(0).asInstanceOf[Output[Long]],
                values = outputs(1),
                denseShape = outputs(2).asInstanceOf[Output[Long]]),
                  outputs.drop(3))
            case Some(_: SparseOutput[Any]) =>
              (SparseOutput(
                indices = outputs(0).asInstanceOf[Output[Long]],
                values = outputs(1),
                denseShape = outputs(2).asInstanceOf[Output[Long]]),
                  outputs.drop(3))
            case None => typeOf[O] match {
              case o if o <:< typeOf[Output[Any]] =>
                (outputs.head, outputs.tail)
              case o if o <:< typeOf[OutputIndexedSlices[Any]] =>
                (OutputIndexedSlices(
                  indices = outputs(0).asInstanceOf[Output[Long]],
                  values = outputs(1),
                  denseShape = outputs(2).asInstanceOf[Output[Long]]),
                    outputs.drop(3))
              case o if o <:< typeOf[SparseOutput[Any]] =>
                (SparseOutput(
                  indices = outputs(0).asInstanceOf[Output[Long]],
                  values = outputs(1),
                  denseShape = outputs(2).asInstanceOf[Output[Long]]),
                    outputs.drop(3))
              case _ =>
                (outputs.head, outputs.tail)
            }
          }
          (value.asInstanceOf[O], remaining)
        }

        @inline override def toBuilderInput(value: O): Seq[Builder.Input] = {
          value match {
            case o: Output[Any] =>
              Seq(Builder.InputTensor(o))
            case o: OutputIndexedSlices[Any] =>
              Seq(
                Builder.InputTensor(o.indices),
                Builder.InputTensor(o.values),
                Builder.InputTensor(o.denseShape))
            case o: SparseOutput[Any] =>
              Seq(
                Builder.InputTensor(o.indices),
                Builder.InputTensor(o.values),
                Builder.InputTensor(o.denseShape))
            case _ =>
              val v = value.asInstanceOf[Output[Any]]
              Seq(Builder.InputTensor(v))
          }
        }

        @inline override def toOutputLikes(value: O): Seq[OutputLike[Any]] = {
          Seq(value)
        }
      }
    }

    // implicit def outputEvidence[T]: OpInputPrimitive[Output[T]] = {
    //   new OpInputPrimitive[Output[T]] {
    //     @inline override def fromOutputs(
    //         outputs: Seq[Output[_]],
    //         reference: Option[Output[T]]
    //     ): (Output[T], Seq[Output[_]]) = {
    //       (outputs.head.asInstanceOf[Output[T]], outputs.tail)
    //     }
    //
    //     @inline override def toBuilderInput(value: Output[T]): Seq[Builder.Input] = {
    //       Seq(Builder.InputTensor(value))
    //     }
    //   }
    // }
    //
    // implicit def outputIndexedSlicesEvidence[T]: OpInputPrimitive[OutputIndexedSlices[T]] = {
    //   new OpInputPrimitive[OutputIndexedSlices[T]] {
    //     @inline override def fromOutputs(
    //         outputs: Seq[Output[_]],
    //         reference: Option[OutputIndexedSlices[T]]
    //     ): (OutputIndexedSlices[T], Seq[Output[_]]) = {
    //       (OutputIndexedSlices(
    //         indices = outputs(0).asInstanceOf[Output[Long]],
    //         values = outputs(1).asInstanceOf[Output[T]],
    //         denseShape = outputs(2).asInstanceOf[Output[Long]]),
    //           outputs.tail)
    //     }
    //
    //     @inline override def toBuilderInput(value: OutputIndexedSlices[T]): Seq[Builder.Input] = {
    //       Seq(
    //         Builder.InputTensor(value.indices),
    //         Builder.InputTensor(value.values),
    //         Builder.InputTensor(value.denseShape))
    //     }
    //   }
    // }
    //
    // implicit def sparseOutputEvidence[T]: OpInputPrimitive[SparseOutput[T]] = {
    //   new OpInputPrimitive[SparseOutput[T]] {
    //     @inline override def fromOutputs(
    //         outputs: Seq[Output[_]],
    //         reference: Option[SparseOutput[T]]
    //     ): (SparseOutput[T], Seq[Output[_]]) = {
    //       (SparseOutput(
    //         indices = outputs(0).asInstanceOf[Output[Long]],
    //         values = outputs(1).asInstanceOf[Output[T]],
    //         denseShape = outputs(2).asInstanceOf[Output[Long]]),
    //           outputs.tail)
    //     }
    //
    //     @inline override def toBuilderInput(value: SparseOutput[T]): Seq[Builder.Input] = {
    //       Seq(
    //         Builder.InputTensor(value.indices),
    //         Builder.InputTensor(value.values),
    //         Builder.InputTensor(value.denseShape))
    //     }
    //   }
    // }

    implicit def seqOutputEvidence[T]: OpInputPrimitive[Seq[Output[T]]] = {
      new OpInputPrimitive[Seq[Output[T]]] {
        @inline override def fromOutputs(
            outputs: Seq[Output[Any]],
            reference: Option[Seq[Output[T]]]
        ): (Seq[Output[T]], Seq[Output[Any]]) = {
          reference match {
            case Some(r) =>
              val parts = outputs.splitAt(r.size)
              (parts._1.map(_.asInstanceOf[Output[T]]), parts._2)
            case None =>
              (outputs.map(_.asInstanceOf[Output[T]]), Seq.empty[Output[Any]])
          }
        }

        @inline override def toBuilderInput(value: Seq[Output[T]]): Seq[Builder.Input] = {
          Seq(Builder.InputTensorList(value))
        }

        @inline override def toOutputLikes(value: Seq[Output[T]]): Seq[OutputLike[Any]] = {
          value
        }
      }
    }
  }

  sealed trait OpOutput[T] {
    @inline def fromOutputs(outputs: Seq[Output[Any]]): T
    @inline def fromOutputLikes(outputs: Seq[OutputLike[Any]]): T
  }

  object OpOutput {
    implicit val unitEvidence: OpOutput[Unit] = {
      new OpOutput[Unit] {
        @inline override def fromOutputs(outputs: Seq[Output[Any]]): Unit = {
          ()
        }

        @inline override def fromOutputLikes(outputs: Seq[OutputLike[Any]]): Unit = {
          Seq.empty
        }
      }
    }

    implicit def outputEvidence[T]: OpOutput[Output[T]] = {
      new OpOutput[Output[T]] {
        @inline override def fromOutputs(outputs: Seq[Output[Any]]): Output[T] = {
          outputs.head.asInstanceOf[Output[T]]
        }

        @inline override def fromOutputLikes(outputs: Seq[OutputLike[Any]]): Output[T] = {
          outputs.head.toOutput.asInstanceOf[Output[T]]
        }
      }
    }

    implicit def outputLikeEvidence[T, O <: OutputLike[T] : TypeTag]: OpOutput[O] = {
      new OpOutput[O] {
        @inline override def fromOutputs(outputs: Seq[Output[Any]]): O = {
          val value = typeOf[O] match {
            case o if o <:< typeOf[Output[Any]] =>
              outputs.head
            case o if o <:< typeOf[OutputIndexedSlices[Any]] =>
              OutputIndexedSlices(
                indices = outputs(0).asInstanceOf[Output[Long]],
                values = outputs(1),
                denseShape = outputs(2).asInstanceOf[Output[Long]])
            case o if o <:< typeOf[SparseOutput[Any]] =>
              SparseOutput(
                indices = outputs(0).asInstanceOf[Output[Long]],
                values = outputs(1),
                denseShape = outputs(2).asInstanceOf[Output[Long]])
            case _ => ???
            // TODO: [OPS] Is this correct?
            // outputs.head
          }
          value.asInstanceOf[O]
        }

        @inline override def fromOutputLikes(outputs: Seq[OutputLike[Any]]): O = {
          val value = typeOf[O] match {
            case o if o <:< typeOf[Output[Any]] =>
              outputs.head.toOutput
            case o if o <:< typeOf[OutputIndexedSlices[Any]] =>
              outputs.head.toOutputIndexedSlices()
            case o if o <:< typeOf[SparseOutput[Any]] => ???
            case _ => ???
          }
          value.asInstanceOf[O]
        }
      }
    }

    implicit def seqOutputEvidence[T]: OpOutput[Seq[Output[T]]] = {
      new OpOutput[Seq[Output[Any]]] {
        @inline override def fromOutputs(outputs: Seq[Output[Any]]): Seq[Output[T]] = {
          outputs.map(_.asInstanceOf[Output[T]])
        }

        @inline override def fromOutputLikes(outputs: Seq[OutputLike[Any]]): Seq[Output[T]] = {
          outputs.map(_.toOutput.asInstanceOf[Output[T]])
        }
      }
    }
  }

  //endregion Type Traits

  //region Graph Construction Helpers

  /** Returns the graph of the current op creation context. */
  private[api] def currentGraph: Graph = {
    graphConstructionScope.value.graph
  }

  /** Returns the name scope of the current op creation context. */
  private[api] def currentNameScope: String = {
    if (graphConstructionScope.value.nameScope == "")
      ""
    else
      s"${graphConstructionScope.value.nameScope}/"
  }

  /** Returns the device of the current op creation context. */
  private[api] def currentDevice: String = {
    graphConstructionScope.value.device
  }

  /** Returns the device function of the current op creation context. */
  private[api] def currentDeviceFunction: OpSpecification => String = {
    graphConstructionScope.value.deviceFunction
  }

  /** Returns the colocation ops of the current op creation context. */
  private[api] def currentColocationOps: Set[UntypedOp] = {
    graphConstructionScope.value.colocationOps
  }

  /** Returns the control dependencies of the current op creation context. */
  private[api] def currentControlDependencies: Set[UntypedOp] = {
    graphConstructionScope.value.controlDependencies
  }

  /** Returns the attributes of the current op creation context. */
  private[api] def currentAttributes: Map[String, Any] = {
    graphConstructionScope.value.attributes
  }

  /** Returns the container of the current op creation context. */
  private[api] def currentContainer: String = {
    graphConstructionScope.value.container
  }

  /** Returns the control flow context of the current op creation context. */
  private[api] def currentControlFlowContext: Option[Context] = {
    graphConstructionScope.value.controlFlowContext
  }

  /** Returns the local seeds an operation should use given an op-specific random seed.
    *
    * Given the op-specific seed, `opSeed`, this helper function returns two seeds derived from graph-level and op-level
    * seeds. Many random operations internally use the two seeds to allow the user to change the seed globally for a
    * graph, or only for specific operations.
    *
    * For details on how the graph-level seed interacts with op seeds, see [[setCurrentGraphRandomSeed]].
    *
    * @param  opSeed Op-specific seed value.
    * @return Tuple of two numbers that should be used for the local seed of this operation.
    */
  private[api] def currentGraphRandomSeed(
      opSeed: Option[Int] = None
  ): (Option[Int], Option[Int]) = {
    (currentGraph.randomSeed, opSeed) match {
      // Avoid (0, 0) as the C++ ops interpret it as non-determinism, which would be unexpected.
      case (Some(0), Some(0)) => (Some(0), Some(Int.MaxValue))
      case (Some(g), Some(o)) => (Some(g), Some(o))
      case (Some(g), None) => (Some(g), Some(currentGraph.ops.length))
      case (None, Some(o)) => (Some(DEFAULT_GRAPH_RANDOM_SEED), Some(o))
      case (None, None) => (None, None)
    }
  }

  /** Sets the graph-level random seed.
    *
    * Operations that rely on a random seed actually derive it from two seeds: the graph-level and the operation-level
    * seeds. This function sets the graph-level seed.
    *
    * Its interactions with operation-level seeds are as follows:
    *   1. If neither the graph-level nor the operation-level seed is set, a random seed is used for this op.
    *   2. If the graph-level seed is set, but the operation-level seed is not, the system deterministically picks an
    * operation-level seed in conjunction with the graph-level seed so that it gets a unique random sequence.
    *   3. If the graph-level seed is not set, but the operation-level seed is set, a default graph-level seed and the
    * specified operation-level seed are used to determine the random sequence.
    *   4. If both the graph-level and the operation-level seed are set, then both seeds are used in conjunction to
    * determine the random sequence.
    *
    * To generate different sequences across sessions, set neither the graph-level nor the op-level seeds.
    *
    * @param  value Value to set the graph-level random seed to.
    */
  private[api] def setCurrentGraphRandomSeed(value: Int): Unit = {
    currentGraph.setRandomSeed(value)
  }


  /** Creates a context that can be used for creating ops according to the provided options.
    *
    * = General Information =
    *
    * During graph creation, a context is maintained that includes:
    *   - The current graph in which new ops are placed.
    *   - The current name scope used for naming these new ops.
    *   - A device function, used to decide in which device (e.g., CPU) the new ops should be placed and executed.
    *   - A set of colocation ops for the newly constructed ops. This means that the newly created ops will be placed on
    * the same device as these colocation ops.
    *   - A set of ops defining control dependencies for the newly constructed ops. This means that the newly
    * constructed ops are constrained to only execute after the provided set of ops has finished executing.
    *   - A map from op attribute names to values for the newly constructed ops. These attributes will be applied to all
    * newly constructed ops.
    *   - A container name for the newly constructed resource ops. All newly constructed resource ops will be placed in
    * the provided container.
    *
    * Note that all arguments of this function are optional. If they are not provided, then the corresponding option in
    * current op creation context is left unchanged.
    *
    * Care must be taken if concurrency is used while creating the graph because the op creation context is wrapped
    * inside a [[scala.util.DynamicVariable]]. More information on this general issue can be found at
    * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
    *
    * = Argument Specifics =
    *
    * == Graph ==
    *
    * When `createWith(...)` is used with a graph, then all ops created within its code block will be placed in the
    * provided graph.
    *
    * For example:
    * {{{
    *   val g = Graph()
    *   createWith(graph = g) {
    *     val c = constant(5.0)
    *     assert(c.graph == g)
    *   }
    * }}}
    *
    * == Name Scope ==
    *
    * When `createWith(...)` is used with a name scope, the provided name scope is appended to the context name scope,
    * generating a new op creation context. This new context is used for all ops created within the code block provided
    * in the `createWith(...)` function. The `nameScope` argument will be interpreted as follows:
    *
    *   - A string not ending with `"/"` will create a new name scope, in which `nameScope` is appended to the prefix of
    * all operations created in the provided code block. If `nameScope` has been used before, it will be made unique
    * by calling `uniqueName(graph = context.graph, name = nameScope)`.
    *   - A string ending with `"/"` will be treated as an "absolute" name scope, which makes it possible to re-enter
    * existing scopes. Such absolute name scopes can be obtained by using the `currentNameScope` function, from
    * within the appropriate context.
    *   - A value of `""` will reset the current name scope to the top-level (i.e., empty) name scope.
    *
    * This function checks the provided `nameScope` for validity by checking whether it matches: (i) the regular
    * expression `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*` if the current context name scope is empty (i.e., at the root), or
    * (ii) the regular expression `[A-Za-z0-9_.\\-/]*`, otherwise.
    *
    * For example:
    * {{{
    *   // No name scope used
    *   val c = constant(1.0, name = "C")
    *   assert(c.op.name == "C")
    *   val c1 = constant(2.0, name = "C_1")
    *   assert(c_1.op.name == "C_1")
    *
    *   // Create a name scope called "Nested"
    *   createWith(nameScope = "Nested") {
    *     val nameScope = currentNameScope
    *     val nestedC = constant(3.0, name = "C")
    *     assert(nestedC.op.name == "Nested/C")
    *
    *     // Create a nested name scope called "Inner"
    *     createWith(nameScope = "Inner") {
    *       val nestedInnerC = constant(4.0, name = "C")
    *       assert(nestedInnerC.op.name == "Nested/Inner/C")
    *     }
    *
    *     // Create a nested name scope called "Inner_1"
    *     createWith(nameScope = "Inner_1") {
    *       val nestedInner1C = constant(5.0, name = "C")
    *       assert(nestedInner1C.op.name == "Nested/Inner_1/C")
    *
    *       createWith(nameScope = nameScope) {
    *         val nestedC1 = constant(6.0, name = "C_1")
    *         assert(nestedC1.op.name == "Nested/C_1")
    *
    *         // Reset the name scope using ""
    *         createWith(nameScope = "") {
    *           val c2 = constant(7.0, name = "C_2")
    *           assert(c2.op.name == "C_2")
    *         }
    *       }
    *     }
    *   }
    * }}}
    *
    * == Device ==
    *
    * When `createWith(...)` is used with a device, a `deviceFunction` argument can be additionally used (aside from the
    * device string representation provided through the `device` argument), that is a function taking an
    * [[OpSpecification]] as input and returning a string representation of the device where the corresponding op should
    * be placed. This function is invoked every time a new op is created within the provided code block. If the function
    * returns `null` for some op, then all subsequent invocations of `createWith(deviceFunction = ...)` in the provided
    * code block will be ignored. For information about the valid syntax of device name strings, see the documentation
    * in [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
    *
    * Note that the device scope may be overridden by op wrappers or other library code. For example, a variable
    * assignment op must be colocated with the corresponding variable. Incompatible device scopes will be ignored.
    *
    * For example:
    * {{{
    *   // Specifying which device to use
    *   createWith(device = "/GPU:0") {
    *     // All ops constructed in this code block will be placed in GPU 0
    *     val gpu0C = constant(7.0)
    *     assert(gpu0C.device == "/device:GPU:0")
    *
    *     // Reset the device being used
    *     createWith(device = null) {
    *       // All ops constructed in this code block will have no assigned device
    *       val c = constant(8.0)
    *       assert(c.device == "")
    *     }
    *   }
    *
    *   // Using a device function
    *   def matmulOnGPU(opSpecification: OpSpecification): String = {
    *     if (opSpecification.opType == "MatMul")
    *       "/GPU:0"
    *     else
    *       "/CPU:0"
    *   }
    *
    *   createWith(deviceFunction = matmulOnGPU) {
    *     // All ops of type "MatMul" constructed in this code block will be placed on GPU 0. All other operations will
    *     // be placed on CPU 0.
    *     val c = constant(9.0)
    *     assert(c.device == "/device:CPU:0")
    *     val m = matmul(c, constant(10.0))
    *     assert(m.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * == Colocation Ops ==
    *
    * When `createWith(...)` is used with a set of colocation ops, then all ops created within its code block will be
    * placed on the same device as the provided colocation ops. Note that if a set of colocation ops already exists in
    * the current op creation context (e.g., as the result of nesting multiple `createWith(colocationOps = ...)` calls),
    * then the new set of colocation ops will be the union of the two sets. If provided an empty colocation ops set,
    * then the new set of colocation ops will also be empty (i.e., it is being reset).
    *
    * Note that using a non-empty set of colocation ops resets any existing device constraints. In other words,
    * colocation ops override any other device placement specification.
    *
    * For example:
    * {{{
    *   val a = createWith(device = "/CPU:0")(constant(1.0))
    *   val b = createWith(device = "/GPU:0")(constant(1.0))
    *   assert(a.colocationOps === Set.empty[Op])
    *   assert(b.colocationOps === Set.empty[Op])
    *   val c = createWith(colocationOps = Set(a))(constant(1.0))
    *   assert(c.colocationOps === Set[Op](a))
    *   createWith(colocationOps = Set[Op](b)) {
    *     val d = constant(1.0)
    *     assert(d.colocationOps === Set[Op](b))
    *     createWith(colocationOps = Set[Op](a, d)) {
    *       val e = constant(1.0)
    *       assert(e.colocationOps === Set[Op](a, b, d))
    *       createWith(colocationOps = Set.empty[Op]) {
    *         val f = constant(1.0)
    *         assert(f.colocationOps === Set.empty[Op])
    *       }
    *     }
    *   }
    * }}}
    *
    * == Control Dependencies ==
    *
    * When `createWith(...)` is used with a set of control dependencies, then all ops created within its code block will
    * be dependent on the control dependency ops. This means that they will be guaranteed to execute only after all of
    * the control dependencies ops have finished executing. Note that if a set of control dependencies already exists in
    * the current op creation context (e.g., as the result of nesting multiple `createWith(controlDependencies = ...)`
    * calls), then the new set of control dependencies will be the union of the two sets. Furthermore, if an empty set
    * is provided, then the control dependencies are cleared, instead of taking the union with the current control
    * dependencies.
    *
    * For example:
    * {{{
    *   val a = constant(1.0)
    *   val b = constant(1.0)
    *   createWith(controlDependencies = Set(a)) {
    *     val c = constant(1.0)
    *     assert(c.controlInputs.toSet == Set(a))
    *     createWith(controlDependencies = Set(b, c)) {
    *       val d = constant(1.0)
    *       assert(d.controlInputs.toSet == Set(a, b, c))
    *       createWith(controlDependencies = Set()) {
    *         createWith(controlDependencies = Set(d)) {
    *           val e = constant(1.0)
    *           assert(e.controlInputs.toSet == Set(d))
    *         }
    *       }
    *     }
    *   }
    *   assert(a.controlOutputs.toSet == Set(c, d))
    *   assert(b.controlOutputs.toSet == Set(d))
    *   assert(c.controlOutputs.toSet == Set())
    *   assert(d.controlOutputs.toSet == Set(e))
    *   assert(e.controlOutputs.toSet == Set())
    * }}}
    *
    * Note that transitive dependencies are eliminated (e.g., if `a` depends on `b` and `c`, and `b` depends on `c`,
    * then the dependency of `a` on `c` is ignored) in order not to add redundant control dependencies to the graph.
    *
    * == Attributes ==
    *
    * When `createWith(...)` is used with a set of attributes, then all ops created within its code block will have
    * those attributes set to the provided values when constructed. Note that if a map from attribute names to values
    * already exists in the current op creation context, then the two maps are merged. If a name exists in both, then
    * the provided value overrides the existing one, otherwise, the union of the two maps is used. Note that if the
    * value for an attribute in the provided map is set to `null`, then that attribute name-value pair is completely
    * removed from the op creation context.
    *
    * For example:
    * {{{
    *   val a = constant(1.0)
    *   assert(a.stringAttribute("_a") == null)
    *   createWith(attributes = Map("_a" -> "foo")) {
    *     val b = constant(1.0)
    *     assert(b.stringAttribute("_a") == "foo")
    *     createWith(attributes = Map("_a" -> "bar")) {
    *       val c = constant(1.0)
    *       assert(c.stringAttribute("_a") == "bar")
    *       createWith(attributes = Map("_a" -> null)) {
    *         val d = constant(1.0)
    *         assert(d.stringAttribute("_a") == null)
    *       }
    *     }
    *   }
    * }}}
    *
    * == Container ==
    *
    * Stateful operations, such as variables and queues, can maintain their states on devices so that they can be shared
    * by multiple processes. A resource container is a string name under which these stateful operations are tracked.
    * These resources can be released or cleared with `Session.reset`. // TODO: [SESSION] Add that function reference.
    *
    * When `createWith(...)` is used with a container, then all resource ops created within its code block will be
    * placed in the provided container. A new value for the container always overrides the previous value, except if
    * `null`, meaning that the previous value is used. The default root container name is `""`.
    *
    * TODO: [VARIABLE] Add example when we have support for variables.
    *
    * == Combining Arguments ==
    *
    * Multiple arguments can be provided to change several aspects of the current op creation scope.
    *
    * For example:
    * {{{
    *   // Changing graph, name scope, and device to use for new ops.
    *   createWith(graph = g, nameScope = "Nested", device = "/GPU:0") {
    *     val c = constant(11.0, name = "C")
    *     assert(c.graph == g)
    *     assert(c.op.name == "Nested/C")
    *     assert(c.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * @param  graph               Graph to use as default for new ops.
    * @param  nameScope           Name scope to use.
    * @param  device              Device to use.
    * @param  deviceFunction      Device function to use.
    * @param  colocationOps       Colocation ops to use.
    * @param  controlDependencies Control dependencies to use.
    * @param  attributes          Attributes to use.
    * @param  container           Container to use for resources.
    * @param  block               Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def createWith[R](
      graph: Graph = null,
      nameScope: String = null,
      device: String = "",
      deviceFunction: OpSpecification => String = _.device,
      colocationOps: Set[UntypedOp] = null,
      controlDependencies: Set[UntypedOp] = null,
      attributes: Map[String, Any] = null,
      container: String = null
  )(block: => R): R = {
    // TODO: Move this to a separate scope class.
    // TODO: !!! The order of the updates matters here so let's make sure everything is fine.
    var updatedContext = graphConstructionScope.value
    val newGraph = mergeGraph(graph, updatedContext.graph)
    updatedContext = updatedContext.copy(graph = newGraph, outerContext = Some(updatedContext))
    val newNameScope = mergeNameScope(nameScope, updatedContext.nameScope, updatedContext.graph.uniqueName(_))
    updatedContext = updatedContext.copy(nameScope = newNameScope, outerContext = Some(updatedContext))
    val newDevice = mergeDevice(device, updatedContext.device)
    updatedContext = updatedContext.copy(device = newDevice, outerContext = Some(updatedContext))
    val newDeviceFunction = mergeDeviceFunction(deviceFunction, updatedContext.deviceFunction, updatedContext.device)
    updatedContext = updatedContext.copy(deviceFunction = newDeviceFunction, outerContext = Some(updatedContext))
    val newColocationOps = mergeColocationOps(colocationOps, updatedContext)
    updatedContext = updatedContext.copy(colocationOps = newColocationOps, outerContext = Some(updatedContext))
    val (newControlDependencies, newControlFlowContext) = mergeControlDependencies(controlDependencies, updatedContext)
    updatedContext = updatedContext.copy(
      controlDependencies = newControlDependencies, controlFlowContext = newControlFlowContext,
      outerContext = Some(updatedContext))
    val newAttributes = mergeAttributes(attributes, updatedContext)
    updatedContext = updatedContext.copy(attributes = newAttributes, outerContext = Some(updatedContext))
    val newContainer = mergeContainer(container, updatedContext.container)
    updatedContext = updatedContext.copy(container = newContainer, outerContext = Some(updatedContext))
    graphConstructionScope.withValue(updatedContext)(block)
  }

  /** Creates a context that can be used for creating ops.
    *
    * This function "pushes" the provided `nameScope` in the op creation context. More details on the op creation
    * context can be found in the documentation of the public API [[createWith]] function of this library.
    *
    * @param  nameScope Name scope to use.
    * @param  block     Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    * @throws GraphMismatchException If any two of the values provided lie in different graphs.
    */
  @throws[GraphMismatchException]
  private[api] def nameScope[R](nameScope: String)(block: => R): R = {
    val scope = graphConstructionScope
    val newNameScope = mergeNameScope(
      nameScope,
      scope.value.nameScope,
      scope.value.graph.uniqueName(_))
    scope.withValue(scope.value.copy(
      nameScope = newNameScope,
      outerContext = Some(scope.value))
    )(block)
  }

  /** Executes the provided block of code placing all created ops in the specified device. A `deviceFunction` argument
    * can be additionally used (aside from the device string representation provided through the `device` argument),
    * that is a function taking an [[OpSpecification]] as input and returning a string representation of the device
    * where the corresponding op should be placed. This function is invoked every time a new op is created within the
    * provided code block. If the function returns `null` for some op, then all subsequent invocations of
    * `device(deviceFunction = ...)` in the provided code block will be ignored. For information about the valid syntax
    * of device name strings, see the documentation in
    * [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
    *
    * Note that the device scope may be overridden by op wrappers or other library code. For example, a variable
    * assignment op must be colocated with the corresponding variable. Incompatible device scopes will be ignored.
    *
    * For example:
    * {{{
    *   // Specifying which device to use
    *   tf.device("/GPU:0") {
    *     // All ops constructed in this code block will be placed in GPU 0
    *     val gpu0C = constant(7.0)
    *     assert(gpu0C.device == "/device:GPU:0")
    *
    *     // Reset the device being used
    *     tf.device(null) {
    *       // All ops constructed in this code block will have no assigned device
    *       val c = constant(8.0)
    *       assert(c.device == "")
    *     }
    *   }
    *
    *   // Using a device function
    *   def matmulOnGPU(opSpecification: OpSpecification): String = {
    *     if (opSpecification.opType == "MatMul")
    *       "/GPU:0"
    *     else
    *       "/CPU:0"
    *   }
    *
    *   tf.device(deviceFunction = matmulOnGPU) {
    *     // All ops of type "MatMul" constructed in this code block will be placed on GPU 0. All other operations will
    *     // be placed on CPU 0.
    *     val c = constant(9.0)
    *     assert(c.device == "/device:CPU:0")
    *     val m = matmul(c, constant(10.0))
    *     assert(m.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * @param  device         Device to use.
    * @param  deviceFunction Device function to use.
    * @param  block          Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def device[R](
      device: String = "",
      deviceFunction: OpSpecification => String = _.device
  )(block: => R): R = {
    createWith(
      device = device,
      deviceFunction = deviceFunction
    )(block)
  }

  /** Creates a context that can be used for creating ops and placing them on the same device as `colocationOps`.
    *
    * Details on the op creation context can be found in the documentation of the public API [[createWith]] function of
    * this library.
    *
    * @param  colocationOps  Colocation ops to use.
    * @param  ignoreExisting Boolean value indicating whether to ignore the colocation ops in the current context.
    * @param  block          Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def colocateWith[R](
      colocationOps: Set[UntypedOp],
      ignoreExisting: Boolean = false
  )(block: => R): R = {
    val newColocationOps: Set[UntypedOp] = {
      if (ignoreExisting)
        colocationOps
      else
        mergeColocationOps(colocationOps, graphConstructionScope.value)
    }
    // By default, `colocateWith` resets the device function stack, since `colocateWith` is typically used in specific
    // internal library functions where colocation is intended to be "stronger" than device functions.
    graphConstructionScope.withValue(graphConstructionScope.value.copy(
      device = "",
      deviceFunction = (opSpec: OpSpecification) => opSpec.device,
      colocationOps = newColocationOps,
      outerContext = Some(graphConstructionScope.value))
    )(block)
  }

  /** Creates a context that can be used for creating gradient ops and placing them on the same device as
    * `colocationOps`.
    *
    * @param  colocationOps  Colocation ops to use.
    * @param  gradientUID    Unique identifier within the graph indicating which invocation of gradients is being
    *                        executed. Used to cluster ops for compilation.
    * @param  ignoreExisting Boolean value indicating whether to ignore the colocation ops in the current context.
    * @param  block          Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def colocateWithForGradient[R](
      colocationOps: Set[UntypedOp],
      gradientUID: Option[String],
      ignoreExisting: Boolean = false
  )(block: => R): R = {
    colocateWith(colocationOps, ignoreExisting) {
      gradientUID match {
        case Some(uid) => graphConstructionScope.value.controlFlowContext match {
          case Some(controlFlowContext) =>
            try {
              controlFlowContext.enterGradientColocation(colocationOps, uid)
              block
            } finally {
              controlFlowContext.exitGradientColocation(colocationOps, uid)
            }
          case None => block
        }
        case None => block
      }
    }
  }

  /** Creates a context that can be used for initialization ops.
    *
    * This context lifts ops out of control-flow scopes and function-building graphs. There is often a need to lift
    * variable initialization ops out of control-flow scopes, and function-building graphs. Entering an `initialization`
    * context is a mechanism for satisfying these desiderata. In particular, entering an `initialization` context has
    * two effects:
    *
    * (1) All control dependencies are cleared the moment the scope is entered; this is equivalent to entering the
    * context defined by `tf.createWith(controlDependencies = Set.empty)`, which has the side-effect of exiting
    * control-flow scopes like `tf.cond(...)` and `tf.whileLoop(...)`.
    *
    * (2) All operations that are created within this context are lifted into the lowest context in the "context stack"
    * that is not building a graph function. Every context switch is "logged" in a thread-local stack; the log entry for
    * a context switch is popped from the stack when the context is exited. Using an `initialization` context is
    * equivalent to crawling up the context stack, finding the first context that is not building a graph function, and
    * using it.
    *
    * @param  block Code block to run using the initialization op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    * @throws IllegalStateException If all graphs in the context stack are used for building functions.
    */
  @throws[IllegalStateException]
  private[api] def initializationScope[R](block: => R): R = {
    // Get the first context that's not building a function.
    var outerContext = graphConstructionScope.value
    while (outerContext.graph.isInstanceOf[FunctionGraph] && outerContext.outerContext.isDefined)
      outerContext = outerContext.outerContext.get
    if (outerContext.graph.isInstanceOf[FunctionGraph])
      throw new IllegalStateException("All graphs are building functions.")
    graphConstructionScope.withValue(outerContext) {
      // Entering an `initScope` preserves the name scope of the current context.
      createWith(
        nameScope = graphConstructionScope.value.nameScope,
        controlDependencies = Set.empty
      )(block)
    }
  }

  /** Merges a graph to the provided op creation context graph and returns the graph to use when specifying the updated
    * op creation context. The merging rules are specified in the documentation of the [[createWith]] function.
    *
    * @param  graph    Graph to merge.
    * @param  oldGraph Current op creation context graph.
    * @return Graph to use for the new op creation context.
    */
  private def mergeGraph(
      graph: Graph,
      oldGraph: Graph
  ): Graph = {
    if (graph == null)
      oldGraph
    else
      graph
  }

  /** Merges a name scope to the provided op creation context name scope and returns the name scope to use when
    * specifying the updated op creation context. The merging rules are specified in the documentation of the
    * [[createWith]] function.
    *
    * @param  nameScope    Name scope to merge.
    * @param  oldNameScope Old (i.e., current) name scope.
    * @param  uniqueNameFn Function that can be used to generate a unique name based on a provided name.
    * @return Name scope to use for the new op creation context.
    * @throws IllegalNameException If the provided name scope does not pass the regular expression validity checks.
    */
  @throws[IllegalNameException]
  private[api] def mergeNameScope(
      nameScope: String,
      oldNameScope: String,
      uniqueNameFn: String => String
  ): String = {
    if (nameScope == null) {
      oldNameScope
    } else {
      // Check whether the provided name scope is valid.
      // If the root name scope is being set, then stricter checks are performed on it (i.e., op naming checks). This
      // makes sure the name scope does not start with any illegal characters (e.g., '_', '-', '\', and '/').
      if ((oldNameScope == "" && nameScope != "" && !checkName(nameScope))
          || (oldNameScope != "" && !checkNameScope(nameScope)))
        throw IllegalNameException(s"Illegal name scope '$nameScope'.")
      if (nameScope == "")
        ""
      else if (nameScope.endsWith("/"))
        convertNameScopeToName(nameScope)
      else
        uniqueNameFn(nameScope)
    }
  }

  /** Merges a device to the provided op creation context device and returns the device to use when specifying the
    * updated op creation context. The merging rules are specified in the documentation of the [[createWith]] function.
    *
    * @param  device    Device to merge.
    * @param  oldDevice Old (i.e., current) device.
    * @return Device to use for the new op creation context.
    */
  private[api] def mergeDevice(device: String, oldDevice: String): String = {
    // Check if the device has been reset or has to be reset for all subsequent nested scopes
    if (oldDevice == null || device == null) {
      null
    } else {
      val oldDeviceSpec = DeviceSpecification.fromString(oldDevice)
      val newDeviceSpec = DeviceSpecification.fromString(device)
      DeviceSpecification.merge(oldDeviceSpec, newDeviceSpec).toString
    }
  }

  /** Merges a device function to the provided op creation context device and returns the device to use when specifying
    * the updated op creation context. The merging rules are specified in the documentation of the [[createWith]]
    * function.
    *
    * @param  deviceFunction    Device function to merge.
    * @param  oldDeviceFunction Old (i.e., current) device function.
    * @param  oldDevice         Old (i.e., current) device.
    * @return Device function to use for the new op creation context.
    */
  private[api] def mergeDeviceFunction(
      deviceFunction: OpSpecification => String,
      oldDeviceFunction: OpSpecification => String,
      oldDevice: String
  ): OpSpecification => String = {
    opSpecification => {
      val oldDeviceSpecString = oldDeviceFunction(opSpecification)
      val newDeviceSpecString = deviceFunction(opSpecification)
      // Check if the device has been reset or has to be reset for all subsequent nested scopes
      if (oldDevice == null || oldDeviceSpecString == null || newDeviceSpecString == null) {
        null
      } else {
        val oldDeviceSpec = DeviceSpecification.fromString(oldDeviceSpecString)
        val newDeviceSpec = DeviceSpecification.fromString(newDeviceSpecString)
        DeviceSpecification.merge(oldDeviceSpec, newDeviceSpec).toString
      }
    }
  }

  /** Merges a set of colocation ops to the provided op creation context set of colocation ops and returns the
    * set of colocation ops to use when specifying the updated op creation context. The merging rules are
    * specified in the documentation of the [[createWith]] function.
    *
    * @param  colocationOps Set of colocation ops to merge.
    * @param  context       Op creation context whose colocation ops need to be updated.
    * @return Set of colocation ops to use for the new op creation context.
    */
  private def mergeColocationOps(
      colocationOps: Set[UntypedOp],
      context: GraphConstructionScope
  ): Set[UntypedOp] = {
    if (colocationOps == null)
      context.colocationOps
    else if (colocationOps.isEmpty)
      Set.empty
    else
      context.colocationOps ++ colocationOps
  }

  /** Merges a set of control dependencies to the provided op creation context set of control dependencies and returns
    * the set of control dependencies to use when specifying the updated op creation context. The merging rules are
    * specified in the documentation of the [[createWith]] function.
    *
    * @param  controlDependencies Set of control dependencies to merge.
    * @param  context             Op creation context whose control dependencies needs to be updated.
    * @return Set of control dependencies to use for the new op creation context.
    */
  private def mergeControlDependencies(
      controlDependencies: Set[UntypedOp],
      context: GraphConstructionScope
  ): (Set[UntypedOp], Option[Context]) = {
    if (controlDependencies == null)
      (context.controlDependencies, context.controlFlowContext)
    else if (controlDependencies.isEmpty)
      (controlDependencies, None)
    else
      (context.controlDependencies ++ controlDependencies, context.controlFlowContext)
  }

  /** Merges a set of attributes to the provided op creation context set of attributes and returns the set of attributes
    * to use when specifying the updated op creation context. The merging rules are specified in the documentation of
    * the [[createWith]] function.
    *
    * @param  attributes Set of attributes to merge.
    * @param  context    Op creation context whose attributes needs to be updated.
    * @return Set of attributes to use for the new op creation context.
    */
  private def mergeAttributes(
      attributes: Map[String, Any],
      context: GraphConstructionScope
  ): Map[String, Any] = {
    if (attributes == null) {
      context.attributes
    } else if (attributes == Map.empty[String, Any]) {
      attributes.filter(attribute => attribute._2 != null)
    } else {
      var mergedMap = Map[String, Any](context.attributes.toSeq: _*)
      attributes.foreach(attribute => {
        if (attribute._2 == null && mergedMap.contains(attribute._1))
          mergedMap -= attribute._1
        else if (attribute._2 != null)
          mergedMap += attribute
      })
      mergedMap
    }
  }

  /** Merges a container to the provided op creation context container and returns the container to use when specifying
    * the updated op creation context. The merging rules are specified in the documentation of the [[createWith]]
    * function.
    *
    * @param  container    Container to merge.
    * @param  oldContainer Current op creation context container.
    * @return Container to use for the new op creation context.
    */
  private[this] def mergeContainer(container: String, oldContainer: String): String = {
    if (container == null) oldContainer else container
  }

  /** Checks whether the provided string is a valid op name.
    *
    * @param  name String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[api] def checkName(name: String): Boolean = {
    VALID_OP_NAME_REGEX.pattern.matcher(name).matches
  }

  /** Checks whether the provided string is a valid name scope for creating ops.
    *
    * @param  nameScope String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[api] def checkNameScope(nameScope: String): Boolean = {
    VALID_NAME_SCOPE_REGEX.pattern.matcher(nameScope).matches
  }

  /** Converts the provided name scope to a valid op name, by removing a trailing `"/"` if there exists one.
    *
    * @param  nameScope Name scope to convert.
    * @return Name obtained from the provided name scope.
    */
  private[api] def convertNameScopeToName(nameScope: String): String = {
    if (nameScope.endsWith("/"))
      nameScope.substring(0, nameScope.length - 1)
    else
      nameScope
  }

  /** Asserts that two ops are defined in the same graph. If they are not, a [[GraphMismatchException]] is thrown.
    *
    * @param  op1 First op.
    * @param  op2 Second op.
    * @throws GraphMismatchException If the two ops lie in different graphs.
    */
  @throws[GraphMismatchException]
  private[ops] def assertSameGraph(
      op1: UntypedOp,
      op2: UntypedOp
  ): Unit = {
    if (op1.graph != op2.graph)
      throw GraphMismatchException(s"'$op1' and '$op2' must be defined in the same graph.")
  }

  /** Returns the appropriate graph to use for the given inputs.
    *
    * This function provides a consistent algorithm for choosing the graph in which an op should be constructed in:
    *
    *   1. If the argument `graph` is provided and is not set to `null`, the function validates that all `inputs` are
    * defined in that graph.
    *   2. Otherwise, we attempt to select a graph from the first op in `inputs` and validate that all other `inputs`
    * are also defined in the same graph.
    *
    * @param  inputs Inputs.
    * @param  graph  Graph to use. If `null`, the graph is inferred from `inputs`.
    * @return The appropriate graph to use for the given inputs.
    * @throws GraphMismatchException If any two of the inputs lie in different graphs, or if `graph` is not `null` and
    *                                at least one of the `inputs` is not defined in it.
    */
  @throws[GraphMismatchException]
  private[ops] def getGraphFromInputs(
      inputs: Set[UntypedOp],
      graph: Graph = null
  ): Graph = {
    val returnGraph = if (graph == null) inputs.head.graph else graph
    inputs.foreach(i => {
      if (graph == null)
        assertSameGraph(inputs.head, i)
      else if (i.graph != returnGraph)
        throw GraphMismatchException(s"'$i' is not defined in the passed-in graph.")
    })
    returnGraph
  }

  //region ProtoBuf Helper Functions

  private[api] def stripNameScope(nameScope: String, name: String): String = {
    if (nameScope != null && nameScope != "")
      name.replaceFirst(s"([\\^]|loc:@|^)$nameScope[\\/]+(.*)", "$1$2")
    else
      name
  }

  private[api] def prependNameScope(nameScope: String, name: String): String = {
    if (nameScope != null && nameScope != "")
      name.replaceFirst("([\\^]|loc:@|^)(.*)", "$1" + nameScope + "/$2")
    else
      name
  }

  //endregion ProtoBuf Helper Functions

  private[ops] def controlDependencies(
      inputs: Set[Output[Any]]
  ): Set[UntypedOp] = {
    val controlDependencies = mutable.Set(Op.currentControlDependencies.toSeq: _*)
    controlDependencies ++= inputs.flatMap(_.op.controlInputs)
    inputs.foreach(input => pruneControlDependencies(controlDependencies, input.op))
    controlDependencies.toSet
  }

  /** Prunes control dependencies from the provided set, given that the op for which these control dependencies are
    * specified uses `op` as direct or indirect (through other ops) input or control input. This eliminates redundant
    * control dependencies due to transitive dependencies (e.g., if `a` depends on `b` and `c`, and `b` depends on
    * `c`, then the dependency of `a` on `c` is pruned).
    *
    * @param  controlDependencies Current set of control dependencies for the op that is being built.
    * @param  op                  Op that is a direct or indirect (through other ops) input or control input, for the op
    *                             that is being built.
    * @param  processedOps        Already processed ops (provided for efficiency purposes so that we do not go through
    *                             them a second time).
    */
  private[this] def pruneControlDependencies(
      controlDependencies: mutable.Set[UntypedOp],
      op: UntypedOp,
      processedOps: mutable.Set[UntypedOp] = mutable.Set.empty,
      maxDepth: Int = 10
  ): Unit = {
    if (maxDepth > 0 && !processedOps.contains(op)) {
      // Prune op that is already used as input to the dependant op
      controlDependencies -= op
      processedOps += op
      // Prune transitive control dependencies
      op._inputs.foreach(input => pruneControlDependencies(controlDependencies, input.op, processedOps, maxDepth - 1))
      op.controlInputs.foreach(pruneControlDependencies(controlDependencies, _, processedOps, maxDepth - 1))
    }
  }

  private[Op] def transitiveColocationOps(
      currentOps: Set[UntypedOp],
      collectedOps: Set[UntypedOp] = Set.empty
  ): Set[UntypedOp] = {
    val newOps = collectedOps ++ currentOps ++ currentOps.flatMap(_.colocationOps)
    if (newOps.size == collectedOps.size) {
      newOps
    } else {
      newOps ++ newOps.foldLeft(newOps)((collected, op) => {
        transitiveColocationOps(Set(op), collected)
      })
    }
  }

  final case class Builder[I: OpInput, O: OpOutput](
      opType: String,
      name: String,
      input: I
  ) {
    private[this] val inputs = implicitly[OpInput[I]].toBuilderInputs(input)

    private[this] val scope = graphConstructionScope.value

    scope.graph.assertNotFrozen()

    if (!checkName(name))
      throw IllegalNameException(s"Illegal op name '$name'.")

    private val graph: Graph = scope.graph

    private var built     : Boolean                             = false
    private var device    : Option[String]                      = None
    private var attributes: Map[String, Any]                    = Map.empty
    private var gradientFn: Option[Gradients.UntypedGradientFn] = None

    def build(): Op[I, O] = graph.synchronized {
      using(graph.reference) { r =>
        if (built)
          throw OpBuilderUsedException("This op builder has already been used to built an op and cannot be re-used.")
        device = Option(scope.deviceFunction(OpSpecification(this.name, opType, scope.device)))

        // Decide on the name of the new op.
        val name = {
          // If a name ends with a "/" then it is a name scope and we use it as-is, after removing the trailing "/".
          if (this.name.endsWith("/"))
            convertNameScopeToName(this.name)
          else
            graph.uniqueName(this.name)
        }
        val nativeHandle = NativeOp.allocate(r.nativeHandle, opType, name)

        // Add inputs and prune the control dependencies while doing that.
        val controlDependencies = mutable.Set(scope.controlDependencies.toSeq: _*)
        inputs.foreach {
          case Builder.InputTensor(inputTensor) =>
            pruneControlDependencies(controlDependencies, inputTensor.op)
            val processedInput = graph.processOpInput(inputTensor)
            NativeOp.addInput(nativeHandle, processedInput.op.nativeHandle, processedInput.index)
          case Builder.InputTensorList(inputTensorList) =>
            inputTensorList.foreach(input => pruneControlDependencies(controlDependencies, input.op))
            val processedInputList = inputTensorList.map(graph.processOpInput)
            NativeOp.addInputList(
              nativeHandle, processedInputList.map(_.op.nativeHandle).toArray, processedInputList.map(_.index).toArray)
        }

        // Add the pruned control dependencies.
        controlDependencies.foreach(op => NativeOp.addControlInput(nativeHandle, op.nativeHandle))

        // Add colocation constraints.
        val colocationOps = transitiveColocationOps(scope.colocationOps.filter(_ != null))
        val opDevice = device match {
          case None | Some("") => colocationOps.find(_.device != "").map(_.device).getOrElse("")
          case Some(d) => d
        }
        colocationOps.toSeq.sortBy(_.name).foreach(op => {
          if (opDevice != "" && op.device != "" && opDevice != op.device) {
            Op.logger.warn(
              s"Tried to colocate '$name' with an op '${op.name}' that has a different device: " +
                  s"$opDevice vs ${op.device}. Ignoring the colocation property.")
          } else {
            NativeOp.colocateWith(nativeHandle, op.nativeHandle)
            op.updateColocationOps(op.colocationOps.map(_.name) + name)
            NativeLibrary.setRequestedDevice(r.nativeHandle, op.nativeHandle, opDevice)
          }
        })

        // Add attributes.
        mergeAttributes(scope.attributes)
        setAttributes(nativeHandle)

        // TODO: !!! Set the "container" attribute when necessary. Need a way to check for statefulness.

        // Build the op and set its requested placement device.
        val op = Op[I, O](graph, Some(input), NativeOp.finish(nativeHandle))
        op.gradientFn = gradientFn
        NativeLibrary.setRequestedDevice(r.nativeHandle, op.nativeHandle, opDevice)
        op.controlFlowContext = scope.controlFlowContext
        op._inputs.map(_.op).foreach(ControlFlow.checkInputFromValidContext(op, _))
        op.controlFlowContext.foreach(_.add(op.asUntyped))
        built = true
        op
      }
    }

    private def mergeAttributes(attributes: Map[String, Any]): Unit = {
      attributes.foreach(attribute => {
        if (!this.attributes.contains(attribute._1))
          this.attributes += attribute
      })
    }

    private def setAttributes(nativeHandle: Long): Unit = {
      attributes.foreach(attribute => {
        attribute._2 match {
          case value: String =>
            NativeOp.setAttrString(nativeHandle, attribute._1, encodeString(value))
          case value: Array[String] =>
            NativeOp.setAttrStringList(nativeHandle, attribute._1, value.map(encodeString))
          case value: Long =>
            NativeOp.setAttrInt(nativeHandle, attribute._1, value)
          case value: Array[Long] =>
            NativeOp.setAttrIntList(nativeHandle, attribute._1, value)
          case value: Float =>
            NativeOp.setAttrFloat(nativeHandle, attribute._1, value)
          case value: Array[Float] =>
            NativeOp.setAttrFloatList(nativeHandle, attribute._1, value)
          case value: Boolean =>
            NativeOp.setAttrBool(nativeHandle, attribute._1, value)
          case value: Array[Boolean] =>
            NativeOp.setAttrBoolList(nativeHandle, attribute._1, value)
          case value: DataType[_] =>
            NativeOp.setAttrType(nativeHandle, attribute._1, value.cValue)
          case value: Array[DataType[_]] =>
            NativeOp.setAttrTypeList(nativeHandle, attribute._1, value.map(_.cValue))
          case value: Tensor[_] =>
            val handle = value.resolve()
            NativeOp.setAttrTensor(nativeHandle, attribute._1, handle)
            NativeTensor.delete(handle)
          case value: Array[Tensor[_]] =>
            val handles = value.map(_.resolve())
            NativeOp.setAttrTensorList(nativeHandle, attribute._1, handles)
            handles.foreach(NativeTensor.delete)
          case value: AttrValue =>
            NativeOp.setAttrProto(nativeHandle, attribute._1, value.toByteArray)
          case value: Shape =>
            if (value.rank != -1)
              NativeOp.setAttrShape(nativeHandle, attribute._1, value.asArray.map(_.toLong), value.rank)
            else
              NativeOp.setAttrShape(nativeHandle, attribute._1, Array.empty[Long], value.rank)
          case value: Array[Shape] =>
            NativeOp.setAttrShapeList(
              nativeHandle, attribute._1,
              value.map(s => if (s.rank != -1) s.asArray.map(_.toLong) else Array.empty[Long]),
              value.map(_.rank), value.length)
          case value: InstantiatedFunction[_, _] =>
            NativeOp.setAttrFuncName(nativeHandle, attribute._1, encodeString(value.hashedName))
          case _ =>
            throw new IllegalArgumentException(s"Unsupported attribute type for attribute named '${attribute._1}.'")
        }
      })
    }

    private def encodeString(value: String): Array[Byte] = value.getBytes(Charset.forName("UTF-8"))

    def setDevice(device: String): Builder[I, O] = {
      this.device = Some(device)
      this
    }

    def setAttribute(name: String, value: String): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[String]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Long): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Long]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Float): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Float]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Boolean): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Boolean]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: DataType[_]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[DataType[_]]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Tensor[_]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Tensor[_]]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Shape): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Shape]): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: AttrValue): Builder[I, O] = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: InstantiatedFunction[_, _]): Builder[I, O] = {
      value.addToGraph(graph)
      attributes += name -> value
      this
    }

    def setGradientFn[GI >: I : Op.OpInput, GO >: O : Op.OpOutput](
        gradientFn: Gradients.GradientFn[I, O, GI, GO]
    ): Builder[I, O] = {
      this.gradientFn = Some(Gradients.convertGradientFn(gradientFn))
      this
    }

    def setGradientFnHelper[GI >: I : Op.OpInput, GO >: O : Op.OpOutput](
        gradientFn: Option[Gradients.GradientFn[I, O, GI, GO]]
    ): Builder[I, O] = {
      this.gradientFn = gradientFn.map(Gradients.convertGradientFn)
      this
    }
  }

  object Builder {
    sealed trait Input
    case class InputTensor[T](input: Output[T]) extends Input
    case class InputTensorList[T](inputList: Seq[Output[T]]) extends Input
  }

  //endregion Graph Construction Helpers
}
