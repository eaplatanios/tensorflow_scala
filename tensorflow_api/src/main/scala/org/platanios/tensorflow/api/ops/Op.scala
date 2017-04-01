package org.platanios.tensorflow.api.ops

import java.nio.charset.Charset

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.jni.{Operation => NativeOperation}

/**
  * @author Emmanouil Antonios Platanios
  */
// TODO: Add device and control inputs options.
private[tensorflow] final case class OpCreationContext(graph: Graph = Graph(), nameScope: String = "")

final case class Op(graph: Graph, unsafeNativeHandle: Long) {
  /** Returns the full name of the Operation. */
  def name: String = using(graph.reference) { _ => NativeOperation.name(unsafeNativeHandle) }

  /** Returns the type of the operation, i.e., the name of the computation performed by the
    * operation. */
  def opName: String = using(graph.reference) { _ => NativeOperation.opName(unsafeNativeHandle) }

  /** Returns the number of tensors produced by this operation. */
  def numOutputs: Int = using(graph.reference) { _ => NativeOperation.numOutputs(unsafeNativeHandle) }

  /** Returns a symbolic handle to one of the tensors produced by this operation. */
  def output(index: Int): Op.Output = Op.Output(op = this, index = index)

  // Package private, meant primarily for the public Output.dataType() method.
  private[api] def dataType(outputIndex: Int): DataType[_] =
    using(graph.reference) { r =>
      DataType.fromCValue(NativeOperation.dataType(r.nativeHandle, unsafeNativeHandle, outputIndex))
    }

  // Package private, meant primarily for the public Output.shape() method.
  private[api] def shape(outputIndex: Int): Array[Long] =
    using(graph.reference) { r => NativeOperation.shape(r.nativeHandle, unsafeNativeHandle, outputIndex) }
}

object Op {
  final case class Output(op: Op, index: Int) {
    def dataType: DataType[_] = op.dataType(index)
    def shape: Shape = Shape(op.shape(index))

    //region Ops

    def +(other: Output): Output = MathOps.add(x = this, y = other)
    def -(other: Output): Output = MathOps.subtract(x = this, y = other)
    def *(other: Output): Output = MathOps.multiply(x = this, y = other)
    def /(other: Output): Output = MathOps.divide(x = this, y = other)

    //endregion Ops
  }

  private[ops] def name(context: OpCreationContext, providedName: String, counter: Int = 0): String = {
    // TODO: Can we make this more efficient by keeping track of how many times a name has been used (maybe in the OpCreationContext)?
    if (context.graph.operation(providedName).isEmpty)
      providedName
    else if (context.graph.operation(s"${providedName}_$counter").isEmpty)
      s"${providedName}_$counter"
    else
      name(context = context, providedName = providedName, counter = counter + 1)
  }

  private[ops] def opBuildHelper(
      context: OpCreationContext, opType: String, name: String, inputs: Output*): Op.Builder = {
    val opName: String = Op.name(context = context, providedName = name)
    val opBuilder: Op.Builder = Op.Builder(graph = context.graph, opType = opType, name = opName)
    inputs.foreach(opBuilder.addInput)
    opBuilder
  }

  private[ops] final case class Builder(graph: Graph, opType: String, name: String) {
    private var nativeHandle: Long = using(graph.reference) { r =>
      NativeOperation.allocate(r.nativeHandle, opType, name)
    }

    def build(): Op = using(graph.reference) { _ =>
      val operation = Op(graph, NativeOperation.finish(nativeHandle))
      nativeHandle = 0
      operation
    }

    def addInput(input: Output): Builder = {
      using(graph.reference) { _ =>
        NativeOperation.addInput(nativeHandle, input.op.unsafeNativeHandle, input.index)
      }
      this
    }

    def addInputList(inputs: Array[Output]): Builder = {
      using(graph.reference) { _ =>
        NativeOperation.addInputList(nativeHandle, inputs.map(_.op.unsafeNativeHandle), inputs.map(_.index))
      }
      this
    }

    def setDevice(device: String): Builder = {
      using(graph.reference) { _ => NativeOperation.setDevice(nativeHandle, device) }
      this
    }

    def setAttribute(name: String, value: String): Builder = {
      setAttribute(name, value.getBytes(Charset.forName("UTF-8")))
      this
    }

    def setAttribute(name: String, value: Array[Byte]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrString(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Long): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrInt(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Array[Long]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrIntList(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Float): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrFloat(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Array[Float]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrFloatList(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Boolean): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrBool(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: Array[Boolean]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrBoolList(nativeHandle, name, value) }
      this
    }

    def setAttribute(name: String, value: DataType[_]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrType(nativeHandle, name, value.cValue) }
      this
    }

    def setAttribute(name: String, value: Array[DataType[_]]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrTypeList(nativeHandle, name, value.map(_.cValue)) }
      this
    }

    def setAttribute(name: String, value: Tensor[_]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrTensor(nativeHandle, name, value.nativeHandle) }
      this
    }

    def setAttribute(name: String, value: Array[Tensor[_]]): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrTensorList(nativeHandle, name, value.map(_.nativeHandle)) }
      this
    }

    def setAttribute(name: String, value: Shape): Builder = {
      using(graph.reference) { _ => NativeOperation.setAttrShape(nativeHandle, name, value.shape, value.rank) }
      this
    }
  }
}
