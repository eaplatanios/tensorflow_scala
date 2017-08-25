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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops.{Basic, Output}
import org.platanios.tensorflow.api.tensors.ops.Basic.stack
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import shapeless.{Generic, HList, Lazy}

import java.nio._
import java.nio.charset.Charset

import scala.collection.{TraversableLike, breakOut}
import scala.language.{higherKinds, postfixOps}
import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
case class Tensor private[tensors](
    private[tensors] var nativeHandle: Long, private[tensors] val _hostBuffer: Option[ByteBuffer]) extends Closeable {
  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Data type of this tensor. */
  val dataType: DataType = DataType.fromCValue(NativeTensor.eagerDataType(nativeHandle))

  /** Shape of this tensor. */
  val shape: Shape = {
    val s = NativeTensor.eagerShape(nativeHandle)
    if (s == null) Shape.unknown() else Shape.fromSeq(s.map(_.toInt))
  }

  /** Device in which the tensor is stored and where all computations for this tensor are performed. */
  val device: String = NativeTensor.eagerDevice(nativeHandle)

  def rank: Int = shape.rank
  def numElements: Int = shape.numElements

  /** Returns a copy of this tensor with its contents backed by host memory. */
  def cpu(): Tensor = copyToDevice("CPU:0")

  /** Returns a copy of this tensor with its contents backed by memory on the GPU.
    *
    * @param  gpuIndex Index of the GPU to use.
    * @return Tensor copy with its contents backed by memory on the GPU.
    */
  def gpu(gpuIndex: Int = 0): Tensor = copyToDevice(s"GPU:$gpuIndex")

  def copyToDevice(device: String)(implicit context: DynamicVariable[Context]): Tensor = {
    // TODO: !!! Kind of hacky.
    val parsedDevice = {
      val dev = DeviceSpecification.fromString(device).toString
      if (dev.startsWith("/device:"))
        dev.substring(8)
      else
        dev
    }
    val handle = NativeTensor.eagerCopyToDevice(nativeHandle, context.value.nativeHandle, parsedDevice)
    parsedDevice match {
      case "CPU:0" =>
        val hostHandle = NativeTensor.eagerResolve(handle)
        val tensor = Tensor.fromTFNativeHandle(hostHandle)
        NativeTensor.delete(hostHandle)
        tensor
      case _ => Tensor(handle)
    }
  }

  private[Tensor] def hostBuffer(): (ByteBuffer, Long) = _hostBuffer match {
    case Some(buffer) => (buffer, 0)
    case None =>
      val resolvedHandle = resolve()
      (NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder), resolvedHandle)
  }

  private[api] def resolve()(implicit context: DynamicVariable[Context]): Long = {
    if (device == "CPU:0") {
      NativeTensor.eagerResolve(nativeHandle)
    } else {
      val hostHandle = NativeTensor.eagerCopyToDevice(nativeHandle, context.value.nativeHandle, "CPU:0")
      val resolvedHandle = NativeTensor.eagerResolve(hostHandle)
      NativeTensor.eagerDelete(hostHandle)
      resolvedHandle
    }
  }

  private[api] def getElementAtFlattenedIndex(index: Int, _buffer: Option[ByteBuffer] = None): dataType.ScalaType = {
    _buffer match {
      case Some(buffer) =>
        dataType match {
          case STRING =>
            val offset = INT64.byteSize * numElements + INT64.getElementFromBuffer(buffer, index * INT64.byteSize).toInt
            dataType.getElementFromBuffer(buffer, offset)
          case _ => dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
        }
      case None =>
        val (buffer, resolvedHandle) = hostBuffer()
        val value = getElementAtFlattenedIndex(index, Some(buffer))
        if (resolvedHandle != 0)
          NativeTensor.delete(resolvedHandle)
        value
    }
  }

  @throws[InvalidShapeException]
  def scalar: dataType.ScalaType = {
    if (numElements != 1)
      throw InvalidShapeException(
        "'Tensor.scalar' can only be called for scalar tensors (i.e., containing only one element).")
    getElementAtFlattenedIndex(0)
  }

  def entriesIterator(_buffer: Option[ByteBuffer] = None): Iterator[dataType.ScalaType] = {
    _buffer match {
      case Some(_) => RowMajorOrder.indexIterator(shape.asArray).map(getElementAtFlattenedIndex(_, _buffer))
      case None =>
        val (buffer, resolvedHandle) = hostBuffer()
        val iterator = entriesIterator(Some(buffer))
        if (resolvedHandle != 0)
          NativeTensor.delete(resolvedHandle)
        iterator
    }
  }

  def apply(indexers: Indexer*): Tensor = this.slice(indexers: _*)

  /** Returns a summary of the contents of this tensor.
    *
    * @param  maxEntries Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *                    `maxEntries`, the output of that axis will be shortened to the first and last three elements.
    *                    Defaults to `6`. Values below `6` are ignored.
    * @return Tensor summary.
    */
  def summarize(maxEntries: Int = 6): String = {
    def summarize(tensor: Tensor, maxEntries: Int): String =
      tensor.rank match {
        case 0 => tensor.scalar.toString
        case 1 =>
          val slice =
            if (tensor.numElements <= math.max(maxEntries, 6))
              tensor.entriesIterator()
            else
              (tensor(0 :: 3).entriesIterator().toSeq :+ "...") ++ tensor(-3 ::).entriesIterator()
          slice.mkString("[", ", ", "]")
        case _ =>
          val innerSummary = {
            def summarizeSlice(index: Int) = summarize(tensor(index).reshape(tensor.shape(1 ::)), maxEntries)

            if (tensor.shape(0) <= math.max(maxEntries, 6))
              for (i <- 0 until tensor.shape(0)) yield summarizeSlice(i)
            else {
              val start = for (i <- 0 until 3) yield summarizeSlice(i)
              val end = for (i <- tensor.shape(0) - 3 until tensor.shape(0)) yield summarizeSlice(i)
              (start :+ "...") ++ end
            }
          }
          val padding = " " * (this.rank - tensor.rank + 1)
          val extraLine = if (tensor.rank >= 3) "\n" else ""
          innerSummary.mkString("[", ",\n" + extraLine + padding, "]")
      }
    toString + "\n" + summarize(this, maxEntries)
  }

  override def toString: String = s"$dataType$shape"

  override def equals(that: Any): Boolean = that match {
    // TODO: !!! [TENSORS] Replace with equality op and all op.
    case that: Tensor =>
      this.shape == that.shape &&
          this.dataType == that.dataType &&
          this.entriesIterator().zip(that.entriesIterator()).forall(p => p._1 == p._2)
    case _ => false
  }

  override def hashCode(): Int = {
    // TODO: !!! [TENSORS] Find more efficient way to do this.
    val prime = 31
    var result = 1
    result = prime * result + dataType.hashCode
    result = prime * result + shape.hashCode
    entriesIterator().foreach(v => result = prime * result + v.hashCode)
    result
  }

  def toOutput: Output = Basic.constant(this.cpu())

  /** Closes this [[Tensor]] and releases any resources associated with it. Note that an [[Tensor]] is not
    * usable after it has been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        NativeTensor.eagerDelete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

object Tensor {
  private[tensors] def apply(nativeHandle: Long): Tensor = {
    if (NativeTensor.eagerDevice(nativeHandle) == "CPU:0") {
      val hostHandle = NativeTensor.eagerResolve(nativeHandle)
      val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
      Tensor(nativeHandle, buffer)
    } else {
      Tensor(nativeHandle, None)
    }
  }

  private[tensors] def apply(nativeHandle: Long, buffer: ByteBuffer): Tensor = new Tensor(nativeHandle, Some(buffer))

  def apply[T](head: T, tail: T*)(implicit ev: TensorConvertible[T]): Tensor = {
    val tensors = head +: tail map ev.toTensor
    val dataTypes = tensors.map(_.dataType)
    Tensor(DataType.mostPrecise(dataTypes: _*), tensors.head, tensors.tail: _*)
  }

  def apply[T](dataType: DataType): Tensor = Tensor.allocate(dataType, Shape(0))

  def apply[T](dataType: DataType, head: T, tail: T*)(implicit ev: TensorConvertible[T]): Tensor = {
    val tensors = head +: tail map ev.toTensor
    val tensor = stack(tensors.map(_.cast(dataType)), 0)
    val hostHandle = tensor.resolve()
    val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
    Tensor(tensor.nativeHandle, buffer)
  }

  def fill[T](dataType: DataType, shape: Shape = null)(value: T)(implicit ev: SupportedType[T]): Tensor = {
    // TODO: Add downcasting warnings.
    val inferredShape = if (shape == null) Shape() else shape
    inferredShape.assertFullyDefined()
    dataType match {
      case STRING =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = inferredShape.numElements * (INT64.byteSize + numEncodedBytes)
        val tensor = Tensor.allocate(STRING, inferredShape, numBytes)
        val buffer = tensor.hostBuffer()._1
        val baseOffset = INT64.byteSize * tensor.numElements
        var index = 0
        var i = 0
        while (i < tensor.numElements) {
          STRING.putElementInBuffer(buffer, baseOffset + index, STRING.cast(value))
          INT64.putElementInBuffer(buffer, i * INT64.byteSize, index.toLong)
          index += numEncodedBytes
          i += 1
        }
        tensor
      case _ =>
        val tensor = allocate(dataType = dataType, shape = inferredShape)
        val buffer = tensor.hostBuffer()._1
        var index = 0
        var i = 0
        while (i < tensor.numElements) {
          dataType.putElementInBuffer(buffer, index, dataType.cast(value))
          index += dataType.byteSize
          i += 1
        }
        tensor
    }
  }

  private[api] def fromTFNativeHandle(nativeHandle: Long): Tensor = {
    Tensor(
      NativeTensor.eagerAllocate(nativeHandle),
      NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder))
  }

  private[api] def allocate(dataType: DataType, shape: Shape): Tensor = dataType match {
    case STRING => throw new IllegalArgumentException(
      "Cannot pre-allocate string tensors because their size is not known.")
    case _ =>
      shape.assertFullyDefined()
      val numBytes = shape.numElements * dataType.byteSize
      val hostHandle = NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
      val tensor = Tensor.fromTFNativeHandle(hostHandle)
      NativeTensor.delete(hostHandle)
      tensor
  }

  private[api] def allocate(dataType: DataType, shape: Shape, numBytes: Long): Tensor = {
    shape.assertFullyDefined()
    val hostHandle = NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
    val tensor = Tensor.fromTFNativeHandle(hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
  }

  @throws[IllegalArgumentException]
  def fromBuffer(dataType: DataType, shape: Shape, numBytes: Long, buffer: ByteBuffer): Tensor = {
    if (!buffer.isDirect)
      throw new IllegalArgumentException("Can only create tensors from direct byte buffers.")
    val hostHandle = NativeTensor.fromBuffer(dataType.cValue, shape.asArray.map(_.toLong), numBytes, buffer)
    val tensor = Tensor.fromTFNativeHandle(hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
  }

  private[tensors] trait Implicits {
    implicit def tensorConvertibleToTensor[T](value: T)(implicit ev: TensorConvertible[T]): Tensor = ev.toTensor(value)
  }
}

trait TensorConvertible[T] {
  def toTensor(value: T): Tensor
}

object TensorConvertible {
  implicit val tensorTensorConvertible: TensorConvertible[Tensor] = new TensorConvertible[Tensor] {
    override def toTensor(value: Tensor): Tensor = value
  }

  implicit val shapeTensorConvertible: TensorConvertible[Shape] = new TensorConvertible[Shape] {
    override def toTensor(value: Shape): Tensor = value.toTensor()
  }

  implicit val rangeTensorConvertible: TensorConvertible[Range] = new TensorConvertible[Range] {
    override def toTensor(value: Range): Tensor = stack(value.map(Tensor.fill(INT32)(_)))
  }

  implicit def supportedTypeTensorConvertible[T](implicit ev: SupportedType[T]): TensorConvertible[T] = {
    new TensorConvertible[T] {
      override def toTensor(value: T): Tensor = Tensor.fill(ev.dataType, Shape())(value)
    }
  }

  implicit def arrayExecutable[T](implicit ev: TensorConvertible[T]): TensorConvertible[Array[T]] = {
    new TensorConvertible[Array[T]] {
      override def toTensor(value: Array[T]): Tensor = stack(value.map(ev.toTensor))
    }
  }

  implicit def traversableExecutable[T, CC[A] <: TraversableLike[A, CC[A]]](
      implicit ev: TensorConvertible[T]): TensorConvertible[CC[T]] = {
    new TensorConvertible[CC[T]] {
      override def toTensor(value: CC[T]): Tensor = stack(value.map(ev.toTensor)(breakOut))
    }
  }

  implicit def productExecutable[T <: Product, L <: HList](implicit
      gen: Generic.Aux[T, L],
      ev: Lazy[TensorConvertible[L]]
  ): TensorConvertible[T] = new TensorConvertible[T] {
    override def toTensor(value: T): Tensor = {
      ev.value.toTensor(gen.to(value))
    }
  }
}
