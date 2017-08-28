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
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops.{Basic, Output}
import org.platanios.tensorflow.api.tensors.ops.Basic.stack
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import shapeless.{Generic, HList, Lazy}
import java.nio._
import java.nio.charset.Charset

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.{TraversableLike, breakOut}
import scala.language.{higherKinds, postfixOps}
import scala.util.DynamicVariable

/** Tensor (i.e., multi-dimensional array).
  *
  * Tensors are the main data structure underlying all operations in TensorFlow. They represent multi-dimensional arrays
  * of various data types (e.g., [[FLOAT32]]). Operations involving tensors can be of two types:
  *   - **Eager:** Operations directly executed on the tensor arguments, returning a new tensor. For example:
  *     {{{
  *       val a = Tensor(2.0, 4.5, 3.0, -1.2)
  *       val b = Tensor(Tensor(0.2, 0.4), Tensor(-2.3, 5.0))
  *       a.reshape(Shape(2, 2)) + b == Tensor(Tensor(2.2, 4.9), Tensor(0.7, 3.8))
  *     }}}
  *   - **Symbolic:** Operations that need to be constructed as part of a computational [[Graph]] before being executing
  *     using a [[Session]]. For example:
  *     {{{
  *       val a = tf.placeholder(FLOAT64, Shape(4))               // Symbolic placeholder for value of a
  *       val b = tf.placeholder(FLOAT64, Shape(2, 2))            // Symbolic placeholder for the value of b
  *       val add = tf.reshape(a, Shape(2, 2)) + b                // Symbolic representation of the computation
  *       val result = Session.run(
  *         feeds = Map(
  *           a -> Tensor(2.0, 4.5, 3.0, -1.2),
  *           b -> Tensor(Tensor(0.2, 0.4), Tensor(-2.3, 5.0))),
  *         fetches = add)                                        // Performs the actual computation
  *       result == Tensor(Tensor(2.2, 4.9), Tensor(0.7, 3.8))
  *     }}}
  *     // TODO: [OPS] Update doc when we enrich op outputs similarly to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
class Tensor private[Tensor](private[tensors] var nativeHandle: Long) extends Closeable {
  /** Lock for the native handle. */
  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Data type of this tensor. */
  val dataType: DataType = DataType.fromCValue(NativeTensor.eagerDataType(nativeHandle))

  /** Shape of this tensor. */
  val shape: Shape = Shape.fromSeq(NativeTensor.eagerShape(nativeHandle).map(_.toInt))

  /** Rank of this tensor (i.e., number of dimensions). */
  def rank: Int = shape.rank

  /** Number of elements contained in this tensor. */
  def numElements: Int = shape.numElements

  /** Device in which the tensor is stored. */
  val device: String = NativeTensor.eagerDevice(nativeHandle)

  /** Returns a copy of this tensor on the CPU. */
  def cpu(): Tensor = copyToDevice("CPU:0")

  /** Returns a copy of this tensor on the GPU.
    *
    * @param  gpuIndex Index of the GPU to use.
    */
  def gpu(gpuIndex: Int = 0): Tensor = copyToDevice(s"GPU:$gpuIndex")

  /** Returns a copy of this tensor on the provided device.
    *
    * @param  device Device name. For example, `"CPU:0"`, or `"GPU:2"`.
    */
  def copyToDevice(device: String)(implicit context: DynamicVariable[Context]): Tensor = {
    val parsedDevice = DeviceSpecification.fromString(device).toString.stripPrefix("/device:")
    val handle = NativeTensor.eagerCopyToDevice(nativeHandle, context.value.nativeHandle, parsedDevice)
    parsedDevice match {
      case "CPU:0" =>
        val hostHandle = NativeTensor.eagerResolve(handle)
        val tensor = Tensor.fromHostNativeHandle(hostHandle)
        NativeTensor.delete(hostHandle)
        tensor
      case _ => Tensor(handle)
    }
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

  private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val value = dataType match {
      case STRING =>
        val offset = INT64.byteSize * numElements + INT64.getElementFromBuffer(buffer, index * INT64.byteSize).toInt
        dataType.getElementFromBuffer(buffer, offset)
      case _ => dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
    }
    if (resolvedHandle != 0)
      NativeTensor.delete(resolvedHandle)
    value
  }

  @throws[InvalidShapeException]
  def scalar: dataType.ScalaType = {
    if (numElements != 1)
      throw InvalidShapeException(
        "'Tensor.scalar' can only be called for scalar tensors (i.e., containing only one element).")
    getElementAtFlattenedIndex(0)
  }

  def entriesIterator: Iterator[dataType.ScalaType] = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val iterator = dataType match {
      case STRING =>
        val offset = INT64.byteSize * numElements
        Iterator.range(0, numElements).map(i => {
          dataType.getElementFromBuffer(buffer, offset + INT64.getElementFromBuffer(buffer, i * INT64.byteSize).toInt)
        })
      case _ => Iterator.range(0, numElements).map(i => dataType.getElementFromBuffer(buffer, i * dataType.byteSize))
    }
    if (resolvedHandle != 0)
      NativeTensor.delete(resolvedHandle)
    iterator
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
              tensor.entriesIterator
            else
              (tensor(0 :: 3).entriesIterator.toSeq :+ "...") ++ tensor(-3 ::).entriesIterator
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
          this.entriesIterator.zip(that.entriesIterator).forall(p => p._1 == p._2)
    case _ => false
  }

  override def hashCode(): Int = {
    // TODO: !!! [TENSORS] Find more efficient way to do this.
    val prime = 31
    var result = 1
    result = prime * result + dataType.hashCode
    result = prime * result + shape.hashCode
    entriesIterator.foreach(v => result = prime * result + v.hashCode)
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
  private[this] val logger = Logger(LoggerFactory.getLogger("Tensor"))

  def fromNativeHandle(nativeHandle: Long): Tensor = new Tensor(nativeHandle)

  private[api] def fromHostNativeHandle(nativeHandle: Long): Tensor = {
    Tensor.fromNativeHandle(NativeTensor.eagerAllocate(nativeHandle))
  }

  def apply[T](head: T, tail: T*)(implicit ev: TensorConvertible[T]): Tensor = {
    val tensors = head +: tail map ev.toTensor
    val dataTypes = tensors.map(_.dataType)
    Tensor(DataType.mostPrecise(dataTypes: _*), tensors.head, tensors.tail: _*)
  }

  def apply[T](dataType: DataType): Tensor = Tensor.allocate(dataType, Shape(0))

  def apply[T](dataType: DataType, head: T, tail: T*)(implicit ev: TensorConvertible[T]): Tensor = {
    stack((head +: tail).map(ev.toTensor), 0).cast(dataType)
  }

  /** Returns a new tensor of type `dataType` with shape `shape` and all elements set to zero.
    *
    * For example:
    * {{{
    *   Tensor.zeros(INT32, Shape(3, 4)) == Tensor(Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0))
    * }}}
    *
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @return Constructed tensor.
    */
  def zeros(dataType: DataType, shape: Shape): Tensor = {
    dataType match {
      case BOOLEAN => Tensor.fill(BOOLEAN, shape)(false)
      case _ => Tensor.fill(dataType, shape)(0)
    }
  }

  /** Returns a new tensor of type `dataType` with shape `shape` and all elements set to ones.
    *
    * For example:
    * {{{
    *   Tensor.ones(INT32, Shape(3, 4)) == Tensor(Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1))
    * }}}
    *
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @return Constructed tensor.
    */
  def ones(dataType: DataType, shape: Shape): Tensor = {
    dataType match {
      case BOOLEAN => Tensor.fill(BOOLEAN, shape)(true)
      case _ => Tensor.fill(dataType, shape)(1)
    }
  }

  /** Returns a new tensor of type `dataType` with shape `shape` and all elements set to `value`.
    *
    * If `dataType` is not provided, then its value is inferred from `value`.
    *
    * For example:
    * {{{
    *   Tensor.fill(INT32, Shape(3, 4))(4) == Tensor(Tensor(4, 4, 4, 4), Tensor(4, 4, 4, 4), Tensor(4, 4, 4, 4))
    * }}}
    *
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @return Constructed tensor.
    */
  def fill[T](dataType: DataType = null, shape: Shape = Shape())(value: T)(implicit ev: SupportedType[T]): Tensor = {
    val inferredDataType = if (dataType == null) ev.dataType else dataType
    if (inferredDataType.priority < ev.dataType.priority)
      logger.warn(s"Downcasting value '$value' while creating tensor with '$dataType' data type.")
    shape.assertFullyDefined()
    inferredDataType match {
      case STRING =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = shape.numElements * (INT64.byteSize + numEncodedBytes)
        val hostHandle = NativeTensor.allocate(STRING.cValue, shape.asArray.map(_.toLong), numBytes)
        val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
        val baseOffset = INT64.byteSize * shape.numElements
        var index = 0
        var i = 0
        while (i < shape.numElements) {
          val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + index, STRING.cast(value))
          INT64.putElementInBuffer(buffer, i * INT64.byteSize, index.toLong)
          index += numEncodedBytes
          i += 1
        }
        val tensor = Tensor.fromNativeHandle(NativeTensor.eagerAllocate(hostHandle))
        NativeTensor.delete(hostHandle)
        tensor
      case _ =>
        val numBytes = shape.numElements * inferredDataType.byteSize
        val hostHandle = NativeTensor.allocate(inferredDataType.cValue, shape.asArray.map(_.toLong), numBytes)
        val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
        var index = 0
        var i = 0
        while (i < shape.numElements) {
          inferredDataType.putElementInBuffer(buffer, index, inferredDataType.cast(value))
          index += inferredDataType.byteSize
          i += 1
        }
        val tensor = Tensor.fromNativeHandle(NativeTensor.eagerAllocate(hostHandle))
        NativeTensor.delete(hostHandle)
        tensor
    }
  }

  /** Allocates a new tensor without worrying about the values stored in it.
    *
    * @param  dataType Tensor data type, which cannot be [[STRING]].
    * @param  shape    Tensor shape.
    * @return Allocated tensor.
    * @throws IllegalArgumentException If `dataType` is [[STRING]], because the number of bytes required for a string
    *                                  tensor are not known until all its element values are known.
    */
  @throws[IllegalArgumentException]
  private[api] def allocate(dataType: DataType, shape: Shape): Tensor = dataType match {
    case STRING => throw new IllegalArgumentException(
      "Cannot pre-allocate string tensors because their size is not known.")
    case _ =>
      shape.assertFullyDefined()
      val numBytes = shape.numElements * dataType.byteSize
      val hostHandle = NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
      val tensor = Tensor.fromHostNativeHandle(hostHandle)
      NativeTensor.delete(hostHandle)
      tensor
  }

  @throws[IllegalArgumentException]
  def fromBuffer(dataType: DataType, shape: Shape, numBytes: Long, buffer: ByteBuffer): Tensor = {
    // TODO: May behave weirdly for direct byte buffers allocated on the Scala side.
    val directBuffer = {
      if (buffer.isDirect) {
        buffer
      } else {
        val direct = ByteBuffer.allocateDirect(buffer.capacity())
        direct.put(buffer)
        direct
      }
    }
    val hostHandle = NativeTensor.fromBuffer(dataType.cValue, shape.asArray.map(_.toLong), numBytes, directBuffer)
    val tensor = Tensor.fromHostNativeHandle(hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
  }

  private[tensors] trait Implicits {
    implicit def tensorConvertibleToTensor[T](value: T)(implicit ev: TensorConvertible[T]): Tensor = ev.toTensor(value)
  }
}

trait TensorConvertible[T] {
  // TODO: Add data type argument.
  @inline def toTensor(value: T): Tensor
}

object TensorConvertible {
  implicit val tensorTensorConvertible: TensorConvertible[Tensor] = new TensorConvertible[Tensor] {
    @inline override def toTensor(value: Tensor): Tensor = value
  }

  implicit val shapeTensorConvertible: TensorConvertible[Shape] = new TensorConvertible[Shape] {
    @inline override def toTensor(value: Shape): Tensor = value.toTensor()
  }

  implicit val rangeTensorConvertible: TensorConvertible[Range] = new TensorConvertible[Range] {
    @inline override def toTensor(value: Range): Tensor = stack(value.map(Tensor.fill(INT32)(_)))
  }

  implicit def supportedTypeTensorConvertible[T](implicit ev: SupportedType[T]): TensorConvertible[T] = {
    new TensorConvertible[T] {
      @inline override def toTensor(value: T): Tensor = Tensor.fill(ev.dataType, Shape())(value)
    }
  }

  implicit def arrayExecutable[T](implicit ev: TensorConvertible[T]): TensorConvertible[Array[T]] = {
    new TensorConvertible[Array[T]] {
      @inline override def toTensor(value: Array[T]): Tensor = stack(value.map(ev.toTensor))
    }
  }

  implicit def traversableExecutable[T, CC[A] <: TraversableLike[A, CC[A]]](
      implicit ev: TensorConvertible[T]): TensorConvertible[CC[T]] = {
    new TensorConvertible[CC[T]] {
      @inline override def toTensor(value: CC[T]): Tensor = stack(value.map(ev.toTensor)(breakOut))
    }
  }

  implicit def productExecutable[T <: Product, L <: HList](implicit
      gen: Generic.Aux[T, L],
      ev: Lazy[TensorConvertible[L]]
  ): TensorConvertible[T] = new TensorConvertible[T] {
    @inline override def toTensor(value: T): Tensor = {
      ev.value.toTensor(gen.to(value))
    }
  }
}
