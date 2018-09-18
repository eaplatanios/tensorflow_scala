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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.io.NPY
import org.platanios.tensorflow.api.ops.{Basic, Output}
import org.platanios.tensorflow.api.tensors.ops.Basic.stack
import org.platanios.tensorflow.api.tensors.ops.Random
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import com.google.protobuf.ByteString
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.TensorProto

import java.nio._
import java.nio.charset.Charset
import java.nio.file.Path

import scala.compat.Platform.ConcurrentModificationException
import scala.language.{higherKinds, postfixOps}

/** Tensor (i.e., multi-dimensional array).
  *
  * Tensors are the main data structure underlying all operations in TensorFlow. They represent multi-dimensional arrays
  * of various data types (e.g., [[FLOAT32]]). Operations involving tensors can be of two types:
  *   - '''Eager:''' Operations directly executed on the tensor arguments, returning a new tensor. For example:
  *     {{{
  *       val a = Tensor(2.0, 4.5, 3.0, -1.2)
  *       val b = Tensor(Tensor(0.2, 0.4), Tensor(-2.3, 5.0))
  *       a.reshape(Shape(2, 2)) + b == Tensor(Tensor(2.2, 4.9), Tensor(0.7, 3.8))
  *     }}}
  *   - '''Symbolic:''' Operations that need to be constructed as part of a computational [[Graph]] before being
  *     executing using a [[Session]]. For example:
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
class Tensor[T] protected (
    private[api] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends TensorLike[T]
    with Closeable
    with ProtoSerializable {
  /** Lock for the native handle. */
  private[api] def NativeHandleLock = nativeHandleWrapper.Lock

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = nativeHandleWrapper.handle

  /** Data type of this tensor. */
  override val dataType: DataType[T] = DataType.fromCValue(NativeTensor.eagerDataType(nativeHandle))

  /** Shape of this tensor. */
  override val shape: Shape = Shape.fromSeq(NativeTensor.eagerShape(nativeHandle).map(_.toInt))

  /** Rank of this tensor (i.e., number of dimensions). */
  def rank: Int = shape.rank

  /** Number of elements contained in this tensor. */
  def size: Long = shape.numElements

  /** Device in which the tensor is stored. */
  override val device: String = NativeTensor.eagerDevice(nativeHandle)

  /** Returns a copy of this tensor on the CPU. */
  def cpu(): Tensor[T] = copyToDevice("CPU:0")

  /** Returns a copy of this tensor on the GPU.
    *
    * @param  gpuIndex Index of the GPU to use.
    */
  def gpu(gpuIndex: Int = 0): Tensor[T] = copyToDevice(s"GPU:$gpuIndex")

  /** Returns a copy of this tensor on the provided device.
    *
    * @param  device Device name. For example, `"CPU:0"`, or `"GPU:2"`.
    */
  def copyToDevice(device: String): Tensor[T] = {
    val parsedDevice = DeviceSpecification.fromString(device).toString.stripPrefix("/device:")
    val handle = NativeTensor.eagerCopyToDevice(nativeHandle, executionContext.value.nativeHandle, parsedDevice)
    Tensor.fromNativeHandle(handle)
  }

  private[api] def resolve(): Long = {
    NativeTensor.eagerResolve(nativeHandle)
  }

  private[api] def getElementAtFlattenedIndex(index: Int): T = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val offset = dataType.asInstanceOf[DataType] match {
      case STRING => INT64.byteSize.get * size.toInt + INT64.getElementFromBuffer(buffer, index * INT64.byteSize.get).toInt
      case _ => index * dataType.byteSize.get
    }
    val value = dataType.getElementFromBuffer(buffer, offset)
    NativeHandleLock synchronized {
      if (resolvedHandle != 0)
        NativeTensor.delete(resolvedHandle)
    }
    value
  }

  @throws[InvalidShapeException]
  def scalar: T = {
    if (size != 1)
      throw InvalidShapeException(
        "'Tensor.scalar' can only be called for scalar tensors (i.e., containing only one element).")
    getElementAtFlattenedIndex(0)
  }

  def entriesIterator: Iterator[T] = {
    object resolved extends (() => Unit) {
      private[Tensor] val lock   = Tensor.this.NativeHandleLock
      private[Tensor] var handle = resolve()

      override def apply(): Unit = {
        if (handle != 0) {
          lock synchronized {
            if (handle != 0) {
              NativeTensor.delete(handle)
              handle = 0
            }
          }
        }
      }
    }

    new Iterator[T] {
      private var i        : Int = 0
      private var remaining: Int = Tensor.this.size.ensuring(_ <= Int.MaxValue).toInt

      override def hasDefiniteSize: Boolean = true
      override def size: Int = remaining

      private val buffer: ByteBuffer = NativeTensor.buffer(resolved.handle).order(ByteOrder.nativeOrder)

      Disposer.add(this, resolved)

      override def hasNext: Boolean = remaining > 0

      override def next(): T = {
        if (!hasNext)
          throw new NoSuchElementException
        assert(resolved.handle != 0)

        val nextElement: T = dataType match {
          case _: STRING =>
            val offset = INT64.byteSize.get * (i + remaining)
            dataType.getElementFromBuffer(
              buffer,
              offset + INT64.getElementFromBuffer(buffer, i * INT64.byteSize.get).ensuring(_ <= Int.MaxValue).toInt)
          case _ => dataType.getElementFromBuffer(buffer, i * dataType.byteSize.get)
        }

        i += 1
        remaining -= 1
        if (0 == remaining) {
          resolved.lock synchronized {
            if (resolved.handle != 0) {
              NativeTensor.delete(resolved.handle)
              resolved.handle = 0
            } else {
              throw new ConcurrentModificationException
            }
          }
        }

        nextElement
      }
    }
  }

  def apply(firstIndexer: Indexer, otherIndexers: Indexer*): Tensor[T] = {
    this.slice(firstIndexer, otherIndexers: _*)
  }

  def slice(firstIndexer: Indexer, otherIndexers: Indexer*): Tensor[T] = {
    new BasicOps(this).slice(firstIndexer, otherIndexers: _*)
  }

  /** Returns a summary of the contents of this tensor.
    *
    * @param  maxEntries  Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *                     `maxEntries`, the output of that axis will be shortened to the first and last three elements.
    *                     Defaults to `6`. Values below `6` are ignored.
    * @param  flattened   If `true`, the summary is flattened to one line. Otherwise, the summary may span multiple
    *                     lines.
    * @param  includeInfo If `true`, the data type and the shape of the tensor are explicitly included in the summary.
    *                     Otherwise, they are not.
    * @return Tensor summary.
    */
  def summarize(maxEntries: Int = 6, flattened: Boolean = false, includeInfo: Boolean = true): String = {
    def summarize(tensor: Tensor[T], maxEntries: Int): String =
      tensor.rank match {
        case 0 => tensor.scalar.toString
        case 1 =>
          val slice =
            if (tensor.size <= math.max(maxEntries, 6))
              tensor.entriesIterator
            else
              (tensor(0 :: maxEntries / 2).entriesIterator.toSeq :+ "...") ++ tensor(-maxEntries / 2 ::).entriesIterator
          slice.mkString("[", ", ", "]")
        case _ =>
          val innerSummary = {
            def summarizeSlice(index: Int) = summarize(tensor(index).reshape(tensor.shape(1 ::)), maxEntries)

            if (tensor.shape(0) <= math.max(maxEntries, 6))
              for (i <- 0 until tensor.shape(0)) yield summarizeSlice(i)
            else {
              val start = for (i <- 0 until maxEntries / 2) yield summarizeSlice(i)
              val end = for (i <- tensor.shape(0) - maxEntries / 2 until tensor.shape(0)) yield summarizeSlice(i)
              (start :+ "...") ++ end
            }
          }
          val padding = " " * (this.rank - tensor.rank + 1)
          val extraLine = if (!flattened && tensor.rank >= 3) "\n" else ""
          innerSummary.mkString("[", (if (!flattened) ",\n" else ", ") + extraLine + padding, "]")
      }

    if (includeInfo)
      toString + (if (!flattened) "\n" else ": ") + summarize(this, maxEntries)
    else
      summarize(this, maxEntries)
  }

  override def toString: String = s"$dataType$shape"

  /** Returns this tensor. */
  override def toTensor: Tensor[T] = this

  /** Returns the tensor indexed slices that has the same value as this tensor. */
  override def toTensorIndexedSlices: TensorIndexedSlices[T] = {
    TensorIndexedSlices(
      indices = Tensor(0, 1 until shape(0): _*),
      values = this,
      denseShape = shape.toTensor(INT64))
  }

  override def toProto: TensorProto = toTensorProto

  /** Constructs and returns a [[TensorProto]] object that represents this tensor.
    *
    * @return Constructed [[TensorProto]].
    */
  def toTensorProto: TensorProto = Tensor.makeProto(this)

  /** Writes this tensor to the provided file, using the Numpy (i.e., `.npy`) file format. Note that this method will
    * replace the file, if it already exists. */
  def writeNPY(file: Path, fortranOrder: Boolean = false): Unit = NPY.write(this, file, fortranOrder)

  override def equals(that: Any): Boolean = that match {
    // TODO: !!! [TENSORS] Find a more efficient way to do this.
    case that: Tensor[_] =>
      this.shape == that.shape &&
          this.dataType == that.dataType &&
          this.entriesIterator.zip(that.entriesIterator).forall(p => p._1 == p._2)
    case _ => false
  }

  override def hashCode(): Int = {
    // TODO: !!! [TENSORS] Find a more efficient way to do this.
    val prime = 31
    var result = 1
    result = prime * result + dataType.hashCode
    result = prime * result + shape.hashCode
    entriesIterator.foreach(v => result = prime * result + v.hashCode)
    result
  }

  def toOutput: Output = Basic.constant(cpu())

  /** Closes this [[Tensor]] and releases any resources associated with it. Note that an [[Tensor]] is not
    * usable after it has been closed. */
  override def close(): Unit = closeFn()
}

object Tensor {
  private[tensors] val logger = Logger(LoggerFactory.getLogger("Tensor"))

  private[api] def fromNativeHandle[T](nativeHandle: Long): Tensor[T] = {
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandleWrapper.handle != 0) {
          NativeTensor.eagerDelete(nativeHandleWrapper.handle)
          nativeHandleWrapper.handle = 0
        }
      }
    }
    val tensor = new Tensor[T](nativeHandleWrapper, closeFn)
    // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(tensor, closeFn)
    tensor
  }

  private[api] def fromHostNativeHandle[T](nativeHandle: Long): Tensor[T] = {
    Tensor.fromNativeHandle(NativeTensor.eagerAllocate(nativeHandle))
  }

  def apply[TC, T](head: TC, tail: TC*)(implicit
      ev: TensorConvertible.Aux[TC, T]
  ): Tensor[T] = {
    stack((head +: tail).map(ev.toTensor), axis = 0)
  }

  def apply[T](dataType: DataType[T]): Tensor[T] = Tensor.allocate(dataType, Shape(0))

  def apply[TC, T, R](dataType: DataType[R], head: TC, tail: TC*)(implicit
      ev: TensorConvertible.Aux[TC, T]
  ): Tensor[R] = {
    stack((head +: tail).map(ev.toTensor), axis = 0).cast(dataType)
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
  def zeros[T](dataType: DataType[T], shape: Shape): Tensor[T] = {
    Tensor.fill(dataType, shape)(dataType.zero)(dataType.evSupportedType)
  }

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to zero. */
  def zerosLike[T](tensor: Tensor[T]): Tensor[T] = zeros(tensor.dataType, tensor.shape)

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
  def ones[T](dataType: DataType[T], shape: Shape): Tensor[T] = {
    Tensor.fill(dataType, shape)(dataType.one)(dataType.evSupportedType)
  }

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to one. */
  def onesLike[T](tensor: Tensor[T]): Tensor[T] = ones(tensor.dataType, tensor.shape)

  /** $OpDocRandomRandomUniform
    *
    * @group RandomOps
    * @param  dataType Data type for the output tensor.
    * @param  shape    Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  minValue Scalar tensor containing the inclusive lower bound on the random of random values to generate.
    *                  Defaults to `0`.
    * @param  maxValue Scalar tensor containing the exclusive upper bound on the random of random values to generate.
    *                  Defaults to `1`.
    * @param  seed     Optional random seed, used to generate a random seed pair for the random number generator, when
    *                  combined with the graph-level seed.
    * @return New random tensor.
    */
  def rand[T: IsInt32OrInt64OrFloat16OrFloat32OrFloat64, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Tensor[I]
  )(
      minValue: Tensor[T] = Tensor.zeros(dataType, Shape()),
      maxValue: Tensor[T] = Tensor.ones(dataType, Shape()),
      seed: Option[Int] = None
  ): Tensor[T] = {
    Random.randomUniform(dataType, shape)(minValue, maxValue, seed)
  }

  /** $OpDocRandomRandomNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor.
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @return New random tensor.
    */
  def randn[T: IsFloat16OrFloat32OrFloat64, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Tensor[I]
  )(
      mean: Tensor[T] = Tensor.zeros(dataType, Shape()),
      standardDeviation: Tensor[T] = Tensor.ones(dataType, Shape()),
      seed: Option[Int] = None
  ): Tensor[T] = {
    Random.randomNormal(dataType, shape)(mean, standardDeviation, seed)
  }

  // TODO: !!! [TYPES] Make this safer.

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
  def fill[T, V: SupportedType](dataType: DataType[T], shape: Shape)(value: V): Tensor[T] = {
    // TODO: [TENSORS] Do we want to keep this warning?
    // if (inferredDataType.priority < ev.dataType.priority)
    //   logger.warn(s"Downcasting value '$value' while creating tensor with '$dataType' data type.")
    shape.assertFullyDefined()
    val hostHandle = dataType match {
      case STRING =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = shape.numElements * (INT64.byteSize.get + numEncodedBytes)
        val hostHandle = NativeTensor.allocate(STRING.cValue, shape.asArray.map(_.toLong), numBytes)
        val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
        val baseOffset = INT64.byteSize.get * shape.numElements.toInt
        var index = 0
        var i = 0
        while (i < shape.numElements) {
          val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + index, STRING.cast(value))
          INT64.putElementInBuffer(buffer, i * INT64.byteSize.get, index.toLong)
          index += numEncodedBytes
          i += 1
        }
        hostHandle
      case _ =>
        val numBytes = shape.numElements * dataType.byteSize.get
        val hostHandle = NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
        val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
        var index = 0
        var i = 0
        while (i < shape.numElements) {
          dataType.putElementInBuffer(buffer, index, dataType.cast(value))
          index += dataType.byteSize.get
          i += 1
        }
        hostHandle
    }
    val tensor = Tensor.fromHostNativeHandle[T](hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
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
  private[api] def allocate[T](dataType: DataType[T], shape: Shape): Tensor[T] = {
    val hostHandle = dataType match {
      case STRING if shape.numElements == 0 => NativeTensor.allocate(dataType.cValue, Array[Long](0), 0)
      case STRING => throw new IllegalArgumentException(
        "Cannot pre-allocate string tensors because their size is not known.")
      case _ =>
        shape.assertFullyDefined()
        val numBytes = shape.numElements * dataType.byteSize.get
        NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
    }
    val tensor = Tensor.fromHostNativeHandle[T](hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
  }

  @throws[IllegalArgumentException]
  def fromBuffer[T](dataType: DataType[T], shape: Shape, numBytes: Long, buffer: ByteBuffer): Tensor[T] = {
    dataType.byteSize match {
      case Some(byteSize) if byteSize * shape.numElements != numBytes =>
        throw InvalidArgumentException(
          s"Trying to load a $dataType tensor with ${shape.numElements} elements, " +
              s"each of size ${dataType.byteSize} bytes, from the first $numBytes " +
              "stored in the provided byte buffer. Either change the data type or the " +
              "`numBytes` argument, to an appropriate value.")
      case _ => ()
    }
    this synchronized {
      // TODO: May behave weirdly for direct byte buffers allocated on the Scala side.
      val directBuffer = {
        if (buffer.isDirect) {
          buffer
        } else {
          val direct = ByteBuffer.allocateDirect(numBytes.toInt)
          val bufferCopy = buffer.duplicate()
          direct.put(bufferCopy.limit(numBytes.toInt).asInstanceOf[ByteBuffer])
          direct
        }
      }
      val hostHandle = NativeTensor.fromBuffer(dataType.cValue, shape.asArray.map(_.toLong), numBytes, directBuffer)
      val tensor = Tensor.fromHostNativeHandle[T](hostHandle)
      NativeTensor.delete(hostHandle)
      tensor
    }
  }

  /** Reads the tensor stored in the provided Numpy (i.e., `.npy`) file. */
  @throws[InvalidDataTypeException]
  @throws[IllegalArgumentException]
  def fromNPY[T](file: Path): Tensor[T] = NPY.read(file)

  @throws[InvalidArgumentException]
  def makeProto[T](value: Tensor[T]): TensorProto = {
    makeProto(value, value.dataType, value.shape)
  }

  @throws[InvalidArgumentException]
  def makeProto[T, D <: DataType[_]](value: Tensor[T], dataType: D): TensorProto = {
    makeProto(value, dataType, value.shape)
  }

  @throws[InvalidArgumentException]
  def makeProto[T](value: Tensor[T], shape: Shape): TensorProto = {
    makeProto(value, value.dataType, shape)
  }

  @throws[InvalidArgumentException]
  def makeProto[T, R](
      value: Tensor[T],
      dataType: DataType[R],
      shape: Shape
  ): TensorProto = {
    val castedValue = value.cast(dataType)
    val tensorProtoBuilder =
      TensorProto.newBuilder()
          .setDtype(dataType.protoType)
          .setTensorShape(shape.toTensorShapeProto)
    if (dataType.byteSize.get * value.size >= Int.MaxValue)
      throw InvalidArgumentException("Cannot serialize tensors whose content is larger than 2GB.")
    if (value.dataType != STRING && value.size == shape.numElements) {
      val resolvedHandle = castedValue.resolve()
      val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
      tensorProtoBuilder.setTensorContent(ByteString.copyFrom(buffer))
      castedValue.NativeHandleLock synchronized {
        if (resolvedHandle != 0)
          NativeTensor.delete(resolvedHandle)
      }
    } else {
      castedValue.entriesIterator.foreach(v => {
        dataType.addToTensorProtoBuilder(tensorProtoBuilder, v)
      })
    }
    tensorProtoBuilder.build()
  }
}
