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
import org.platanios.tensorflow.api.ops.{Op, Output, Basic => OutputBasic}
import org.platanios.tensorflow.api.tensors.ops.{Basic, Math, Random}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}
import org.platanios.tensorflow.jni.generated.tensors.{Random => NativeTensorOpsRandom}
import org.platanios.tensorflow.jni.generated.tensors.{Sparse => NativeTensorOpsSparse}

import com.google.protobuf.ByteString
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.TensorProto

import java.nio._
import java.nio.charset.Charset
import java.nio.file.Path

import scala.compat.Platform.ConcurrentModificationException
import scala.language.{higherKinds, postfixOps}

/** Represents tensor-like objects.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorLike[+T] {
  /** Data type of this tensor. */
  val dataType: DataType[T]

  /** Shape of this tensor. */
  val shape: Shape

  /** Device on which this tensor is stored. */
  val device: String

  /** Returns the tensor that this tensor-like object represents. */
  def toTensor: Tensor[T]

  /** Returns the tensor indexed slices that has the same value as this tensor-like object. */
  def toTensorIndexedSlices: TensorIndexedSlices[T]
}

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
class Tensor[+T] protected (
    private[api] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends TensorLike[T]
    with Closeable
    with ProtoSerializable {
  /** Lock for the native handle. */
  private[api] def NativeHandleLock = {
    nativeHandleWrapper.Lock
  }

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = {
    nativeHandleWrapper.handle
  }

  /** Data type of this tensor. */
  override val dataType: DataType[T] = {
    DataType.fromCValue(NativeTensor.eagerDataType(nativeHandle))
  }

  /** Shape of this tensor. */
  override val shape: Shape = {
    Shape.fromSeq(NativeTensor.eagerShape(nativeHandle).map(_.toInt))
  }

  /** Rank of this tensor (i.e., number of dimensions). */
  def rank: Int = {
    shape.rank
  }

  /** Number of elements contained in this tensor. */
  def size: Long = {
    shape.numElements
  }

  /** Device in which the tensor is stored. */
  override val device: String = {
    NativeTensor.eagerDevice(nativeHandle)
  }

  /** Returns a copy of this tensor on the CPU. */
  def cpu(): Tensor[T] = {
    copyToDevice("CPU:0")
  }

  /** Returns a copy of this tensor on the GPU.
    *
    * @param  gpuIndex Index of the GPU to use.
    */
  def gpu(gpuIndex: Int = 0): Tensor[T] = {
    copyToDevice(s"GPU:$gpuIndex")
  }

  /** Returns a copy of this tensor on the provided device.
    *
    * @param  device Device name. For example, `"CPU:0"`, or `"GPU:2"`.
    */
  def copyToDevice(device: String): Tensor[T] = {
    val parsedDevice = DeviceSpecification.fromString(device).toString.stripPrefix("/device:")
    val handle = NativeTensor.eagerCopyToDevice(
      nativeHandle, executionContext.value.nativeHandle, parsedDevice)
    Tensor.fromNativeHandle[T](handle)
  }

  private[api] def resolve(): Long = {
    NativeTensor.eagerResolve(nativeHandle)
  }

  private[api] def getElementAtFlattenedIndex(index: Int): T = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val offset = dataType match {
      case STRING =>
        val lengthsOffset = INT64.byteSize.get * size.toInt
        val length = DataType.getElementFromBuffer[Long](buffer, index * INT64.byteSize.get).toInt
        lengthsOffset + length
      case _ => index * dataType.byteSize.get
    }
    val value = DataType.getElementFromBuffer[T](
      buffer = buffer,
      index = offset
    )(TF.fromDataType(dataType))
    NativeHandleLock synchronized {
      if (resolvedHandle != 0)
        NativeTensor.delete(resolvedHandle)
    }
    value
  }

  @throws[InvalidShapeException]
  def scalar: T = {
    if (size != 1) {
      throw InvalidShapeException(
        "'Tensor.scalar' can only be called for scalar tensors " +
            "(i.e., containing only one element).")
    }
    getElementAtFlattenedIndex(0)
  }

  def entriesIterator: Iterator[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)

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

      private val buffer: ByteBuffer = {
        NativeTensor.buffer(resolved.handle).order(ByteOrder.nativeOrder)
      }

      Disposer.add(this, resolved)

      override def hasNext: Boolean = remaining > 0

      override def next(): T = {
        if (!hasNext)
          throw new NoSuchElementException
        assert(resolved.handle != 0)

        val nextElement: T = dataType match {
          case STRING =>
            val lengthsOffset = INT64.byteSize.get * (i + remaining)
            val length = DataType.getElementFromBuffer[Long](buffer, i * INT64.byteSize.get)
            DataType.getElementFromBuffer[T](
              buffer,
              lengthsOffset + length.ensuring(_ <= Int.MaxValue).toInt)
          case _ =>
            DataType.getElementFromBuffer[T](buffer, i * dataType.byteSize.get)
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

  def apply(
      firstIndexer: Indexer,
      otherIndexers: Indexer*
  ): Tensor[T] = {
    this.slice(firstIndexer, otherIndexers: _*)
  }

  /** Slices this tensor according to the provided indexers.
    *
    * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
    *
    * @param  firstIndexer  First indexer to use.
    * @param  otherIndexers Rest of the indexers to use.
    * @return Resulting tensor.
    */
  def slice(
      firstIndexer: Indexer,
      otherIndexers: Indexer*
  ): Tensor[T] = {
    val stridedSlice = Indexer.toStridedSlice(firstIndexer, otherIndexers: _*)
    val beginTensor: Tensor[Int] = stridedSlice._1
    val endTensor: Tensor[Int] = stridedSlice._2
    val stridesTensor: Tensor[Int] = stridedSlice._3
    ops.Basic.stridedSlice(
      this, beginTensor, endTensor, stridesTensor, stridedSlice._4,
      stridedSlice._5, stridedSlice._6, stridedSlice._7, stridedSlice._8
    )(TF.fromDataType(dataType), IsInt32OrInt64[Int], TF[Int])
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
  def summarize(
      maxEntries: Int = 6,
      flattened: Boolean = false,
      includeInfo: Boolean = true
  ): String = {
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
            def summarizeSlice(index: Int) = {
              implicit val evTF: TF[T] = TF.fromDataType(dataType)
              summarize(tensor(index).reshape(tensor.shape(1 ::)), maxEntries)
            }

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

  /** Returns this tensor. */
  override def toTensor: Tensor[T] = {
    this
  }

  /** Returns the tensor indexed slices that has the same value as this tensor. */
  override def toTensorIndexedSlices: TensorIndexedSlices[T] = {
    TensorIndexedSlices(
      indices = (0 until shape(0)).toTensor.toLong,
      values = this,
      denseShape = shape.toTensor)
  }

  def toOutput: Output[T] = {
    OutputBasic.constant(cpu())
  }

  override def toProto: TensorProto = {
    toTensorProto
  }

  /** Constructs and returns a [[TensorProto]] object that represents this tensor.
    *
    * @return Constructed [[TensorProto]].
    */
  def toTensorProto: TensorProto = {
    Tensor.makeProto(this)
  }

  /** Writes this tensor to the provided file, using the Numpy (i.e., `.npy`) file format. Note that this method will
    * replace the file, if it already exists. */
  def writeNPY(file: Path, fortranOrder: Boolean = false): Unit = {
    NPY.write(this, file, fortranOrder)(TF.fromDataType(dataType))
  }

  override def toString: String = {
    s"Tensor[$dataType, $shape]"
  }

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

  /** Closes this [[Tensor]] and releases any resources associated with it. Note that an [[Tensor]] is not
    * usable after it has been closed. */
  override def close(): Unit = {
    closeFn()
  }
}

object Tensor {
  private[tensors] val logger = Logger(LoggerFactory.getLogger("Tensor"))

  //region Constructors

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

  def empty[T: TF]: Tensor[T] = {
    Tensor.allocate[T](Shape(0))
  }

  def apply[T: TF](): Tensor[T] = {
    empty[T]
  }

  def apply[T: TF](tensors: Tensor[T]*): Tensor[T] = {
    Basic.stack(tensors, axis = 0)
  }

  /** Returns a new tensor with shape `shape` and all elements set to zero.
    *
    * For example:
    * {{{
    *   Tensor.zeros[Int](Shape(3, 4)) == Tensor(Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0))
    * }}}
    *
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Constructed tensor.
    */
  def zeros[T: TF](shape: Shape): Tensor[T] = {
    fill[T](shape)(DataType.zero[T])
  }

  /** Returns a new tensor with shape `shape` and all elements set to zero.
    *
    * For example:
    * {{{
    *   Tensor.zeros[Int](Shape(3, 4)) == Tensor(Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0), Tensor(0, 0, 0, 0))
    * }}}
    *
    * @param dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @return Constructed tensor.
    */
  def zeros[T](dataType: DataType[T], shape: Shape): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    fill[T](shape)(DataType.zero[T])
  }

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to zero. */
  def zerosLike[T](
      tensor: Tensor[T],
      shape: Shape = null
  ): Tensor[T] = {
    if (shape == null) {
      zeros(tensor.dataType, tensor.shape)
    } else {
      zeros(tensor.dataType, shape)
    }
  }

  /** Returns a new tensor with shape `shape` and all elements set to ones.
    *
    * For example:
    * {{{
    *   Tensor.ones[Int](Shape(3, 4)) == Tensor(Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1))
    * }}}
    *
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Constructed tensor.
    */
  def ones[T: TF](shape: Shape): Tensor[T] = {
    fill[T](shape)(DataType.one[T])
  }

  /** Returns a new tensor with shape `shape` and all elements set to ones.
    *
    * For example:
    * {{{
    *   Tensor.ones[Int](Shape(3, 4)) == Tensor(Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1), Tensor(1, 1, 1, 1))
    * }}}
    *
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @return Constructed tensor.
    */
  def ones[T](dataType: DataType[T], shape: Shape): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    fill[T](shape)(DataType.one[T])
  }

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to one. */
  def onesLike[T](
      tensor: Tensor[T],
      shape: Shape = null
  ): Tensor[T] = {
    if (shape == null) {
      ones(tensor.dataType, tensor.shape)
    } else {
      ones(tensor.dataType, shape)
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
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Constructed tensor.
    */
  def fill[T: TF](shape: Shape)(value: T): Tensor[T] = {
    shape.assertFullyDefined()
    val dataType = implicitly[TF[T]].dataType
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
          val numEncodedBytes = DataType.putElementInBuffer[T](buffer, baseOffset + index, value)
          DataType.putElementInBuffer[Long](buffer, i * INT64.byteSize.get, index.toLong)
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
          DataType.putElementInBuffer[T](buffer, index, value)
          index += dataType.byteSize.get
          i += 1
        }
        hostHandle
    }
    val tensor = Tensor.fromHostNativeHandle[T](hostHandle)
    NativeTensor.delete(hostHandle)
    tensor
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
  def fill[T](dataType: DataType[T], shape: Shape)(value: T): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    fill[T](shape)(value)
  }

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
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return New random tensor.
    */
  def rand[T: IsInt32OrInt64OrFloat16OrFloat32OrFloat64, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Tensor[I],
      minValue: Tensor[T] = null,
      maxValue: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val minValueWithDefault = if (minValue == null) Tensor.zeros(dataType, Shape()) else minValue
    val maxValueWithDefault = if (maxValue == null) Tensor.ones(dataType, Shape()) else maxValue
    if (dataType.isInteger) {
      Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomUniformInt(
        executionContext.value.nativeHandle, shape.nativeHandle, minValueWithDefault.nativeHandle,
        maxValueWithDefault.nativeHandle, graphSeed.getOrElse(0).toLong, opSeed.getOrElse(0).toLong))
    } else {
      val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomUniform(
        executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
        opSeed.getOrElse(0).toLong))
      Math.add(random * (maxValueWithDefault - minValueWithDefault), minValue)
    }
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
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return New random tensor.
    */
  def randn[T: IsFloat16OrFloat32OrFloat64, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Tensor[I],
      mean: Tensor[T] = null,
      standardDeviation: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Tensor.zeros(dataType, Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Tensor.ones(dataType, Shape()) else standardDeviation
    val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomStandardNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
    Math.add(random * standardDeviationWithDefault, meanWithDefault)
  }

  /** $OpDocRandomRandomTruncatedNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor.
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return Result as a new tensor.
    */
  def randTruncN[T: IsFloat16OrFloat32OrFloat64, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Tensor[I],
      mean: Tensor[T] = null,
      standardDeviation: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Tensor.zeros(dataType, Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Tensor.ones(dataType, Shape()) else standardDeviation
    val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.truncatedNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
    Math.add(random * standardDeviationWithDefault, meanWithDefault)
  }

  /** $OpDocBasicOneHot
    *
    * @group BasicOps
    * @param  indices  Tensor containing the indices for the "on" values.
    * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
    * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
    *                  Defaults to the value `1` with type `dataType`.
    * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
    *                  Defaults to the value `0` with type `dataType`.
    * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
    * @tparam T Data type of the resulting tensor.
    * @tparam I Indices tensor data type.
    * @return Result as a new tensor.
    */
  def oneHot[T: TF, I: IsInt32OrInt64OrUInt8](
      indices: Tensor[I],
      depth: Tensor[Int],
      onValue: Tensor[T] = null,
      offValue: Tensor[T] = null,
      axis: Int = -1
  ): Tensor[T] = {
    val actualOnValue = if (onValue != null) onValue else Tensor.ones[T](Shape())
    val actualOffValue = if (offValue != null) offValue else Tensor.zeros[T](Shape())
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.oneHot(
      executionContext.value.nativeHandle, indices.nativeHandle, depth.nativeHandle, actualOnValue.nativeHandle,
      actualOffValue.nativeHandle, axis))
  }

  /** Allocates a new tensor without worrying about the values stored in it.
    *
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Allocated tensor.
    * @throws IllegalArgumentException If `T` is `String` and the shape is non-empty, because the number of bytes
    *                                  required for a string tensor are not known until all its element values
    *                                  are known.
    */
  @throws[IllegalArgumentException]
  private def allocate[T: TF](shape: Shape): Tensor[T] = {
    val dataType = implicitly[TF[T]].dataType
    val hostHandle = dataType match {
      case STRING if shape.numElements == 0 =>
        NativeTensor.allocate(dataType.cValue, Array[Long](0), 0)
      case STRING =>
        throw new IllegalArgumentException(
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
  def fromBuffer[T: TF](
      shape: Shape,
      numBytes: Long,
      buffer: ByteBuffer
  ): Tensor[T] = {
    val dataType = implicitly[TF[T]].dataType
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
  def fromNPY[T: TF](file: Path): Tensor[T] = {
    NPY.read(file)
  }

  //endregion Constructors

  @throws[InvalidArgumentException]
  def makeProto[T](value: Tensor[T]): TensorProto = {
    makeProto[T](value, value.shape)
  }

  @throws[InvalidArgumentException]
  def makeProto[T](
      value: Tensor[T],
      shape: Shape
  ): TensorProto = {
    val dataType = value.dataType
    val tensorProtoBuilder =
      TensorProto.newBuilder()
          .setDtype(dataType.protoType)
          .setTensorShape(shape.toTensorShapeProto)
    if (value.dataType != STRING && dataType.byteSize.get * value.size >= Int.MaxValue)
      throw InvalidArgumentException("Cannot serialize tensors whose content is larger than 2GB.")
    if (value.dataType != STRING && value.size == shape.numElements) {
      val resolvedHandle = value.resolve()
      val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
      tensorProtoBuilder.setTensorContent(ByteString.copyFrom(buffer))
      value.NativeHandleLock synchronized {
        if (resolvedHandle != 0)
          NativeTensor.delete(resolvedHandle)
      }
    } else {
      value.entriesIterator.foreach(v => {
        DataType.addToTensorProtoBuilder[T](tensorProtoBuilder, v)(TF.fromDataType(value.dataType))
      })
    }
    tensorProtoBuilder.build()
  }
}

/** Sparse representation of a set of tensor slices at given indices.
  *
  * This class if a simple wrapper for a pair (or a set of three) of [[Tensor]] objects:
  *   - `indices`: A one-dimensional integer [[Tensor]] with shape `[D0]`.
  *   - `values`: An [[Tensor]] of any data type, with shape `[D0, D1, ..., Dn]`.
  *   - `denseShape`: Optionally, an integer [[Tensor]] with shape `[LARGE0, D1, ..., Dn]`.
  *
  * An [[TensorIndexedSlices]] is typically used to represent a subset of a larger [[Output]], `dense`, of shape
  * `[LARGE0, D1, ..., Dn]`, where `LARGE0 >> D0`. The values in `indices` are the indices in the first dimension of
  * the slices that have been extracted from the larger tensor.
  *
  * The dense [[Tensor]], `dense`, represented by [[TensorIndexedSlices]], `slices`, has:
  * {{{
  *   dense(slices.indices(i), ::, ::, ...) = slices.values(i, ::, ::, ...)
  * }}}
  *
  * The [[TensorIndexedSlices]] class is used primarily in the definition of gradients for operations that have
  * sparse gradients, such as `gather`.
  *
  * Note that this is different than [[SparseTensor]] which uses multi-dimensional indices and scalar values.
  *
  * @param  indices    Indices along the first dimension of the corresponding dense [[Tensor]].
  * @param  values     Values corresponding to the provided indices.
  * @param  denseShape Shape of the corresponding dense [[Tensor]].
  *
  * @author Emmanouil Antonios Platanios
  */
final case class TensorIndexedSlices[+T](
    indices: Tensor[Long],
    values: Tensor[T],
    denseShape: Tensor[Long] = null
) extends TensorLike[T] {
  /** Data type of these tensor indexed slices. */
  override val dataType: DataType[T] = {
    values.dataType
  }

  /** Shape of these tensor indexed slices. */
  override val shape: Shape = {
    Shape(denseShape.toInt.entriesIterator.toSeq: _*)
  }

  /** Device on which these tensor indexed slices will be placed. */
  override val device: String = {
    values.device
  }

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  @throws[IllegalStateException]
  override def toTensor: Tensor[T] = {
    if (denseShape != null)
      throw new IllegalStateException(
        s"Conversion of 'TensorIndexedSlices', '$this', " +
            s"which has no dense shape information available, is not possible.")
    if (denseShape.prod().scalar > 100000000)
      Tensor.logger.warn(
        "Converting large (> 100000000 elements) tensor indexed slices object to a tensor " +
            "(may consume too much memory).")

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = new IsNumeric[T] {}

    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    Math.unsortedSegmentSum(
      data = values,
      segmentIndices = indices,
      segmentsNumber = denseShape(0).toInt)
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  override def toTensorIndexedSlices: TensorIndexedSlices[T] = {
    this
  }

  override def toString: String = {
    "TensorIndexedSlices[" +
        s"indices = $indices, " +
        s"values = $values, " +
        s"denseShape = $denseShape, " +
        s"device = $device]"
  }
}

/** Represents a sparse tensor.
  *
  * TensorFlow represents a sparse tensor as three separate dense tensors: `indices`, `values`, and `denseShape`. In
  * Scala, the three tensors are collected into a [[SparseTensor]] class for ease of use.  If you have separate
  * `indices`, `values`, and `denseShape` tensors, wrap them in a `SparseTensor` object before passing to the
  * relevant sparse tensor manipulation.
  *
  * Concretely, the sparse tensor `SparseTensor(indices, values, denseShape)` comprises the following components,
  * where `N` and `rank` are the number of values and number of dimensions in the [[SparseTensor]], respectively:
  *
  *   - `indices`: Two-dimensional [[INT64]] tensor with shape `[N, rank]`, which specifies the indices of the elements
  *     in the sparse tensor that have nonzero values (elements are zero-indexed). For example,
  *     `indices = [[1, 3], [2, 4]]` specifies that the elements with indexes `[1, 3]` and `[2, 4]` have nonzero
  *     values.
  *   - `values`: One-dimensional tensor of any type, with shape `[N]`, which supplies the values for each element in
  *     `indices`. For example, given `indices = [[1, 3], [2, 4]]`, the parameter `values = [18, 3.6]` specifies that
  *      element `[1, 3]` of the sparse tensor has a value of `18`, and element `[2, 4]` of the tensor has a value of
  *      `3.6`.
  *   - `denseShape`: One-dimensional [[INT64]] tensor with shape `[rank]`, which specifies the dense shape of the
  *     sparse tensor.  For example, `denseShape = [3, 6]` specifies a two-dimensional 3x6 tensor,
  *     `denseShape = [2, 3, 4]` specifies a three-dimensional 2x3x4 tensor, and `denseShape = [9]` specifies a
  *     one-dimensional tensor with 9 elements.
  *
  * The corresponding dense tensor, `dense`, satisfies:
  * {{{
  *   dense.shape == denseShape
  *   dense(indices(i)) = values(i) // Using a somewhat loose notation with respect to indexing.
  * }}}
  *
  * IMPORTANT NOTE: By convention, `indices` should be sorted in row-major order (or equivalently lexicographic order
  * on `indices(i)`). This is not enforced when `SparseTensor` objects are constructed, but most ops assume correct
  * ordering. If the ordering of sparse tensor `st` is wrong, a fixed version can be obtained by calling
  * `sparseReorder(st)`.
  *
  * For example, the sparse tensor `SparseTensor(indices = [[0, 0], [1, 2]], values = [1, 2], denseShape = [3, 4])`,
  * represents the dense tensor `[[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]`.
  *
  * @param  indices    Two-dimensional tensor with shape `[N, rank]`.
  * @param  values     One-dimensional tensor with shape `[N]`.
  * @param  denseShape One-dimensional tensor with shape `[rank]`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class SparseTensor[+T](
    indices: Tensor[Long],
    values: Tensor[T],
    denseShape: Tensor[Long]
) extends TensorLike[T] {
  Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
  Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

  /** Data type of this sparse op output. */
  override val dataType: DataType[T] = {
    values.dataType
  }

  /** Shape of this sparse tensor. */
  override val shape: Shape = {
    Shape(denseShape.toInt.entriesIterator.toSeq: _*)
  }

  /** Device on which this sparse op output will be placed. */
  override val device: String = {
    values.device
  }

  /** Returns the tensor that this [[TensorLike]] object represents. */
  override def toTensor: Tensor[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    toTensor()
  }

  /** Converts this sparse tensor to a dense tensor.
    *
    * The constructed op builds a tensor `dense` with shape `input.denseShape`, such that:
    * {{{
    *   // If input.indices is scalar:
    *   dense(i) ==> (i == input.indices ? input.values : defaultValue)
    *
    *   // If input.indices is a vector, then for each i:
    *   dense(input.indices(i)) ==> input.values(i)
    *
    *   // If input.indices is an n by d matrix, then for each i in [0, n):
    *   dense(input.indices(i)(0), ..., input.indices(i)(d-1)) ==> input.values(i)
    * }}}
    *
    * All other values in `dense` are set to `defaultValue`. If `input.values` is a scalar, then all sparse indices
    * are set to that single value.
    *
    * `input.indices` should be sorted in lexicographic order and they must not contain any repeats. If
    * `validateIndices` is `true`, then these properties are checked during execution.
    *
    * @param  defaultValue    Scalar tensor with the same data type as `input.values`, containing the value set for
    *                         indices that are not specified in `input.indices`.
    * @param  validateIndices If `true`, the indices in `input.indices` are checked to make sure that they are sorted in
    *                         lexicographic order and that there are no repeats.
    * @return Result as a new tensor, with the same data type as `input.values` and shape `input.denseShape`.
    */
  def toTensor[V >: T : TF](
      defaultValue: Tensor[V] = null,
      validateIndices: Boolean = true
  ): Tensor[T] = {
    val defaultValueWithDefault = if (defaultValue == null) Tensor.zeros[V](Shape()) else defaultValue
    Tensor.fromNativeHandle[T](NativeTensorOpsSparse.sparseToDense(
      executionContext.value.nativeHandle, indices.nativeHandle, denseShape.nativeHandle, values.nativeHandle,
      defaultValueWithDefault.nativeHandle, validateIndices))
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  @throws[UnsupportedOperationException]
  override def toTensorIndexedSlices: TensorIndexedSlices[T] = {
    throw new UnsupportedOperationException(
      s"Cannot convert sparse tensor '$this' to tensor indexed slices.")
  }

  override def toString: String = {
    "SparseTensor[" +
        s"indices = $indices, " +
        s"values = $values, " +
        s"denseShape = $denseShape, " +
        s"device = $device]"
  }
}
