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
import org.platanios.tensorflow.api.tensors.ops.Basic.{BasicOps, stack}
import org.platanios.tensorflow.api.tensors.ops.{Math, Random}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import org.platanios.tensorflow.jni.generated.tensors.{Sparse => NativeTensorOpsSparse}
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import shapeless.{Generic, HList, Lazy}
import java.nio._
import java.nio.charset.Charset

import scala.collection.{TraversableLike, breakOut}
import scala.language.{higherKinds, postfixOps}
import scala.util.DynamicVariable

/** Represents tensor-like objects.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorLike {
  /** Data type of this tensor. */
  val dataType: DataType

  /** Shape of this tensor. */
  val shape: Shape

  /** Device on which this tensor is stored. */
  val device: String

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  def toTensor: Tensor

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  def toTensorIndexedSlices: TensorIndexedSlices
}

object TensorLike {
  implicit def tensorLikeToTensor[T <: TensorLike](value: T): Tensor = value.toTensor
}

/** Type trait for defining functions operating on and returning tensors. */
private[tensors] trait TensorOps[T] {
  /** Applies a unary function to the provided tensor and returns the result.
    *
    * @param  tensorLike Tensor-like object to apply the unary op function on.
    * @param  function   Unary function to apply.
    * @return Resulting tensor-like object that matches the type of `tensorLike`.
    */
  @inline
  def applyUnary(tensorLike: T, function: Tensor => Tensor): T
}

/** Companion object that defines supported [[TensorOps]] implicit values. */
private[tensors] object TensorOps {
  implicit val tensorOps: TensorOps[Tensor] = new TensorOps[Tensor] {
    @inline
    override def applyUnary(tensorLike: Tensor, function: Tensor => Tensor): Tensor = function(tensorLike)
  }

  implicit val tensorIndexedSlicesOps: TensorOps[TensorIndexedSlices] = new TensorOps[TensorIndexedSlices] {
    @inline
    override def applyUnary(tensorLike: TensorIndexedSlices, function: Tensor => Tensor): TensorIndexedSlices = {
      tensorLike.copy(values = function(tensorLike.values))
    }
  }

  implicit val sparseTensorOps: TensorOps[SparseTensor] = new TensorOps[SparseTensor] {
    @inline
    override def applyUnary(tensorLike: SparseTensor, function: Tensor => Tensor): SparseTensor = {
      tensorLike.copy(values = function(tensorLike.values))
    }
  }

  implicit val tensorLikeOps: TensorOps[TensorLike] = new TensorOps[TensorLike] {
    @inline
    override def applyUnary(tensorLike: TensorLike, function: Tensor => Tensor): TensorLike = {
      tensorLike match {
        case t: Tensor => function(t)
        case t: TensorIndexedSlices => t.copy(values = function(t.values))
        case t: SparseTensor => t.copy(values = function(t.values))
      }
    }
  }
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
class Tensor private[Tensor](private[api] var nativeHandle: Long) extends TensorLike with Closeable {
  /** Lock for the native handle. */
  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Data type of this tensor. */
  override val dataType: DataType = DataType.fromCValue(NativeTensor.eagerDataType(nativeHandle))

  /** Shape of this tensor. */
  override val shape: Shape = Shape.fromSeq(NativeTensor.eagerShape(nativeHandle).map(_.toInt))

  /** Rank of this tensor (i.e., number of dimensions). */
  def rank: Int = shape.rank

  /** Number of elements contained in this tensor. */
  def size: Int = shape.numElements

  /** Device in which the tensor is stored. */
  override val device: String = NativeTensor.eagerDevice(nativeHandle)

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
      NativeHandleLock synchronized {
        if (hostHandle != 0)
          NativeTensor.eagerDelete(hostHandle)
      }
      resolvedHandle
    }
  }

  private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val value = dataType match {
      case STRING =>
        val offset = INT64.byteSize * size + INT64.getElementFromBuffer(buffer, index * INT64.byteSize).toInt
        dataType.getElementFromBuffer(buffer, offset)
      case _ => dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
    }
    NativeHandleLock synchronized {
      if (resolvedHandle != 0)
        NativeTensor.delete(resolvedHandle)
    }
    value
  }

  @throws[InvalidShapeException]
  def scalar: dataType.ScalaType = {
    if (size != 1)
      throw InvalidShapeException(
        "'Tensor.scalar' can only be called for scalar tensors (i.e., containing only one element).")
    getElementAtFlattenedIndex(0)
  }

  def entriesIterator: Iterator[dataType.ScalaType] = {
    val resolvedHandle = resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val iterator = dataType match {
      case STRING =>
        val offset = INT64.byteSize * size
        Iterator.range(0, size).map(i => {
          dataType.getElementFromBuffer(buffer, offset + INT64.getElementFromBuffer(buffer, i * INT64.byteSize).toInt)
        })
      case _ => Iterator.range(0, size).map(i => dataType.getElementFromBuffer(buffer, i * dataType.byteSize))
    }
    NativeHandleLock synchronized {
      if (resolvedHandle != 0)
        NativeTensor.delete(resolvedHandle)
    }
    iterator
  }

  def apply(indexers: Indexer*): Tensor = this.slice(indexers: _*)

  def slice(indexers: Indexer*): Tensor = BasicOps(this).slice(indexers: _*)

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
    def summarize(tensor: Tensor, maxEntries: Int): String =
      tensor.rank match {
        case 0 => tensor.scalar.toString
        case 1 =>
          val slice =
            if (tensor.size <= math.max(maxEntries, 6))
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
  override def toTensor: Tensor = this

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  override def toTensorIndexedSlices: TensorIndexedSlices = {
    val denseShape = shape.toTensor(INT32)
    TensorIndexedSlices(indices = 0 until shape(0), values = this, denseShape = denseShape)
  }

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
  private[tensors] val logger = Logger(LoggerFactory.getLogger("Tensor"))

  private[api] def fromNativeHandle(nativeHandle: Long): Tensor = new Tensor(nativeHandle)

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

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to zero. */
  def zerosLike(tensor: Tensor): Tensor = zeros(tensor.dataType, tensor.shape)

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

  /** Returns a new tensor with the same data type and shape as the provided tensor, and all elements set to one. */
  def onesLike(tensor: Tensor): Tensor = ones(tensor.dataType, tensor.shape)

  /** $OpDocRandomRandomUniform
    *
    * @group RandomOps
    * @param  dataType Data type for the output tensor. Must be one of: [[FLOAT16]], [[FLOAT32]], [[FLOAT64]],
    *                  [[INT32]], or [[INT64]].
    * @param  shape    Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  minValue Scalar tensor containing the inclusive lower bound on the random of random values to generate.
    *                  Defaults to `0`.
    * @param  maxValue Scalar tensor containing the exclusive upper bound on the random of random values to generate.
    *                  Defaults to `1`.
    * @param  seed     Optional random seed, used to generate a random seed pair for the random number generator, when
    *                  combined with the graph-level seed.
    * @return New random tensor.
    */
  def rand(
      dataType: DataType = FLOAT32, shape: Tensor = Shape.scalar(), minValue: Tensor = 0.0, maxValue: Tensor = 1.0,
      seed: Option[Int] = None): Tensor = {
    Random.randomUniform(dataType, shape, minValue, maxValue, seed)
  }

  /** $OpDocRandomRandomNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor. Must be one of: [[FLOAT16]], [[FLOAT32]], or
    *                           [[FLOAT64]].
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @return New random tensor.
    */
  def randn(
      dataType: DataType = FLOAT32, shape: Tensor = Shape.scalar(), mean: Tensor = 0.0, standardDeviation: Tensor = 1.0,
      seed: Option[Int] = None): Tensor = {
    Random.randomNormal(dataType, shape, mean, standardDeviation, seed)
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
        synchronized {
          val tensor = Tensor.fromHostNativeHandle(hostHandle)
          NativeTensor.delete(hostHandle)
          tensor
        }
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
        synchronized {
          val tensor = Tensor.fromHostNativeHandle(hostHandle)
          NativeTensor.delete(hostHandle)
          tensor
        }
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
    case STRING =>
      if (shape.numElements == 0) {
        val hostHandle = NativeTensor.allocate(dataType.cValue, Array[Long](0), 0)
        val tensor = Tensor.fromHostNativeHandle(hostHandle)
        NativeTensor.delete(hostHandle)
        tensor
      } else {
        throw new IllegalArgumentException(
          "Cannot pre-allocate string tensors because their size is not known.")
      }
    case _ =>
      shape.assertFullyDefined()
      val numBytes = shape.numElements * dataType.byteSize
      val hostHandle = NativeTensor.allocate(dataType.cValue, shape.asArray.map(_.toLong), numBytes)
      val tensor = Tensor.fromHostNativeHandle(hostHandle)
      NativeTensor.delete(hostHandle)
      tensor
  }

  @throws[IllegalArgumentException]
  def fromBuffer(dataType: DataType, shape: Shape, numBytes: Long, buffer: ByteBuffer): Tensor = this synchronized {
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
final case class TensorIndexedSlices private(indices: Tensor, values: Tensor, denseShape: Tensor = null)
    extends TensorLike {
  /** Data type of these tensor indexed slices. */
  override val dataType: DataType = values.dataType

  /** Shape of these tensor indexed slices. */
  override val shape: Shape = Shape(denseShape.entriesIterator.map(_.asInstanceOf[Long].toInt).toSeq: _*)

  /** Device on which these tensor indexed slices will be placed. */
  override val device: String = values.device

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  @throws[IllegalStateException]
  override def toTensor: Tensor = {
    if (denseShape != null)
      throw new IllegalStateException(
        s"Conversion of 'TensorIndexedSlices', '$this', " +
            s"which has no dense shape information available, is not possible.")
    if (denseShape.prod().cast(INT32).scalar.asInstanceOf[Int] > 100000000)
      Tensor.logger.warn(
        "Converting large (> 100000000 elements) tensor indexed slices object to a tensor " +
            "(may consume too much memory).")
    Math.unsortedSegmentSum(data = values, segmentIndices = indices, segmentsNumber = denseShape(0))
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  override def toTensorIndexedSlices: TensorIndexedSlices = this

  override def toString: String = {
    s"TensorIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
        s"device = $device)}"
  }
}

/** Represents a sparse op output.
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
  * @param  indices    Two-dimensional [[INT64]] tensor with shape `[N, rank]`.
  * @param  values     One-dimensional tensor with shape `[N]`.
  * @param  denseShape One-dimensional [[INT64]] tensor with shape `[rank]`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class SparseTensor(indices: Tensor, values: Tensor, denseShape: Tensor) extends TensorLike {
  // TODO: Add constructor from scala arrays?
  require(indices.dataType == INT64, s"Indices cannot have '${indices.dataType}' data type. They have to be 'INT64'.")
  require(denseShape.dataType == INT64,
          s"Dense shape cannot have '${denseShape.dataType}' data type. They have to be 'INT64'.")

  Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
  Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

  /** Data type of this sparse op output. */
  override val dataType: DataType = values.dataType

  /** Shape of this sparse tensor. */
  override val shape: Shape = Shape(denseShape.entriesIterator.map(_.asInstanceOf[Long].toInt).toSeq: _*)

  /** Device on which this sparse op output will be placed. */
  override val device: String = values.device

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  override def toTensor: Tensor = toTensor()

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
  def toTensor(
      defaultValue: Tensor = 0, validateIndices: Boolean = true)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsSparse.sparseToDense(
      context.value.nativeHandle, indices.nativeHandle, denseShape.nativeHandle, values.nativeHandle,
      defaultValue.nativeHandle, validateIndices))
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  @throws[UnsupportedOperationException]
  override def toTensorIndexedSlices: TensorIndexedSlices = {
    throw new UnsupportedOperationException(s"Cannot convert sparse tensor '$this' to tensor indexed slices.")
  }

  override def toString: String = {
    s"TensorIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
        s"device = $device)}"
  }
}

trait TensorConvertible[T] {
  // TODO: Add data type argument.
  /** Converts `value` to a dense tensor. */
  @inline def toTensor(value: T): Tensor
}

object TensorConvertible {
  implicit def tensorLikeTensorConvertible[T <: TensorLike]: TensorConvertible[T] = new TensorConvertible[T] {
    @inline override def toTensor(value: T): Tensor = value.toTensor
  }

  implicit val shapeTensorConvertible: TensorConvertible[Shape] = new TensorConvertible[Shape] {
    /** Converts `value` to a dense tensor. */
    @inline override def toTensor(value: Shape): Tensor = value.toTensor()
  }

  implicit val rangeTensorConvertible: TensorConvertible[Range] = new TensorConvertible[Range] {
    /** Converts `value` to a dense tensor. */
    @inline override def toTensor(value: Range): Tensor = {
      if (value.nonEmpty)
        Tensor(INT32, value.head, value.tail: _*)
      else
        Tensor(INT32)
    }
  }

  implicit def supportedTypeTensorConvertible[T](implicit ev: SupportedType[T]): TensorConvertible[T] = {
    new TensorConvertible[T] {
      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: T): Tensor = Tensor.fill(ev.dataType, Shape())(value)
    }
  }

  implicit def arrayExecutable[T](implicit ev: TensorConvertible[T]): TensorConvertible[Array[T]] = {
    new TensorConvertible[Array[T]] {
      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Array[T]): Tensor = stack(value.map(ev.toTensor))
    }
  }

  implicit def traversableExecutable[T, CC[A] <: TraversableLike[A, CC[A]]](
      implicit ev: TensorConvertible[T]): TensorConvertible[CC[T]] = {
    new TensorConvertible[CC[T]] {
      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: CC[T]): Tensor = stack(value.map(ev.toTensor)(breakOut))
    }
  }

  implicit def productExecutable[T <: Product, L <: HList](implicit
      gen: Generic.Aux[T, L],
      ev: Lazy[TensorConvertible[L]]
  ): TensorConvertible[T] = new TensorConvertible[T] {
    /** Converts `value` to a dense tensor. */
    @inline override def toTensor(value: T): Tensor = {
      ev.value.toTensor(gen.to(value))
    }
  }
}
