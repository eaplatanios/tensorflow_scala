package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import org.platanios.tensorflow.api.Exception.ShapeMismatchException

import java.nio._
import java.nio.charset.Charset

import scala.annotation.tailrec

// TODO: Specialized slices (e.g., contiguous).
// TODO: Is there a need to complicate the flattened index function for the plain tensor?
// TODO: Add casting support.
// TODO: Should we keep assuming that tensor shapes are fully defined here?

/**
  * @author Emmanouil Antonios Platanios
  */
sealed class Tensor protected (
    val dataType: DataType, val shape: Shape, private[api] val buffer: ByteBuffer,
    val order: Tensor.Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER, private[api] val nativeHandle: Long = 0) {
  require(shape.numElements.get > 0, "")
  require(shape.isFullyDefined, s"The shape of a Tensor object must be fully defined. Shape '$shape' is not.")

  // TODO: Remove these from the plain tensor --- use them only for slices.
  private[api] val underlyingTensorDimensions: Array[Int] = shape.asArray
  val beginOffsets: Array[Int] = Array.fill(shape.rank)(0)
  val endOffsets  : Array[Int] = shape.asArray
  val strides     : Array[Int] = Array.fill(shape.rank)(1)

  def rank: Int = shape.rank
  def numElements: Int = shape.numElements.get // TODO: Convert this to an option?

  protected def flattenedIndex(indices: Array[Int]): Int = {
    order.index(underlyingTensorDimensions, beginOffsets, strides, indices)
  }

  protected def flattenedIndexIterator: Iterator[Int] = {
    order.flattenedIndexIterator(underlyingTensorDimensions, beginOffsets, endOffsets, strides)
  }

  // def update(indices: Array[Int], value: DataType.SupportedScalaType): Unit = {
  //   require(indices.length == rank, "Incomplete set of indices provided.")
  //   val index = order.index(shape.asArray, beginOffsets, strides, indices)
  //   dataType.putElementInBuffer(buffer = buffer, index = index, element = dataType.cast(value))
  // }
  //
  // def update(indexers: Seq[Indexer], value: DataType.SupportedScalaType): Unit = {
  //   val castedValue = dataType.cast(value)
  //   for (index <- slice(indexers: _*).flattenedIndexIterator)
  //     dataType.putElementInBuffer(buffer = buffer, index = index, element = castedValue)
  // }
  //
  // def update(index: Int, tensor: Tensor): Unit = update(Seq[Indexer](index), tensor)
  //
  // def update(indexers: Seq[Indexer], tensor: Tensor): Unit = slice(indexers: _*).set(tensor)

  def fill(value: SupportedScalaType): Tensor = {
    val castedValue: dataType.ScalaType = dataType.cast(value)
    dataType match {
      case DataType.Str =>
        throw new UnsupportedOperationException("String tensors are immutable in the TensorFlow Scala API.")
      case _: DataType  =>
        for (index <- flattenedIndexIterator)
          dataType.putElementInBuffer(buffer = buffer, index = index, element = castedValue)
    }
    this
  }

  // TODO: Find a way to add this method for performance benefits.
  // def set(value: SupportedScalaType): Tensor = fill(value)

  def set(tensor: Tensor): Tensor = {
    if (shape != tensor.shape && tensor.numElements != 1)
      throw ShapeMismatchException(s"Assigned tensor shape '${tensor.shape}' does not match assignee shape '$shape'")
    dataType match {
      case DataType.Str =>
        throw new UnsupportedOperationException("String tensors are immutable in the TensorFlow Scala API.")
      case _: DataType  =>
        if (tensor.numElements == 1) {
          dataType.putElementInBuffer(buffer = buffer, index = 0, element = dataType.cast(tensor.scalar))
        } else {
          for ((thisIndex, tensorIndex) <- flattenedIndexIterator zip tensor.flattenedIndexIterator)
            setElementAtFlattenedIndex(thisIndex, tensor.getElementAtFlattenedIndex(tensorIndex))
        }
    }
    this
  }

  def setElementAtFlattenedIndex(index: Int, value: SupportedScalaType): Tensor = {
    dataType match {
      case DataType.Str =>
        throw new UnsupportedOperationException("String tensors are immutable in the TensorFlow Scala API.")
      case _: DataType  => dataType.putElementInBuffer(buffer, index * dataType.byteSize, dataType.cast(value))
    }
    this
  }

  def scalar: dataType.ScalaType = {
    if (numElements != 1)
      throw new IllegalStateException(s"Cannot obtain a scalar value from a non-scalar tensor with shape '$shape'.")
    getElementAtFlattenedIndex(flattenedIndex(Array.fill[Int](shape.rank)(0))) // TODO: Fix this.
  }

  def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = {
    dataType match {
      case DataType.Str =>
        val numElements = underlyingTensorDimensions.product
        val offset = DataType.Int64.byteSize * numElements +
            DataType.Int64.getElementFromBuffer(buffer, index * DataType.Int64.byteSize).toInt
        dataType.getElementFromBuffer(buffer, offset)
      case _: DataType  => dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
    }
  }

  def apply(indexers: Indexer*): Tensor = {
    if (shape.rank == 0 && indexers.length == 1
        && indexers.head.isInstanceOf[Index] && indexers.head.asInstanceOf[Index].index == 0)
      this
    else
      slice(indexers: _*)
    //    if (dataType.byteSize == -1)
    //      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
    //    // TODO: Add checks for whether the indexers provided are within bounds.
    //    if ((indexers.length == rank || (indexers.length == 1 && rank == 0)) && indexers.forall(_.isInstanceOf[Index])) {
    //      val index = flattenedIndex(indexers.map(_.asInstanceOf[Index].index).toArray)
    //      dataType.getElementFromBuffer(buffer = buffer, index = index * dataType.byteSize)
    //    } else {
    //      throw InvalidIndexerException(
    //        "Only sequences of single indices that match in length the rank of a tensor, are supported for obtaining the " +
    //            "value of a tensor element.")
    //    }
  }

  // TODO: Return Tensor objects for contiguous slices.
  def slice(indexers: Indexer*): Tensor = TensorSlice(tensor = this, indexers = indexers)

  // TODO: Use this for creating slices: Buffer.slice().position(sliceStart).limit(sliceSize).

  // TODO: This will sometimes copy sometimes not (e.g., for TensorSlice, the data are copied -- non-contiguous).
  private[api] def nativeView: Tensor.NativeView = {
    Tensor.NativeView(NativeTensor.fromBuffer(
      buffer, dataType.cValue, shape.asArray.map(_.toLong), numElements * dataType.byteSize))
  }

  // TODO: Change the string representation of tensor objects.
  override def toString: String = s"$dataType Tensor with shape [${shape.asArray.mkString(", ")}]."

  // TODO: Add implementations for equals and hashCode.
}

final case class TensorSlice(tensor: Tensor, indexers: Seq[Indexer])
    extends Tensor(tensor.dataType, tensor.shape, tensor.buffer, tensor.order) {
  override val (underlyingTensorDimensions, shape, beginOffsets, endOffsets, strides) = {
    val decoded = Indexer.decode(tensor.shape, indexers)
    (decoded._1, Shape.fromSeq(decoded._2), decoded._3, decoded._4, decoded._5)
  }

  override def rank: Int = shape.rank
  override def numElements: Int = shape.numElements.get

  // TODO: !!! This has to make a copy.
  private[api] override def nativeView: Tensor.NativeView = {
    Tensor.NativeView(NativeTensor.fromBuffer(
      buffer, dataType.cValue, shape.asArray.map(_.toLong), numElements * dataType.byteSize))
  }

  //  def apply(indexers: Indexer*): Any = {
  //    if (tensor.dataType.byteSize == -1)
  //      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
  //    // TODO: Add checks for whether the indexers provided are within bounds.
  //    val elementIndex = tensor.order.index(tensor.shape, this.indexers, indexers: _*) * tensor.dataType.byteSize
  //    tensor.dataType.getElementFromBuffer(buffer = tensor.buffer, index = elementIndex)
  //  }
}

object Tensor {
  private[api] final case class NativeView(private[api] var nativeHandle: Long) extends Closeable {
    override def close(): Unit = {
      if (nativeHandle != 0) {
        NativeTensor.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

  private[api] def fromNativeHandle(nativeHandle: Long): Tensor = {
    val tensor = new Tensor(
      dataType = DataType.fromCValue(NativeTensor.dataType(nativeHandle)),
      shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
      buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
      order = RowMajorOrder, nativeHandle = nativeHandle)
    // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(tensor, () => NativeTensor.delete(nativeHandle))
    tensor
  }

  private def allocate(
      dataType: DataType, shape: Shape, order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor = {
    if (dataType.byteSize < 0)
      throw new IllegalArgumentException(s"Unsupported data type '$dataType'.")
    val numBytes: Int = dataType.byteSize * shape.numElements.get
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
    new Tensor(dataType = dataType, shape = shape, buffer = buffer, order = order)
  }

  def fill(dataType: DataType = null, shape: Shape = null)(value: SupportedScalaType): Tensor = {
    // TODO: Add downcasting warnings.
    val inferredDataType = if (dataType == null) DataType.dataTypeOf(value) else dataType
    val inferredShape = if (shape == null) Shape() else shape
    inferredDataType match {
      case DataType.Str =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = inferredShape.numElements.get * (DataType.Int64.byteSize + numEncodedBytes)
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
        val tensor = new Tensor(DataType.Str, inferredShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = DataType.Int64.byteSize * tensor.numElements
        var index = 0
        var i = 0
        while (i < tensor.numElements) {
          DataType.Str.putElementInBuffer(buffer, baseOffset + index, value.toStr)
          DataType.Int64.putElementInBuffer(buffer, i * DataType.Int64.byteSize, index.toLong)
          index += numEncodedBytes
          i += 1
        }
        tensor
      case _: DataType  => allocate(dataType = inferredDataType, shape = inferredShape).fill(value)
    }
  }

  // TODO: Find a way to add this method for performance benefits.
  //  def apply(value: SupportedScalaType, values: SupportedScalaType*): Tensor = {
  //    val allValues = value +: values
  //    val dataType = allValues.map(DataType.dataTypeOf).maxBy(_.priority)
  //    val shape = if (allValues.length > 1) Shape(allValues.length) else Shape()
  //    val tensor = allocate(dataType = dataType, shape = shape)
  //    val tensorIndexIterator = tensor.flattenedIndexIterator
  //    var i = 0
  //    while (i < allValues.length) {
  //      tensor.setElementAtFlattenedIndex(tensorIndexIterator.next(), allValues(i))
  //      i += 1
  //    }
  //    tensor
  //  }

  def apply(tensors: Tensor*): Tensor = apply(dataType = tensors.map(_.dataType).maxBy(_.priority), tensors: _*)

  def apply(dataType: DataType, tensors: Tensor*): Tensor = {
    // TODO: What about column-major string tensors?
    val shape = tensors.head.shape
    require(tensors.tail.forall(_.shape == shape), "All provided tensor shapes must match.")
    val newShape = Shape(tensors.length +: shape.asArray: _*)
    dataType match {
      case DataType.Str =>
        // TODO: Make this more efficient.
        val numElements = newShape.numElements.get
        var size = 0
        var t = 0
        while (t < tensors.length) {
          size += tensors(t).buffer.capacity() // TODO: This will not work with slices.
          t += 1
        }
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
        val tensor = new Tensor(DataType.Str, newShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = DataType.Int64.byteSize * tensor.numElements
        var byteIndex = 0
        var elementIndex = 0
        t = 0
        while (t < tensors.length) {
          val otherBaseOffset = tensors(t).numElements * DataType.Int64.byteSize
          var i = 0
          while (i < tensors(t).numElements) {
            val otherOffset = otherBaseOffset +
                DataType.Int64.getElementFromBuffer(tensors(t).buffer, i * DataType.Int64.byteSize).toInt
            val string = DataType.Str.getElementFromBuffer(tensors(t).buffer, otherOffset)
            val numEncodedBytes = DataType.Str.putElementInBuffer(buffer, baseOffset + byteIndex, string)
            DataType.Int64.putElementInBuffer(buffer, elementIndex * DataType.Int64.byteSize, byteIndex.toLong)
            byteIndex += numEncodedBytes
            elementIndex += 1
            i += 1
          }
          t += 1
        }
        tensor
      case _: DataType  =>
        val tensor = allocate(dataType, newShape)
        val newTensorIndexIterator = tensor.flattenedIndexIterator
        tensors.foreach(t => t.flattenedIndexIterator.foreach(index => {
          tensor.setElementAtFlattenedIndex(newTensorIndexIterator.next(), t.getElementAtFlattenedIndex(index))
        }))
        tensor
    }
  }

  private[api] sealed trait Order {
    def index(dimensions: Array[Int], beginOffsets: Array[Int], strides: Array[Int], indices: Array[Int]): Int
    def flattenedIndexIterator(
        dimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int], strides: Array[Int]): Iterator[Int]
  }

  private[api] object RowMajorOrder extends Order {
    override def index(
        dimensions: Array[Int], beginOffsets: Array[Int], strides: Array[Int], indices: Array[Int]): Int = {
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < dimensions.length) {
        var sizesProduct: Int = 1
        var k: Int = dimension + 1
        while (k < dimensions.length) {
          sizesProduct *= dimensions(k)
          k += 1
        }
        index += sizesProduct * (beginOffsets(dimension) + indices(dimension) * strides(dimension))
        dimension += 1
      }
      index
    }

    override def flattenedIndexIterator(
        dimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int],
        strides: Array[Int]): Iterator[Int] = {
      if (dimensions.length > 0) {
        new Iterator[Int] {
          private val dimCount: Array[Int] = beginOffsets.clone()
          private val dimSizes: Array[Int] = dimensions.scanRight(1)(_ * _).takeRight(dimensions.length)
          private var dim     : Int        = dimensions.length - 1
          private var index   : Int        = {
            var i = 0
            var sum = 0
            while (i < dimensions.length) {
              sum += beginOffsets(i) * dimSizes(i)
              i += 1
            }
            sum
          }

          override def hasNext: Boolean = dimCount.head < endOffsets.head

          @tailrec
          override def next(): Int = {
            if (dim < dimensions.length - 1 && dimCount(dim) < endOffsets(dim)) {
              dim += 1
              next()
            } else if (dimCount(dim) < endOffsets(dim)) {
              val nextIndex = index
              dimCount(dim) += strides(dim)
              index += strides(dim)
              while (dim > 0 && dimCount(dim) >= endOffsets(dim)) {
                index += dimSizes(dim) * (strides(dim - 1) * dimensions(dim) - dimCount(dim) + beginOffsets(dim))
                dimCount(dim) = beginOffsets(dim)
                dim -= 1
                dimCount(dim) += strides(dim)
              }
              nextIndex
            } else {
              throw new NoSuchElementException("This flattened index iterator has reached its end.")
            }
          }
        }
      } else {
        Iterator.range(0, 1)
      }
    }
  }

  private[api] object ColumnMajorOrder extends Order {
    override def index(
        dimensions: Array[Int], beginOffsets: Array[Int], strides: Array[Int], indices: Array[Int]): Int = {
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < dimensions.length) {
        var sizesProduct: Int = 1
        var k: Int = 0
        while (k < dimension) {
          sizesProduct *= dimensions(k)
          k += 1
        }
        index += sizesProduct * (beginOffsets(dimension) + indices(dimension) * strides(dimension))
        dimension += 1
      }
      index
    }

    override def flattenedIndexIterator(
        dimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int],
        strides: Array[Int]): Iterator[Int] = {
      if (dimensions.length > 0) {
        new Iterator[Int] {
          private val dimCount: Array[Int] = beginOffsets.clone()
          private val dimSizes: Array[Int] = dimensions.scanLeft(1)(_ * _).take(dimensions.length)
          private var dim     : Int        = 0
          private var index   : Int        = beginOffsets.head * dimSizes.head

          override def hasNext: Boolean = dimCount.head < endOffsets.head

          @tailrec
          override def next(): Int = {
            if (dim > 0 && dimCount(dim) < endOffsets(dim)) {
              dim -= 1
              next()
            } else if (dimCount(dim) < endOffsets(dim)) {
              val nextIndex = index
              dimCount(dim) += strides(dim)
              index += strides(dim)
              while (dim < dimensions.length - 1 && dimCount(dim) >= endOffsets(dim)) {
                index += dimSizes(dim) * (strides(dim + 1) * dimensions(dim) - dimCount(dim) + beginOffsets(dim))
                dimCount(dim) = beginOffsets(dim)
                dim += 1
                dimCount(dim) += strides(dim)
              }
              nextIndex
            } else {
              throw new NoSuchElementException("This flattened index iterator has reached its end.")
            }
          }
        }
      } else {
        Iterator.range(0, 1)
      }
    }
  }

  //  def apply[T: DataType.SupportedScalaTypes#Member](values: T*): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(values.head)
  //    val shape: Shape = Shape(values.length)
  //    if (valueDataType != DataType.String) {
  //      null
  //    } else {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    }
  //  }
  //
  //  /** Creates a [[Tensor]].
  //    *
  //    * The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
  //    * (optionally) `shape` (see examples below).
  //    *
  //    * The argument `value` can be a constant value, or an array (potentially multi-dimensional) with elements of type
  //    * `dataType`. If `value` is a one-dimensional array, then its length should be less than or equal to the number of
  //    * elements implied by the `shape` argument (if specified). In the case where the array length is less than the
  //    * number of elements specified by `shape`, the last element in the array will be used to fill the remaining entries.
  //    *
  //    * The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
  //    *
  //    * The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not present,
  //    * the shape of `value` is used.
  //    *
  //    * **IMPORTANT NOTE** The data type argument and the shape arguments are not currently being used.
  //    *
  //    * @param  value       A constant value of data type `dataType`.
  //    * @param  dataType    Data type of the resulting tensor. If not provided, its value will be inferred from the type
  //    *                     of `value`.
  //    * @param  shape       Shape of the resulting tensor.
  //    * @param  verifyShape If `true` and `shape` is not `null`, then the shape of `value` will be verified (i.e., checked
  //    *                     to see if it is equal to the provided shape.
  //    * @return Created tensor.
  //    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
  //    *                               the provided `shape`.
  //    */
  //  def create(value: Any, dataType: DataType = null, shape: Shape = null, verifyShape: Boolean = false): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(value)
  //    val inferredDataType: DataType = if (dataType == null) valueDataType else dataType
  //    val inferredShape: Shape = if (shape == null) Tensor.shape(value) else shape
  //    // TODO: !!! Fix this so that it actually does verify the shape and the data type and does appropriate type casts.
  //    if (inferredDataType != DataType.String) {
  //      val numElements = inferredShape.numElements.get
  //      val byteSize = inferredDataType.byteSize * numElements
  //      val nativeHandle = NativeTensor.allocate(inferredDataType.cValue, inferredShape.asArray, byteSize)
  //      if (inferredDataType != valueDataType) {
  //        val tensor: Tensor = allocateForBuffer(dataType, inferredShape, numElements)
  //        castAndWriteTo(tensor.buffer, value, dataType)
  //        tensor
  //      } else {
  //        NativeTensor.setValue(nativeHandle, value)
  //        Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //      }
  //    } else if (inferredShape.rank != 0) {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    } else {
  //      val nativeHandle = NativeTensor.allocateScalarBytes(value.asInstanceOf[Array[Byte]])
  //      Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //    }
  //  }
  //
  //  private[this] def castAndWriteTo(buffer: ByteBuffer, value: Any, dataType: DataType): Unit = {
  //    // TODO: May be doable more efficiently.
  //    def writeToHelper(buffer: ByteBuffer, bufferIndex: Int, value: Any, dataType: DataType): Int = {
  //      value match {
  //        case array: Array[_] =>
  //          var bytesWritten = 0
  //          var i = 0
  //          while (i < array.length) {
  //            bytesWritten += writeToHelper(buffer, bufferIndex + bytesWritten, array(i), dataType)
  //            i += 1
  //          }
  //          bytesWritten
  //        case scalar =>
  //          dataType.putElementInBuffer(buffer, bufferIndex, dataType.cast(scalar))
  //          dataType.byteSize
  //      }
  //    }
  //    writeToHelper(buffer, 0, value, dataType)
  //  }
  //
  //  def create(shape: Shape, data: FloatBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float32, shape, data.remaining())
  //    tensor.buffer.asFloatBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: DoubleBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float64, shape, data.remaining())
  //    tensor.buffer.asDoubleBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: IntBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int32, shape, data.remaining())
  //    tensor.buffer.asIntBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: LongBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int64, shape, data.remaining())
  //    tensor.buffer.asLongBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(dataType: DataType, shape: Shape, data: ByteBuffer): Tensor = {
  //    val numRemaining: Int = {
  //      if (dataType != DataType.String) {
  //        if (data.remaining() % dataType.byteSize != 0)
  //          throw new IllegalArgumentException(s"A byte buffer with ${data.remaining()} bytes is not compatible with a " +
  //                                                 s"${dataType.toString} Tensor (${dataType.byteSize} bytes/element).")
  //        data.remaining() / dataType.byteSize
  //      } else {
  //        data.remaining()
  //      }
  //    }
  //    val tensor: Tensor = allocateForBuffer(dataType, shape, numRemaining)
  //    tensor.buffer.put(data)
  //    tensor
  //  }
  //  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  //  // a java.nio.Buffer.
  //  private def allocateForBuffer(dataType: DataType, shape: Shape, numBuffered: Int): Tensor = {
  //    val size: Long = shape.numElements.get
  //    val numBytes: Long = {
  //      if (dataType != DataType.String) {
  //        if (numBuffered != size)
  //          throw incompatibleBufferException(numBuffered, shape)
  //        size * dataType.byteSize
  //      } else {
  //        // DataType.String tensor encoded in a ByteBuffer.
  //        numBuffered
  //      }
  //    }
  //    val nativeHandle: Long = NativeTensor.allocate(dataType.cValue, shape.asArray.clone(), numBytes)
  //    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  //  }
  //
  //  private def incompatibleBufferException(buffer: Buffer, dataType: DataType): IllegalArgumentException = {
  //    new IllegalArgumentException(s"Cannot use ${buffer.getClass.getName} with a Tensor of type $dataType.")
  //  }
  //
  //  private def incompatibleBufferException(numElements: Int, shape: Shape): IllegalArgumentException = {
  //    new IllegalArgumentException(
  //      s"A buffer with $numElements elements is not compatible with a Tensor with shape '$shape'.")
  //  }
  //
  //  private def rank(value: Any): Int = {
  //    value match {
  //      // Array[Byte] is a DataType.STRING scalar.
  //      case _: Array[Byte] => 0
  //      case value: Array[_] => 1 + rank(value(0))
  //      case _ => 0
  //    }
  //  }
  //
  //  private def shape(value: Any): Shape = {
  //    def fillShape(value: Any, axis: Int, shape: Array[Long]): Unit = {
  //      if (shape != null && axis != shape.length) {
  //        if (shape(axis) == 0) {
  //          value match {
  //            case value: Array[_] => shape(axis) = value.length
  //            case _ => shape(axis) = 1
  //          }
  //        } else {
  //          val mismatchedShape = value match {
  //            case value: Array[_] => (shape(axis) != value.length, value.length)
  //            case _ => (shape(axis) != 1, 1)
  //          }
  //          if (mismatchedShape._1)
  //            throw new IllegalArgumentException(
  //              s"Mismatched lengths (${shape(axis)} and ${mismatchedShape._2}) for dimension $axis.")
  //        }
  //        value match {
  //          case value: Array[_] =>
  //            var i = 0
  //            while (i < value.length) {
  //              fillShape(value(i), axis + 1, shape)
  //              i += 1
  //            }
  //        }
  //      }
  //    }
  //
  //    val shapeArray = Array.ofDim[Long](rank(value))
  //    fillShape(value = value, axis = 0, shape = shapeArray)
  //    Shape.fromSeq(shapeArray)
  //  }
}
