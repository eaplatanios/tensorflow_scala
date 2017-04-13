package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{TensorFlow, Tensor => NativeTensor}
import org.platanios.tensorflow.api.Exception.{InvalidIndexerException, InvalidShapeException}
import java.nio._

import scala.collection.mutable.ArrayBuffer

// TODO: Make this class internal to the API so we do not have to expose memory management for it.
/**
  * @author Emmanouil Antonios Platanios
  */
final case class Tensor(
    dataType: DataType, shape: Shape, order: Tensor.Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER,
    private[api] var nativeHandle: Long) extends Closeable {
  def rank: Int = shape.rank
  def numElements: Long = shape.numElements.get
  def numBytes: Int = buffer.remaining()

  private[api] lazy val buffer: ByteBuffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder())

  def apply(indexers: Indexer*): Any = {
    if (dataType.byteSize == -1)
      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
    // TODO: Add checks for whether the indexers provided are within bounds.
    if (indexers.length == rank && indexers.forall(_.isInstanceOf[Index]))
      dataType.getElementFromBuffer(
        buffer = buffer, index = order.index(shape, indexers: _*) * dataType.byteSize)
    else
      slice(indexers: _*)
  }

  def slice(indexers: Indexer*): TensorSlice = TensorSlice(tensor = this, indexers = indexers)

  // TODO: Use this for creating slices: Buffer.slice().position(sliceStart).limit(sliceSize).

  def scalarValue: Any = dataType match {
    case DataType.Float32 => NativeTensor.scalarFloat(nativeHandle)
    case DataType.Float64 => NativeTensor.scalarDouble(nativeHandle)
    case DataType.Int32 => NativeTensor.scalarInt(nativeHandle)
    case DataType.UInt8 => ???
    case DataType.String => ???
    case DataType.Int64 => NativeTensor.scalarLong(nativeHandle)
    case DataType.Boolean => NativeTensor.scalarBoolean(nativeHandle)
    case _ => throw new IllegalArgumentException(
      s"DataType '$dataType' is not recognized in the TensorFlow Scala API (TensorFlow version ${TensorFlow.version}).")
  }

  def bytesValue: Array[Byte] = NativeTensor.scalarBytes(nativeHandle)

  def copyTo[R](array: Array[R]): Array[R] = {
    throwExceptionIfDataTypeIsIncompatible(array)
    NativeTensor.readNDArray(nativeHandle, array)
    array
  }

  def writeTo(buffer: FloatBuffer): Unit = {
    if (dataType != DataType.Float32)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asFloatBuffer())
  }

  def writeTo(buffer: DoubleBuffer): Unit = {
    if (dataType != DataType.Float64)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asDoubleBuffer())
  }

  def writeTo(buffer: IntBuffer): Unit = {
    if (dataType != DataType.Int32)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asIntBuffer())
  }

  def writeTo(buffer: LongBuffer): Unit = {
    if (dataType != DataType.Int64)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asLongBuffer())
  }

  def writeTo(buffer: ByteBuffer): Unit = buffer.put(this.buffer)

  override def close(): Unit = {
    if (nativeHandle != 0) {
      NativeTensor.delete(nativeHandle)
      nativeHandle = 0
    }
  }

  private def throwExceptionIfDataTypeIsIncompatible(value: Any): Unit = {
    if (Tensor.rank(value) != rank)
      throw new IllegalArgumentException(
        s"Cannot copy Tensor with $rank dimensions into an object with ${Tensor.rank(value)} dimensions.")
    if (DataType.dataTypeOf(value) != dataType)
      throw new IllegalArgumentException(
        s"Cannot copy $dataType Tensor into an object of type ${value.getClass.getName}.")
    val valueShape = Tensor.shape(value)
    var i: Int = 0
    while (i < valueShape.rank) {
      if (valueShape(i) != shape(i))
        throw new IllegalArgumentException(
          s"Cannot copy Tensor with shape '$shape' into an object with shape '$valueShape'.")
      i += 1
    }
  }

  override def toString: String = s"$dataType Tensor with shape [${shape.asArray.mkString(", ")}]."
}

final case class TensorSlice(tensor: Tensor, indexers: Seq[Indexer]) {
  val shape: Shape = tensor.shape.applyIndexers(indexers: _*)

  def rank: Int = shape.rank
  def numElements: Long = shape.numElements.get
//  def numBytes: Int = tensor.buffer.remaining() // TODO: This is wrong

  private[api] lazy val buffer: ByteBuffer = NativeTensor.buffer(tensor.nativeHandle).order(ByteOrder.nativeOrder())

  def apply(indexers: Indexer*): Any = {
    if (tensor.dataType.byteSize == -1)
      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
    // TODO: Add checks for whether the indexers provided are within bounds.
    val elementIndex = tensor.order.index(tensor.shape, this.indexers, indexers: _*) * tensor.dataType.byteSize
    tensor.dataType.getElementFromBuffer(buffer = buffer, index = elementIndex)
  }
}

object Tensor {
  sealed trait Order {
    def index(shape: Shape, indexers: Indexer*): Int
    def index(shape: Shape, existingIndexers: Seq[Indexer], indexers: Indexer*): Int
  }

  object RowMajorOrder extends Order {
    override def index(shape: Shape, indexers: Indexer*): Int = {
      if (!indexers.forall(_.isInstanceOf[Index]))
        throw new IllegalArgumentException("Only integer indexers are supported for tensors.")
      if (indexers.length != shape.rank)
        throw new IllegalArgumentException("Only integer indexers over all dimensions of the tensor are supported.")
      val indices: Array[Int] = indexers.zipWithIndex.map(index => {
        val i = index._1.asInstanceOf[Index].index
        if (i < 0) i + shape(index._2).asInstanceOf[Int] - 1 else i
      }).toArray
      val shapeArray: Array[Int] = shape.asArray.map(_.asInstanceOf[Int]) // TODO: Make shapes integer based.
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < shape.rank) {
        var sizesProduct: Int = 1
        var k: Int = dimension + 1
        while (k < shape.rank) {
          sizesProduct *= shapeArray(k)
          k += 1
        }
        index += sizesProduct * indices(dimension)
        dimension += 1
      }
      index
    }

    override def index(shape: Shape, existingIndexers: Seq[Indexer], indexers: Indexer*): Int = {
      val newShape = shape.applyIndexers(existingIndexers: _*)
      val begin = ArrayBuffer[Int]()
      val stride = ArrayBuffer[Int]()
      val indices = ArrayBuffer[Int]()
      var newAxesIndices = ArrayBuffer[Int]()
      var ellipsisIndex: Int = -1
      // TODO: Implicitly added ellipsis to the end of the indexers sequence.
      existingIndexers.zipWithIndex foreach {
        case (Ellipsis, i) =>
          if (ellipsisIndex > -1)
            throw InvalidIndexerException("At most one ellipsis ('---') can be used per indexers sequence.")
          ellipsisIndex = i
        case (NewAxis, i) =>
          newAxesIndices += i
        case (Index(j), i) =>
          begin += (if (j < 0) j + shape(i).asInstanceOf[Int] - 1 else j)
          stride += 1
        case (Slice(sliceBegin, _, sliceStep, _), i) =>
          begin += (if (sliceBegin < 0) sliceBegin + shape(i).asInstanceOf[Int] - 1 else sliceBegin)
          stride += sliceStep
      }
      if (ellipsisIndex > -1) {
        val ellipsisLength = shape.rank - begin.length
        begin.insertAll(ellipsisIndex, Array.fill[Int](ellipsisLength)(0))
        stride.insertAll(ellipsisIndex, Array.fill[Int](ellipsisLength)(1))
        newAxesIndices = newAxesIndices.map(newAxisIndex => {
          if (newAxisIndex > ellipsisIndex)
            newAxisIndex + ellipsisLength
          else
            newAxisIndex
        })
      }
      indexers.zipWithIndex foreach {
        case (Index(k), i) =>
          if (!newAxesIndices.contains(i))
            indices += (if (k < 0) k + newShape(i).asInstanceOf[Int] - 1 else k)
      }
      val shapeArray: Array[Int] = shape.asArray.map(_.asInstanceOf[Int]) // TODO: Make shapes integer based.
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < shape.rank) {
        var sizesProduct: Int = 1
        var k: Int = dimension + 1
        while (k < shape.rank) {
          sizesProduct *= shapeArray(k)
          k += 1
        }
        index += sizesProduct * (begin(dimension) + indices(dimension) * stride(dimension))
        dimension += 1
      }
      index
    }
  }

  object ColumnMajorOrder extends Order {
    override def index(shape: Shape, indexers: Indexer*): Int = {
      if (!indexers.forall(_.isInstanceOf[Index]))
        throw new IllegalArgumentException("Only integer indexers are supported for tensors.")
      if (indexers.length != shape.rank)
        throw new IllegalArgumentException("Only integer indexers over all dimensions of the tensor are supported.")
      val indices: Array[Int] = indexers.zipWithIndex.map(index => {
        val i = index._1.asInstanceOf[Index].index
        if (i < 0) i + shape(index._2).asInstanceOf[Int] - 1 else i
      }).toArray
      val shapeArray: Array[Int] = shape.asArray.map(_.asInstanceOf[Int]) // TODO: Make shapes integer based.
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < shape.rank) {
        var sizesProduct: Int = 1
        var k: Int = 0
        while (k < dimension) {
          sizesProduct *= shapeArray(k)
          k += 1
        }
        index += sizesProduct * indices(dimension)
        dimension += 1
      }
      index
    }

    override def index(shape: Shape, existingIndexers: Seq[Indexer], indexers: Indexer*): Int = {
      var begin = ArrayBuffer[Int]()
      var stride = ArrayBuffer[Int]()
      var indices = ArrayBuffer[Int]()
      existingIndexers.zip(indexers).zipWithIndex foreach {
        case ((Ellipsis, Index(_)), _) =>
          ???
        case ((NewAxis, Index(k)), _) =>
          assert(k == 0)
        case ((Index(j), Index(k)), i) =>
          assert(k == 0)
          begin += (if (j < 0) j + shape(i).asInstanceOf[Int] - 1 else j)
          stride += 1
          indices += k
        case ((s@Slice(sliceBegin, _, sliceStep, _), Index(k)), i) =>
          begin += (if (sliceBegin < 0) sliceBegin + shape(i).asInstanceOf[Int] - 1 else sliceBegin)
          stride += sliceStep
          indices += (if (k < 0) k + s.length(shape(i).asInstanceOf[Int]) - 1 else k)
      }
      val shapeArray: Array[Int] = shape.asArray.map(_.asInstanceOf[Int]) // TODO: Make shapes integer based.
      var index: Int = 0
      var dimension: Int = 0
      while (dimension < shape.rank) {
        var sizesProduct: Int = 1
        var k: Int = 0
        while (k < dimension) {
          sizesProduct *= shapeArray(k)
          k += 1
        }
        index += sizesProduct * (begin(dimension) + indices(dimension) * stride(dimension))
        dimension += 1
      }
      index
    }
  }

  def apply[T: DataType.SupportedScalaTypes#Member](values: T*): Tensor = {
    val valueDataType: DataType = DataType.dataTypeOf(values.head)
    val shape: Shape = Shape(values.length)
    if (valueDataType != DataType.String) {
      null
    } else {
      // TODO: Support String tensors.
      throw new UnsupportedOperationException(
        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
    }
  }

  /** Creates a [[Tensor]].
    *
    * The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
    * (optionally) `shape` (see examples below).
    *
    * The argument `value` can be a constant value, or an array (potentially multi-dimensional) with elements of type
    * `dataType`. If `value` is a one-dimensional array, then its length should be less than or equal to the number of
    * elements implied by the `shape` argument (if specified). In the case where the array length is less than the
    * number of elements specified by `shape`, the last element in the array will be used to fill the remaining entries.
    *
    * The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
    *
    * The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not present,
    * the shape of `value` is used.
    *
    * **IMPORTANT NOTE** The data type argument and the shape arguments are not currently being used.
    *
    * @param  value       A constant value of data type `dataType`.
    * @param  dataType    Data type of the resulting tensor. If not provided, its value will be inferred from the type
    *                     of `value`.
    * @param  shape       Shape of the resulting tensor.
    * @param  verifyShape If `true` and `shape` is not `null`, then the shape of `value` will be verified (i.e., checked
    *                     to see if it is equal to the provided shape.
    * @return Created tensor.
    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
    *                               the provided `shape`.
    */
  def create(value: Any, dataType: DataType = null, shape: Shape = null, verifyShape: Boolean = false): Tensor = {
    val valueDataType: DataType = DataType.dataTypeOf(value)
    val inferredDataType: DataType = if (dataType == null) valueDataType else dataType
    val inferredShape: Shape = if (shape == null) Tensor.shape(value) else shape
    // TODO: !!! Fix this so that it actually does verify the shape and the data type and does appropriate type casts.
    if (inferredDataType != DataType.String) {
      val numElements = inferredShape.numElements.get.asInstanceOf[Int]
      val byteSize = inferredDataType.byteSize * numElements
      val nativeHandle = NativeTensor.allocate(inferredDataType.cValue, inferredShape.asArray, byteSize)
      if (inferredDataType != valueDataType) {
        val tensor: Tensor = allocateForBuffer(dataType, inferredShape, numElements)
        castAndWriteTo(tensor.buffer, value, dataType)
        tensor
      } else {
        NativeTensor.setValue(nativeHandle, value)
        Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
      }
    } else if (inferredShape.rank != 0) {
      // TODO: Support String tensors.
      throw new UnsupportedOperationException(
        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
    } else {
      val nativeHandle = NativeTensor.allocateScalarBytes(value.asInstanceOf[Array[Byte]])
      Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
    }
  }

  private[this] def castAndWriteTo(buffer: ByteBuffer, value: Any, dataType: DataType): Unit = {
    // TODO: May be doable more efficiently.
    def writeToHelper(buffer: ByteBuffer, bufferIndex: Int, value: Any, dataType: DataType): Int = {
      value match {
        case array: Array[_] =>
          var bytesWritten = 0
          var i = 0
          while (i < array.length) {
            bytesWritten += writeToHelper(buffer, bufferIndex + bytesWritten, array(i), dataType)
            i += 1
          }
          bytesWritten
        case scalar =>
          dataType.putElementInBuffer(buffer, bufferIndex, dataType.cast(scalar))
          dataType.byteSize
      }
    }
    writeToHelper(buffer, 0, value, dataType)
  }

  def create(shape: Shape, data: FloatBuffer): Tensor = {
    val tensor: Tensor = allocateForBuffer(DataType.Float32, shape, data.remaining())
    tensor.buffer.asFloatBuffer().put(data)
    tensor
  }

  def create(shape: Shape, data: DoubleBuffer): Tensor = {
    val tensor: Tensor = allocateForBuffer(DataType.Float64, shape, data.remaining())
    tensor.buffer.asDoubleBuffer().put(data)
    tensor
  }

  def create(shape: Shape, data: IntBuffer): Tensor = {
    val tensor: Tensor = allocateForBuffer(DataType.Int32, shape, data.remaining())
    tensor.buffer.asIntBuffer().put(data)
    tensor
  }

  def create(shape: Shape, data: LongBuffer): Tensor = {
    val tensor: Tensor = allocateForBuffer(DataType.Int64, shape, data.remaining())
    tensor.buffer.asLongBuffer().put(data)
    tensor
  }

  def create(dataType: DataType, shape: Shape, data: ByteBuffer): Tensor = {
    val numRemaining: Int = {
      if (dataType != DataType.String) {
        if (data.remaining() % dataType.byteSize != 0)
          throw new IllegalArgumentException(s"A byte buffer with ${data.remaining()} bytes is not compatible with a " +
                                                 s"${dataType.toString} Tensor (${dataType.byteSize} bytes/element).")
        data.remaining() / dataType.byteSize
      } else {
        data.remaining()
      }
    }
    val tensor: Tensor = allocateForBuffer(dataType, shape, numRemaining)
    tensor.buffer.put(data)
    tensor
  }

  def fromNativeHandle(nativeHandle: Long): Tensor = {
    val dataType: DataType = DataType.fromCValue(NativeTensor.dataType(nativeHandle))
    val shape: Array[Long] = NativeTensor.shape(nativeHandle)
    Tensor(dataType = dataType, shape = Shape.fromSeq(shape), nativeHandle = nativeHandle)
  }

  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  // a java.nio.Buffer.
  private def allocateForBuffer(dataType: DataType, shape: Shape, numBuffered: Int): Tensor = {
    val size: Long = shape.numElements.get
    val numBytes: Long = {
      if (dataType != DataType.String) {
        if (numBuffered != size)
          throw incompatibleBufferException(numBuffered, shape)
        size * dataType.byteSize
      } else {
        // DataType.String tensor encoded in a ByteBuffer.
        numBuffered
      }
    }
    val nativeHandle: Long = NativeTensor.allocate(dataType.cValue, shape.asArray.clone(), numBytes)
    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  }

  private def incompatibleBufferException(buffer: Buffer, dataType: DataType): IllegalArgumentException = {
    new IllegalArgumentException(s"Cannot use ${buffer.getClass.getName} with a Tensor of type $dataType.")
  }

  private def incompatibleBufferException(numElements: Int, shape: Shape): IllegalArgumentException = {
    new IllegalArgumentException(
      s"A buffer with $numElements elements is not compatible with a Tensor with shape '$shape'.")
  }

  private def rank(value: Any): Int = {
    value match {
      // Array[Byte] is a DataType.STRING scalar.
      case _: Array[Byte] => 0
      case value: Array[_] => 1 + rank(value(0))
      case _ => 0
    }
  }

  private def shape(value: Any): Shape = {
    def fillShape(value: Any, axis: Int, shape: Array[Long]): Unit = {
      if (shape != null && axis != shape.length) {
        if (shape(axis) == 0) {
          value match {
            case value: Array[_] => shape(axis) = value.length
            case _ => shape(axis) = 1
          }
        } else {
          val mismatchedShape = value match {
            case value: Array[_] => (shape(axis) != value.length, value.length)
            case _ => (shape(axis) != 1, 1)
          }
          if (mismatchedShape._1)
            throw new IllegalArgumentException(
              s"Mismatched lengths (${shape(axis)} and ${mismatchedShape._2}) for dimension $axis.")
        }
        value match {
          case value: Array[_] =>
            var i = 0
            while (i < value.length) {
              fillShape(value(i), axis + 1, shape)
              i += 1
            }
        }
      }
    }

    val shapeArray = Array.ofDim[Long](rank(value))
    fillShape(value = value, axis = 0, shape = shapeArray)
    Shape.fromSeq(shapeArray)
  }
}
