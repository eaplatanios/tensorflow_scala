package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{Tensor => NativeTensor, TensorFlow}

import java.nio._

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Tensor[T](dataType: DataType[T], shape: Array[Long], var nativeHandle: Long)
    extends Closeable {
  def rank: Int = shape.length
  def numElements: Long = Tensor.numElements(shape)
  def numBytes: Int = buffer.remaining()

  private def buffer: ByteBuffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder())

  def scalarValue: T = dataType match {
    case DataType.float => NativeTensor.scalarFloat(nativeHandle).asInstanceOf[T]
    case DataType.double => NativeTensor.scalarDouble(nativeHandle).asInstanceOf[T]
    case DataType.int32 => NativeTensor.scalarInt(nativeHandle).asInstanceOf[T]
    case DataType.uint8 => ???
    case DataType.string => ???
    case DataType.int64 => NativeTensor.scalarLong(nativeHandle).asInstanceOf[T]
    case DataType.boolean => NativeTensor.scalarBoolean(nativeHandle).asInstanceOf[T]
    case _ => throw new IllegalArgumentException(
      s"DataType $dataType is not recognized in Scala (TensorFlow version ${TensorFlow.version}).")
  }

  def bytesValue: Array[Byte] = NativeTensor.scalarBytes(nativeHandle)

  def copyTo[R](array: Array[R]): Array[R] = {
    throwExceptionIfDataTypeIsIncompatible(array)
    NativeTensor.readNDArray(nativeHandle, array)
    array
  }

  def writeTo(buffer: FloatBuffer): Unit = {
    if (dataType != DataType.float)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asFloatBuffer())
  }

  def writeTo(buffer: DoubleBuffer): Unit = {
    if (dataType != DataType.double)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asDoubleBuffer())
  }

  def writeTo(buffer: IntBuffer): Unit = {
    if (dataType != DataType.int32)
      throw Tensor.incompatibleBufferException(buffer, dataType)
    buffer.put(this.buffer.asIntBuffer())
  }

  def writeTo(buffer: LongBuffer): Unit = {
    if (dataType != DataType.int64)
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
    if (Tensor.dataTypeOf(value) != dataType)
      throw new IllegalArgumentException(
        s"Cannot copy $dataType Tensor into an object of type ${value.getClass.getName}.")
    val valueShape: Array[Long] = Array.ofDim[Long](rank)
    Tensor.fillShape(value = value, axis = 0, shape = valueShape)
    var i: Int = 0
    while (i < valueShape.length) {
      if (valueShape(i) != shape(i))
        throw new IllegalArgumentException(
          s"Cannot copy Tensor with shape [${shape.mkString(", ")}] into an object with shape " +
              s"[${valueShape.mkString(", ")}].")
      i += 1
    }
  }

  override def toString: String = s"$dataType Tensor with shape [${shape.mkString(", ")}]."
}

object Tensor {
  def create(value: Any): Tensor[_] = {
    implicit val dataType: DataType[_] = dataTypeOf(value)
    val shape: Array[Long] = Array.ofDim[Long](rank(value))
    fillShape(value = value, axis = 0, shape = shape)
    if (dataType != DataType.string) {
      val byteSize = dataType.byteSize * numElements(shape)
      val nativeHandle = NativeTensor.allocate(dataType.cValue, shape, byteSize)
      NativeTensor.setValue(nativeHandle, value)
      Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
    } else if (shape.length != 0) {
      throw new UnsupportedOperationException(
        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
    } else {
      val nativeHandle = NativeTensor.allocateScalarBytes(value.asInstanceOf[Array[Byte]])
      Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
    }
  }

  def create(shape: Array[Long], data: FloatBuffer): Tensor[Float] = {
    val tensor: Tensor[Float] = allocateForBuffer(DataType.float, shape, data.remaining())
    tensor.buffer.asFloatBuffer().put(data)
    tensor
  }

  def create(shape: Array[Long], data: DoubleBuffer): Tensor[Double] = {
    val tensor: Tensor[Double] = allocateForBuffer(DataType.double, shape, data.remaining())
    tensor.buffer.asDoubleBuffer().put(data)
    tensor
  }

  def create(shape: Array[Long], data: IntBuffer): Tensor[Int] = {
    val tensor: Tensor[Int] = allocateForBuffer(DataType.int32, shape, data.remaining())
    tensor.buffer.asIntBuffer().put(data)
    tensor
  }

  def create(shape: Array[Long], data: LongBuffer): Tensor[Long] = {
    val tensor: Tensor[Long] = allocateForBuffer(DataType.int64, shape, data.remaining())
    tensor.buffer.asLongBuffer().put(data)
    tensor
  }

  def create[T](dataType: DataType[T], shape: Array[Long], data: ByteBuffer): Tensor[T] = {
    val numRemaining: Int = {
      if (dataType != DataType.string) {
        if (data.remaining() % dataType.byteSize != 0)
          throw new IllegalArgumentException(s"A byte buffer with ${data.remaining()} bytes is not compatible with a " +
                                                 s"${dataType.toString} Tensor (${dataType.byteSize} bytes/element).")
        data.remaining() / dataType.byteSize
      } else {
        data.remaining()
      }
    }
    val tensor: Tensor[T] = allocateForBuffer(dataType, shape, numRemaining)
    tensor.buffer.put(data)
    tensor
  }

  def fromNativeHandle(nativeHandle: Long): Tensor[_] = {
    val dataType: DataType[_] = DataType.fromCValue(NativeTensor.dataType(nativeHandle))
    val shape: Array[Long] = NativeTensor.shape(nativeHandle)
    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  }

  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  // a java.nio.Buffer.
  private def allocateForBuffer[T](dataType: DataType[T], shape: Array[Long], numBuffered: Int): Tensor[T] = {
    val size: Long = numElements(shape)
    val numBytes: Long = {
      if (dataType != DataType.string) {
        if (numBuffered != size)
          throw incompatibleBufferException(numBuffered, shape)
        size * dataType.byteSize
      } else {
        // DataType.String tensor encoded in a ByteBuffer.
        numBuffered
      }
    }
    val shapeCopy: Array[Long] = shape.clone()
    val nativeHandle: Long = NativeTensor.allocate(dataType.cValue, shapeCopy, numBytes)
    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  }

  private def incompatibleBufferException(buffer: Buffer, dataType: DataType[_]): IllegalArgumentException = {
    new IllegalArgumentException(s"Cannot use ${buffer.getClass.getName} with a Tensor of type $dataType.")
  }

  private def incompatibleBufferException(numElements: Int, shape: Array[Long]): IllegalArgumentException = {
    new IllegalArgumentException(
      s"A buffer with $numElements elements is not compatible with a Tensor with shape [${shape.mkString(", ")}].")
  }

  private def numElements(shape: Array[Long]): Long = {
    var n: Long = 1
    var i: Int = 0
    while (i < shape.length) {
      n *= shape(i)
      i += 1
    }
    n
  }

  private def dataTypeOf(value: Any): DataType[_] = {
    value match {
      // Array[Byte] is a DataType.STRING scalar.
      case value: Array[Byte] =>
        if (value.length == 0)
          throw new IllegalArgumentException("Cannot create a tensor with size 0.")
        DataType.string
      case value: Array[_] =>
        if (value.length == 0)
          throw new IllegalArgumentException("Cannot create a tensor with size 0.")
        dataTypeOf(value(0))
      case _: Float => DataType.float
      case _: Double => DataType.double
      case _: Int => DataType.int32
      case _: Long => DataType.int64
      case _: Boolean => DataType.boolean
      case _ => throw new IllegalArgumentException(s"Cannot create a tensor of type ${value.getClass.getName}.")
    }
  }

  private def rank(value: Any): Int = {
    value match {
      // Array[Byte] is a DataType.STRING scalar.
      case _: Array[Byte] => 0
      case value: Array[_] => 1 + rank(value(0))
      case _ => 0
    }
  }

  private def fillShape(value: Any, axis: Int, shape: Array[Long]): Unit = {
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
}
