package org.platanios.tensorflow.api.tensors

import java.nio.ByteOrder

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowNative {
  private[api] class DataTypeOps(val dataType: DataType) extends AnyVal {
    private[api] def tensorFromTFNativeHandle(nativeHandle: Long): Tensor = {
      val tensor = dataType match {
        case TFString =>
          new StringTensor(
            shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
            buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
            order = RowMajorOrder)
        case d: RealNumericDataType =>
          new RealNumericTensor(
            dataType = d, shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
            buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
            order = RowMajorOrder)
        case d: NumericDataType =>
          new NumericTensor(
            dataType = d, shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
            buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
            order = RowMajorOrder)
        case d: FixedSizeDataType =>
          new FixedSizeTensor(
            dataType = d, shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
            buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
            order = RowMajorOrder)
        case d => throw InvalidDataTypeException(s"Tensors with data type '$d' are not supported on the Scala side.")
      }
      // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
      // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
      // potential memory leak.
      Disposer.add(tensor, () => NativeTensor.delete(nativeHandle))
      tensor
    }
  }

  private[api] final class NativeView(private[api] var nativeHandle: Long) extends Closeable {
    override def close(): Unit = {
      if (nativeHandle != 0) {
        NativeTensor.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

  private[api] class NativeViewOps(tensor: Tensor) {
    // TODO: This will sometimes copy sometimes not (e.g., for TensorSlice, the data are copied -- non-contiguous).
    private[api] def nativeView: NativeView = {
      if (tensor.order != RowMajorOrder)
        throw new IllegalArgumentException("Only row-major tensors can be used in the TensorFlow native library.")
      new NativeView(NativeTensor.fromBuffer(
        tensor.buffer, tensor.dataType.cValue, tensor.shape.asArray.map(_.toLong),
        tensor.numElements * tensor.dataType.byteSize))
    }
  }

  trait Implicits {
    private[api] implicit def dataTypeOps(dataType: DataType): DataTypeOps = new DataTypeOps(dataType)

    // TODO: !!! What about the TensorSlice native view? It needs to make a copy.
    private[api] implicit def nativeViewOps(tensor: Tensor): NativeViewOps = new NativeViewOps(tensor)
  }
}
