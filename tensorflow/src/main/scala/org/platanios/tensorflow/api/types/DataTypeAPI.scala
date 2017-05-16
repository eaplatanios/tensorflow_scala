package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.types

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait DataTypeAPI {
  type SupportedType[T] = types.SupportedType[T]
  type FixedSizeSupportedType[T] = types.FixedSizeSupportedType[T]
  type NumericSupportedType[T] = types.NumericSupportedType[T]
  type SignedNumericSupportedType[T] = types.SignedNumericSupportedType[T]
  type RealNumericSupportedType[T] = types.RealNumericSupportedType[T]
  type ComplexNumericSupportedType[T] = types.ComplexNumericSupportedType[T]

  type DataType = types.DataType
  type FixedSizeDataType = types.FixedSizeDataType
  type NumericDataType = types.NumericDataType
  type SignedNumericDataType = types.SignedNumericDataType
  type RealNumericDataType = types.RealNumericDataType
  type ComplexNumericDataType = types.ComplexNumericDataType

  val DataType = types.DataType

  val STRING   = types.STRING
  val BOOLEAN  = types.BOOLEAN
  // val FLOAT16 = types.TFFloat16
  val FLOAT32  = types.FLOAT32
  val FLOAT64  = types.FLOAT64
  // val BFLOAT16 = types.TFBFloat16
  // val COMPLEX64 = types.TFComplex64
  // val COMPLEX128 = types.TFComplex128
  val INT8     = types.INT8
  val INT16    = types.INT16
  val INT32    = types.INT32
  val INT64    = types.INT64
  val UINT8    = types.UINT8
  val UINT16   = types.UINT16
  val QINT8    = types.QINT8
  val QINT16   = types.QINT16
  val QINT32   = types.QINT32
  val QUINT8   = types.QUINT8
  val QUINT16  = types.QUINT16
  val RESOURCE = types.RESOURCE
}
