package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.DataType
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException

/**
  * @author Emmanouil Antonios Platanios
  */
object MathOps {
  def addN(inputs: Array[Op.Output], name: String = "AddN"): Op.Output =
    Op.Builder(opType = "AddN", name = name)
        .addInputs(inputs)
        .build().outputs(0)

  def matMul(
      a: Op.Output, b: Op.Output, transposeA: Boolean = false, transposeB: Boolean = false,
      name: String = "MatMul"): Op.Output = {
    Op.Builder(opType = "MatMul", name = name)
        .addInput(a)
        .addInput(b)
        .setAttribute(name = "transpose_a", value = transposeA)
        .setAttribute(name = "transpose_b", value = transposeB)
        .build().outputs(0)
  }

  def batchMatMul(
      x: Op.Output, y: Op.Output, adjointX: Boolean = false, adjointY: Boolean = false,
      name: String = "BatchMatMul"): Op.Output =
    Op.Builder(opType = "BatchMatMul", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute(name = "adj_x", value = adjointX)
        .setAttribute(name = "adj_y", value = adjointY)
        .build().outputs(0)

  def cast(x: Op.Output, dataType: DataType, name: String = "Cast"): Op.Output =
    Op.Builder(opType = "Cast", name = name)
        .addInput(x)
        .setAttribute(name = "DstT", value = dataType)
        .build().outputs(0)

  //region Unary Ops

  def abs(x: Op.Output, name: String = "Abs"): Op.Output =
    Op.Builder(opType = "Abs", name = name)
        .addInput(x)
        .build().outputs(0)

  def complexAbs(x: Op.Output, name: String = "ComplexAbs"): Op.Output =
    Op.Builder(opType = "ComplexAbs", name = name)
        .addInput(x)
        .build().outputs(0)

  def reciprocal(x: Op.Output, name: String = "Reciprocal"): Op.Output =
    Op.Builder(opType = "Reciprocal", name = name)
        .addInput(x)
        .build().outputs(0)

  def square(x: Op.Output, name: String = "Square"): Op.Output =
    Op.Builder(opType = "Square", name = name)
        .addInput(x)
        .build().outputs(0)

  def sqrt(x: Op.Output, name: String = "Sqrt"): Op.Output =
    Op.Builder(opType = "Sqrt", name = name)
        .addInput(x)
        .build().outputs(0)

  def reciprocalSqrt(x: Op.Output, name: String = "Rsqrt"): Op.Output =
    Op.Builder(opType = "Rsqrt", name = name)
        .addInput(x)
        .build().outputs(0)

  def round(x: Op.Output, name: String = "Round"): Op.Output =
    Op.Builder(opType = "Round", name = name)
        .addInput(x)
        .build().outputs(0)

  def exp(x: Op.Output, name: String = "Exp"): Op.Output =
    Op.Builder(opType = "Exp", name = name)
        .addInput(x)
        .build().outputs(0)

  def expMinus1(x: Op.Output, name: String = "Expm1"): Op.Output =
    Op.Builder(opType = "Expm1", name = name)
        .addInput(x)
        .build().outputs(0)

  def log(x: Op.Output, name: String = "Log"): Op.Output =
    Op.Builder(opType = "Log", name = name)
        .addInput(x)
        .build().outputs(0)

  def log1Plus(x: Op.Output, name: String = "Log1p"): Op.Output =
    Op.Builder(opType = "Log1p", name = name)
        .addInput(x)
        .build().outputs(0)

  def tanh(x: Op.Output, name: String = "Tanh"): Op.Output =
    Op.Builder(opType = "Tanh", name = name)
        .addInput(x)
        .build().outputs(0)

  def logGamma(x: Op.Output, name: String = "Lgamma"): Op.Output =
    Op.Builder(opType = "Lgamma", name = name)
        .addInput(x)
        .build().outputs(0)

  def digamma(x: Op.Output, name: String = "Digamma"): Op.Output =
    Op.Builder(opType = "Digamma", name = name)
        .addInput(x)
        .build().outputs(0)

  def erf(x: Op.Output, name: String = "Erf"): Op.Output =
    Op.Builder(opType = "Erf", name = name)
        .addInput(x)
        .build().outputs(0)

  def complementaryErf(x: Op.Output, name: String = "Erfc"): Op.Output =
    Op.Builder(opType = "Erfc", name = name)
        .addInput(x)
        .build().outputs(0)

  def sigmoid(x: Op.Output, name: String = "Sigmoid"): Op.Output =
    Op.Builder(opType = "Sigmoid", name = name)
        .addInput(x)
        .build().outputs(0)

  def sin(x: Op.Output, name: String = "Sin"): Op.Output =
    Op.Builder(opType = "Sin", name = name)
        .addInput(x)
        .build().outputs(0)

  def cos(x: Op.Output, name: String = "Cos"): Op.Output =
    Op.Builder(opType = "Cos", name = name)
        .addInput(x)
        .build().outputs(0)

  def tan(x: Op.Output, name: String = "Tan"): Op.Output =
    Op.Builder(opType = "Tan", name = name)
        .addInput(x)
        .build().outputs(0)

  def asin(x: Op.Output, name: String = "Asin"): Op.Output =
    Op.Builder(opType = "Asin", name = name)
        .addInput(x)
        .build().outputs(0)

  def acos(x: Op.Output, name: String = "Acos"): Op.Output =
    Op.Builder(opType = "Acos", name = name)
        .addInput(x)
        .build().outputs(0)

  def atan(x: Op.Output, name: String = "Atan"): Op.Output =
    Op.Builder(opType = "Atan", name = name)
        .addInput(x)
        .build().outputs(0)

  def isNaN(x: Op.Output, name: String = "IsNan"): Op.Output =
    Op.Builder(opType = "IsNan", name = name)
        .addInput(x)
        .build().outputs(0)

  def isInf(x: Op.Output, name: String = "IsInf"): Op.Output =
    Op.Builder(opType = "IsInf", name = name)
        .addInput(x)
        .build().outputs(0)

  def isFinite(x: Op.Output, name: String = "IsFinite"): Op.Output =
    Op.Builder(opType = "IsFinite", name = name)
        .addInput(x)
        .build().outputs(0)

  def sign(x: Op.Output, name: String = "Sign"): Op.Output =
    Op.Builder(opType = "Sign", name = name)
        .addInput(x)
        .build().outputs(0)

  def floor(x: Op.Output, name: String = "Floor"): Op.Output =
    Op.Builder(opType = "Floor", name = name)
        .addInput(x)
        .build().outputs(0)

  def ceil(x: Op.Output, name: String = "Ceil"): Op.Output =
    Op.Builder(opType = "Ceil", name = name)
        .addInput(x)
        .build().outputs(0)

  def roundInt(x: Op.Output, name: String = "Rint"): Op.Output =
    Op.Builder(opType = "Rint", name = name)
        .addInput(x)
        .build().outputs(0)

  //endregion Unary Ops

  //region Binary Ops

  def add(x: Op.Output, y: Op.Output, name: String = "Add"): Op.Output =
    Op.Builder(opType = "Add", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def subtract(x: Op.Output, y: Op.Output, name: String = "Sub"): Op.Output =
    Op.Builder(opType = "Sub", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def multiply(x: Op.Output, y: Op.Output, name: String = "Mul"): Op.Output =
    Op.Builder(opType = "Mul", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def divide(x: Op.Output, y: Op.Output, name: String = "Div"): Op.Output =
    Op.Builder(opType = "Div", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def floorDivide(x: Op.Output, y: Op.Output, name: String = "FloorDiv"): Op.Output =
    Op.Builder(opType = "FloorDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def truncateDivide(x: Op.Output, y: Op.Output, name: String = "TruncateDiv"): Op.Output =
    Op.Builder(opType = "TruncateDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def realDivide(x: Op.Output, y: Op.Output, name: String = "RealDiv"): Op.Output =
    Op.Builder(opType = "RealDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def squaredDifference(x: Op.Output, y: Op.Output, name: String = "SquaredDifference"): Op.Output =
    Op.Builder(opType = "SquaredDifference", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def maximum(x: Op.Output, y: Op.Output, name: String = "Maximum"): Op.Output =
    Op.Builder(opType = "Maximum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def minimum(x: Op.Output, y: Op.Output, name: String = "Minimum"): Op.Output =
    Op.Builder(opType = "Minimum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def mod(x: Op.Output, y: Op.Output, name: String = "Mod"): Op.Output =
    Op.Builder(opType = "Mod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def floorMod(x: Op.Output, y: Op.Output, name: String = "FloorMod"): Op.Output =
    Op.Builder(opType = "FloorMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def truncateMod(x: Op.Output, y: Op.Output, name: String = "TruncateMod"): Op.Output =
    Op.Builder(opType = "TruncateMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def pow(x: Op.Output, y: Op.Output, name: String = "Pow"): Op.Output =
    Op.Builder(opType = "Pow", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def igammac(a: Op.Output, x: Op.Output, name: String = "Igammac"): Op.Output =
    Op.Builder(opType = "Igammac", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def igamma(a: Op.Output, x: Op.Output, name: String = "Igamma"): Op.Output =
    Op.Builder(opType = "Igamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def zeta(x: Op.Output, q: Op.Output, name: String = "Zeta"): Op.Output =
    Op.Builder(opType = "Zeta", name = name)
        .addInput(x)
        .addInput(q)
        .build().outputs(0)

  def polygamma(a: Op.Output, x: Op.Output, name: String = "Polygamma"): Op.Output =
    Op.Builder(opType = "Polygamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  //endregion Binary Ops

  def betainc(a: Op.Output, b: Op.Output, x: Op.Output, name: String = "Betainc"): Op.Output =
    Op.Builder(opType = "Betainc", name = name)
        .addInput(a)
        .addInput(b)
        .addInput(x)
        .build().outputs(0)

  //region Segment Ops

  /** Creates an op that computes the sum along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \sum_{j...} data(j,...)` where the sum is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentSum`, `segmentIndices` need be sorted.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`). Values should be sorted and
    *                        can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentSum(data: Op.Output, segmentIndices: Op.Output, name: String = "SegmentSum"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    Op.Builder(opType = "SegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** Creates an op that computes the mean along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \frac{sum_{j...} data(j,...)}{N}` where the sum is over all `j`
    * such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike
    * `unsortedSegmentMean`, `segmentIndices` need be sorted.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`). Values should be sorted and
    *                        can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMean(data: Op.Output, segmentIndices: Op.Output, name: String = "SegmentMean"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    Op.Builder(opType = "SegmentMean", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** Creates an op that computes the product along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \prod_{j...} data(j,...)` where the product is over all `j` such
    * that `segmentIndices(j) == i`. Unlike `unsortedSegmentProd`, `segmentIndices` need be sorted.
    *
    * If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`). Values should be sorted and
    *                        can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentProd(data: Op.Output, segmentIndices: Op.Output, name: String = "SegmentProd"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    Op.Builder(opType = "SegmentProd", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** Creates an op that computes the min along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \min_{j...} data(j,...)` where the min is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentMin`, `segmentIndices` need be sorted.
    *
    * If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`). Values should be sorted and
    *                        can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMin(data: Op.Output, segmentIndices: Op.Output, name: String = "SegmentMin"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    Op.Builder(opType = "SegmentMin", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** Creates an op that computes the max along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \max_{j...} data(j,...)` where the max is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentMax`, `segmentIndices` need be sorted.
    *
    * If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`). Values should be sorted and
    *                        can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMax(data: Op.Output, segmentIndices: Op.Output, name: String = "SegmentMax"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    Op.Builder(opType = "SegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** Creates an op that computes the sum along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \sum_{j...} data(j...)` where the sum is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `segmentSum`, `segmentIndices` need not be sorted and need not cover all values
    * in the full range of valid values.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * `segmentsNumber` should equal the number of distinct segment indices.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`).
    * @param  segmentsNumber Number of segments (must have data type of `int32`).
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def unsortedSegmentSum(
      data: Op.Output, segmentIndices: Op.Output, segmentsNumber: Op.Output,
      name: String = "UnsortedSegmentSum"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    if (segmentsNumber.dataType != DataType.int32)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentsNumber.dataType}', is not 'int32', as required.")
    Op.Builder(opType = "UnsortedSegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  /** Creates an op that computes the max along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \max_{j...} data(j...)` where the max is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `segmentMax`, `segmentIndices` need not be sorted and need not cover all values
    * in the full range of valid values.
    *
    * If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * `segmentsNumber` should equal the number of distinct segment indices.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `int32` or `int64`).
    * @param  segmentsNumber Number of segments (must have data type of `int32`).
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def unsortedSegmentProd(
      data: Op.Output, segmentIndices: Op.Output, segmentsNumber: Op.Output,
      name: String = "UnsortedSegmentMax"): Op.Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != DataType.int32 && segmentIndices.dataType != DataType.int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    if (segmentsNumber.dataType != DataType.int32)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentsNumber.dataType}', is not 'int32', as required.")
    Op.Builder(opType = "UnsortedSegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  // TODO: Add sparse segment ops.

  //endregion Segment Ops
}
