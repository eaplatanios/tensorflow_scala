package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.{DataType, SupportedType, Tensor}
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.SupportedType

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  /** Creates an op that selects elements from `x` or `y`, depending on `condition`.
    *
    * The `x`, and `y` tensors must have the same shape. The output tensor will also have the same shape.
    *
    * The `condition` tensor must be a scalar if `x` and `y` are scalars. If `x` and `y` are vectors or higher rank,
    * then `condition` must be either a scalar, or a vector with size matching the first dimension of `x`, or it must
    * have the same shape as `x`.
    *
    * The `condition` tensor acts as a mask that chooses, based on the value at each element, whether the corresponding
    * element / row in the output should be taken from `x` (if true) or `y` (if false).
    *
    * If `condition` is a vector and `x` and `y` are higher rank matrices, then it chooses which row (outer dimension)
    * to copy from `x` and `y`. If `condition` has the same shape as `x` and `y`, then it chooses which element to copy
    * from `x` and `y`.
    *
    * For example:
    * {{{
    *   // 'condition' tensor is [[true,  false], [false, true]]
    *   // 'x' is [[1, 2], [3, 4]]
    *   // 'y' is [[5, 6], [7, 8]]
    *   select(condition, x, y) == [[1, 6], [7, 4]]
    *
    *   // 'condition' tensor is [true, false]
    *   // 'x' is [[1, 2], [3, 4]]
    *   // 'y' is [[5, 6], [7, 8]]
    *   select(condition, x, y) == [[1, 2], [7, 8]]
    * }}}
    *
    * @param  condition Boolean condition tensor.
    * @param  x         Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may
    *                   have a higher rank, but its first dimension must match the size of `condition`.
    * @param  y         Tensor with the same data type and shape as `t`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def select(condition: Op.Output, x: Op.Output, y: Op.Output, name: String = "Select"): Op.Output = {
    Op.Builder(opType = "Select", name = name)
        .addInput(condition)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that constructs a sequence of numbers.
    *
    * The op creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not
    * including `limit`. The data type of the resulting tensor is inferred from the inputs unless it is provided
    * explicitly.
    *
    * For example:
    * {{{
    *   // 'start' is 3
    *   // 'limit' is 18
    *   // 'delta' is 3
    *   range(start, limit, delta) == [3, 6, 9, 12, 15]
    *
    *   // 'start' is 3
    *   // 'limit' is 1
    *   // 'delta' is -0.5
    *   range(start, limit, delta) == [3.0, 2.5, 2.0, 1.5]
    * }}}
    *
    * @param  start    Start of the number sequence.
    * @param  limit    End (exclusive) of the number sequence.
    * @param  delta    Difference between consecutive numbers in the sequence.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def range(
      start: Op.Output, limit: Op.Output, delta: Op.Output = Basic.constant(1), dataType: DataType = null,
      name: String = "Range"): Op.Output = {
    var castedStart: Op.Output = null
    var castedLimit: Op.Output = null
    var castedDelta: Op.Output = null
    Op.createWith(nameScope = name) {
      val supportedDataTypes = Set[DataType](DataType.Int32, DataType.Int64, DataType.Float32, DataType.Float64)
      require(supportedDataTypes.contains(start.dataType), s"Unsupported data type '${start.dataType}'.")
      require(supportedDataTypes.contains(limit.dataType), s"Unsupported data type '${limit.dataType}'.")
      require(supportedDataTypes.contains(delta.dataType), s"Unsupported data type '${delta.dataType}'.")
      val inferredDataType = {
        if (dataType != null)
          dataType
        else
          Set(start.dataType, limit.dataType, delta.dataType).maxBy(_.priority)
      }
      if (start.dataType != inferredDataType)
        castedStart = cast(start, inferredDataType)
      if (limit.dataType != inferredDataType)
        castedLimit = cast(limit, inferredDataType)
      if (delta.dataType != inferredDataType)
        castedDelta = cast(delta, inferredDataType)
    }
    Op.Builder(opType = "Range", name = name)
        .addInput(castedStart)
        .addInput(castedLimit)
        .addInput(castedDelta)
        .build().outputs(0)
  }

  // TODO: Add the "linspace" op.

  /** Creates an op that casts a tensor to a new data type.
    *
    * The op casts `x` to the provided data type.
    *
    * For example:
    * {{{
    *   // `a` is a tensor with values [1.8, 2.2], and data type Float32
    *   cast(a, Int32) == [1, 2] // with data type Int32
    * }}}
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def cast(x: Op.Output, dataType: DataType, name: String = "Cast"): Op.Output = {
    Op.Builder(opType = "Cast", name = name)
        .addInput(x)
        .setAttribute("DstT", dataType)
        .build().outputs(0)
  }

  /** Creates an op that casts a sparse tensor to a new data type.
    *
    * The op casts `x.values` to the provided data type.
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sparseCast(x: Op.SparseOutput, dataType: DataType, name: String = "Cast"): Op.SparseOutput = {
    val castedValues = Op.Builder(opType = "Cast", name = name)
        .addInput(x.values)
        .setAttribute("DstT", dataType)
        .build().outputs(0)
    Op.SparseOutput(x.indices, castedValues, x.denseShape)
  }

  @throws[IllegalArgumentException]
  def conjugate(input: Op.Output, name: String = "Conjugate"): Op.Output = {
    if (input.dataType.isComplex) {
      Op.Builder(opType = "Conj", name = name)
          .addInput(input)
          .build().outputs(0)
    } else if (input.dataType.isNumeric) {
      input
    } else {
      throw new IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
    }
  }

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
        .setAttribute("transpose_a", transposeA)
        .setAttribute("transpose_b", transposeB)
        .build().outputs(0)
  }

  def batchMatMul(
      x: Op.Output, y: Op.Output, adjointX: Boolean = false, adjointY: Boolean = false,
      name: String = "BatchMatMul"): Op.Output =
    Op.Builder(opType = "BatchMatMul", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("adj_x", adjointX)
        .setAttribute("adj_y", adjointY)
        .build().outputs(0)

  //region Unary Ops

  def negate(x: Op.Output, name: String = "Negate"): Op.Output = {
    Op.Builder(opType = "Neg", name = name)
        .addInput(x)
        .build().outputs(0)
  }

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

  //region Logical Ops

  /** Creates an op that computes the truth value of `!x` element-wise.
    *
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalNot(x: Op.Output, name: String = "LogicalNot"): Op.Output = {
    Op.Builder(opType = "LogicalNot", name = name)
        .addInput(x)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x && y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalAnd(x: Op.Output, y: Op.Output, name: String = "LogicalAnd"): Op.Output = {
    Op.Builder(opType = "LogicalAnd", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x || y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalOr(x: Op.Output, y: Op.Output, name: String = "LogicalOr"): Op.Output = {
    Op.Builder(opType = "LogicalOr", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `(x || y) && !(x && y)` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalXor(x: Op.Output, y: Op.Output, name: String = "LogicalXor"): Op.Output = {
    logicalAnd(logicalOr(x, y), logicalNot(logicalAnd(x, y)), name = name)
  }

  //endregion Logical Ops

  //region Comparison Ops

  /** Creates an op that computes the truth value of `x == y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def equal(x: Op.Output, y: Op.Output, name: String = "Equal"): Op.Output = {
    Op.Builder(opType = "Equal", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x != y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def notEqual(x: Op.Output, y: Op.Output, name: String = "NotEqual"): Op.Output = {
    Op.Builder(opType = "NotEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `abs(x - y) < tolerance`  element-wise.
    *
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def approximatelyEqual(
      x: Op.Output, y: Op.Output, tolerance: Float = 0.00001f, name: String = "ApproximatelyEqual"): Op.Output = {
    Op.Builder(opType = "ApproximateEqual", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("tolerance", tolerance)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x < y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def less(x: Op.Output, y: Op.Output, name: String = "Less"): Op.Output = {
    Op.Builder(opType = "Less", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x <= y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def lessEqual(x: Op.Output, y: Op.Output, name: String = "LessEqual"): Op.Output = {
    Op.Builder(opType = "LessEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x > y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greater(x: Op.Output, y: Op.Output, name: String = "Greater"): Op.Output = {
    Op.Builder(opType = "Greater", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `x >= y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greaterEqual(x: Op.Output, y: Op.Output, name: String = "GreaterEqual"): Op.Output = {
    Op.Builder(opType = "GreaterEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes(tensor: Op.Output, axes: Array[Int]): Op.Output = {
    if (axes != null)
      Basic.constant(Tensor(axes.map(Tensor(_)): _*))
    else
      Basic.constant(Tensor((0 until tensor.shape.rank).map(Tensor(_)): _*))
  }

  /** Creates an op that computes the sum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *   reduceSum(x) == 6
    *   reduceSum(x, 0) == [2, 2, 2]
    *   reduceSum(x, 1) == [3, 3]
    *   reduceSum(x, 1, keepDims = true) == [[3], [3]]
    *   reduceSum(x, [0, 1]) == 6
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceSum(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceSum"): Op.Output = {
    Op.Builder(opType = "Sum", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the mean of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   reduceMean(x) == 1.5
    *   reduceMean(x, 0) == [1.5, 1.5]
    *   reduceMean(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceMean(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceMean"): Op.Output = {
    Op.Builder(opType = "Mean", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the product of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *   reduceProd(x) == 1
    *   reduceProd(x, 0) == [1, 1, 1]
    *   reduceProd(x, 1) == [1, 1]
    *   reduceProd(x, 1, keepDims = true) == [[1], [1]]
    *   reduceProd(x, [0, 1]) == 1
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceProd(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceProd"): Op.Output = {
    Op.Builder(opType = "Prod", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the minimum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   reduceMin(x) == 1.0
    *   reduceMin(x, 0) == [1.0, 1.0]
    *   reduceMin(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceMin(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceMin"): Op.Output = {
    Op.Builder(opType = "Min", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the maximum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   reduceMax(x) == 2.0
    *   reduceMax(x, 0) == [2.0, 2.0]
    *   reduceMax(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceMax(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceMax"): Op.Output = {
    Op.Builder(opType = "Max", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the logical AND of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[true, true], [false, false]]
    *   reduceAll(x) == false
    *   reduceAll(x, 0) == [false, false]
    *   reduceAll(x, 1) == [true, false]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceAll(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceAll"): Op.Output = {
    Op.Builder(opType = "All", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the logical OR of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[true, true], [false, false]]
    *   reduceAll(x) == true
    *   reduceAll(x, 0) == [true, true]
    *   reduceAll(x, 1) == [true, false]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceAny(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "ReduceAny"): Op.Output = {
    Op.Builder(opType = "Any", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the log-sum-exp of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * This function is more numerically stable than `log(sum(exp(input)))`. It avoids overflows caused by computing the
    * exponential of large inputs, and underflows caused by computing the logarithm of small inputs.
    *
    * For example:
    * {{{
    *   // 'x' is [[0, 0, 0], [0, 0, 0]]
    *   reduceLogSumExp(x) == log(6)
    *   reduceLogSumExp(x, 0) == [log(2), log(2), log(2)]
    *   reduceLogSumExp(x, 1) == [log(3), log(3)]
    *   reduceLogSumExp(x, 1, keepDims = true) == [[log(3)], [log(3)]]
    *   reduceLogSumExp(x, [0, 1]) == log(6)
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def reduceLogSumExp(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false,
      name: String = "ReduceLogSumExp"): Op.Output = {
    Op.createWith(nameScope = name) {
      val max = Basic.stopGradient(reduceMax(input, axes, keepDims = true))
      val result = log(reduceSum(exp(input - max), axes, keepDims = true)) + max
      if (keepDims)
        result
      else
        Basic.squeeze(result, axes)
    }
  }

  /** Creates an op that computes the number of non-zero elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * IMPORTANT NOTE: Floating point comparison to zero is done by exact floating point equality check. Small values are
    * **not** rounded to zero for the purposes of the non-zero check.
    *
    * For example:
    * {{{
    *   // 'x' is [[0, 1, 0], [1, 1, 0]]
    *   countNonZero(x) == 3
    *   countNonZero(x, 0) == [1, 2, 0]
    *   countNonZero(x, 1) == [1, 2]
    *   countNonZero(x, 1, keepDims = true) == [[1], [2]]
    *   countNonZero(x, [0, 1]) == 3
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output with `Int64` data type.
    */
  def countNonZero(
      input: Op.Output, axes: Array[Int] = null, keepDims: Boolean = false,
      name: String = "CountNonZero"): Op.Output = {
    Op.createWith(nameScope = name) {
      reduceSum(cast(notEqual(input, Basic.constant(0)), DataType.Int64), axes, keepDims)
    }
  }

  //endregion Reduction Ops

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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    if (segmentsNumber.dataType != DataType.Int32)
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
    if (segmentIndices.dataType != DataType.Int32 && segmentIndices.dataType != DataType.Int64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'int32' or 'int64', as required.")
    if (segmentsNumber.dataType != DataType.Int32)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentsNumber.dataType}', is not 'int32', as required.")
    Op.Builder(opType = "UnsortedSegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  // TODO: [SPARSE] Add sparse segment ops.

  //endregion Segment Ops

  object Gradients {
    GradientsRegistry.register("MatMul", matMulGradient)
    GradientsRegistry.register("BatchMatMul", batchMatMulGradient)

    def matMulGradient(op: Op, outputGradients: Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
      matMulGradientCommon(op, outputGradients, "transpose_a", "transpose_b", isBatch = false)
    }

    def batchMatMulGradient(op: Op, outputGradients: Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
      matMulGradientCommon(op, outputGradients, "adj_x", "adj_y", isBatch = true)
    }

    private[this] def matMulGradientCommon(
        op: Op, outputGradients: Seq[Op.OutputLike], transposeAAttribute: String, transposeBAttribute: String,
        isBatch: Boolean): Seq[Op.OutputLike] = {
      val transposeA = op.booleanAttribute(transposeAAttribute)
      val transposeB = op.booleanAttribute(transposeBAttribute)
      val a = conjugate(op.inputs(0))
      val b = conjugate(op.inputs(1))
      val outputGradient = outputGradients.head.asInstanceOf[Op.OutputConvertible].toOpOutput
      if (!transposeA && !transposeB)
        matMulGradientHelper(
          outputGradient, b, a, outputGradient,
          transposeX0 = false, transposeX1 = true, transposeY0 = true, transposeY1 = false, isBatch = isBatch)
      else if (!transposeA && transposeB)
        matMulGradientHelper(
          outputGradient, b, outputGradient, a,
          transposeX0 = false, transposeX1 = false, transposeY0 = true, transposeY1 = false, isBatch = isBatch)
      else if (transposeA && !transposeB)
        matMulGradientHelper(
          b, outputGradient, a, outputGradient,
          transposeX0 = false, transposeX1 = true, transposeY0 = false, transposeY1 = false, isBatch = isBatch)
      else
        matMulGradientHelper(
          b, outputGradient, outputGradient, a,
          transposeX0 = true, transposeX1 = true, transposeY0 = true, transposeY1 = true, isBatch = isBatch)
    }

    private[this] def matMulGradientHelper(
        x0: Op.Output, x1: Op.Output, y0: Op.Output, y1: Op.Output, transposeX0: Boolean, transposeX1: Boolean,
        transposeY0: Boolean, transposeY1: Boolean, isBatch: Boolean): Seq[Op.OutputLike] = {
      if (!isBatch) {
        val gradientX = matMul(x0, x1, transposeA = transposeX0, transposeB = transposeX1, name = "MatMul_1")
        val gradientY = matMul(y0, y1, transposeA = transposeY0, transposeB = transposeY1, name = "MatMul_2")
        Seq[Op.OutputLike](gradientX, gradientY)
      } else {
        val gradientX = batchMatMul(x0, x1, adjointX = transposeX0, adjointY = transposeX1, name = "MatMul_1")
        val gradientY = batchMatMul(y0, y1, adjointX = transposeY0, adjointY = transposeY1, name = "MatMul_2")
        Seq[Op.OutputLike](gradientX, gradientY)
      }
    }
  }
}
