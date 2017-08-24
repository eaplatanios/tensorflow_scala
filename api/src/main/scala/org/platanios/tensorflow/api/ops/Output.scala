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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Indexer, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Op.{createWith, getGraphFromInputs}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64}
import org.platanios.tensorflow.api.utilities.using
import org.platanios.tensorflow.jni.{Op => NativeOp}

import spire.math.UShort

/** Helper trait for tagging output convertible objects so that implicit conversions to op outputs can be used.
  *
  * @author Emmanouil Antonios Platanios
  */
trait OutputConvertible {
  // TODO: !!! [TYPE-TRAIT] Convert this to a type trait.

  /** Returns the [[Output]] that this [[OutputConvertible]] object represents. */
  def toOutput: Output
}

object OutputConvertible {
  implicit def outputConvertibleToOutput(value: OutputConvertible): Output = value.toOutput
}

/** Trait representing outputs of an [[Op]]'s computation.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait OutputLike extends OutputConvertible {
  /** Graph where the op belongs. */
  def graph: Graph

  /** Name of this op output. */
  def name: String

  /** Data type of this op output. */
  def dataType: DataType

  /** Device on which this op output will be placed. */
  def device: String

  /** Op that generates this output. */
  def op: Op

  /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
  def consumers: Array[Input]

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices
}

/** Type trait for defining functions operating on and returning op outputs. */
private[ops] trait OutputOps[T] {
  /** Applies a unary op function to the provided output and returns the result.
    *
    * @param  outputLike Output-like object to apply the unary op function on.
    * @param  opFunction Unary op function to apply.
    * @return Resulting output-like object that matches the type of `outputLike`.
    */
  @inline
  def unaryOp(outputLike: T, opFunction: Output => Output): T
}

/** Companion object that defines supported [[OutputOps]] implicit values. */
private[ops] object OutputOps {
  implicit val outputOps = new OutputOps[Output] {
    @inline
    override def unaryOp(outputLike: Output, opFunction: (Output) => Output): Output = opFunction(outputLike)
  }

  implicit val outputIndexedSlicesOps = new OutputOps[OutputIndexedSlices] {
    @inline
    override def unaryOp(outputLike: OutputIndexedSlices, opFunction: Output => Output): OutputIndexedSlices = {
      outputLike.copy(values = opFunction(outputLike.values))
    }
  }

  implicit val sparseOutputOps = new OutputOps[SparseOutput] {
    @inline
    override def unaryOp(outputLike: SparseOutput, opFunction: Output => Output): SparseOutput = {
      outputLike.copy(values = opFunction(outputLike.values))
    }
  }

  implicit val outputLikeOps = new OutputOps[OutputLike] {
    @inline
    override def unaryOp(outputLike: OutputLike, opFunction: (Output) => Output): OutputLike = {
      outputLike match {
        case o: Output => opFunction(o)
        case o: OutputIndexedSlices => o.copy(values = opFunction(o.values))
        case o: SparseOutput => o.copy(values = opFunction(o.values))
      }
    }
  }
}

/** Representation of one of the outputs of an [[Op]]'s computation.
  *
  * An `Output` is a symbolic handle to one of the outputs of an [[Op]]. It does not hold the values of that op's
  * output, but instead provides a means of computing those values in a TensorFlow [[Session]].
  *
  * This class has two primary purposes:
  *
  *   1. An [[Output]] can be passed as input to another [[Op]]. This builds a dataflow connection between ops, which
  *      enables TensorFlow to execute an entire [[Graph]] that represents a large, multi-step computation.
  *   2. After the graph has been launched in a [[Session]], the value of an [[Output]] can be computed by passing
  *      it to [[Session.run]].
  *   3. `Output.evaluate` can also be used to compute the value of an [[Output]] If no session is provided,
  * then the default session is used.
  *
  * In the following example, `c`, `d`, and `e` are symbolic [[Output]] objects, whereas `result` is a Scala array
  * that stores a concrete value:
  * {{{
  *   val c = constant(Array(Array(1.0, 2.0), Array(3.0, 4.0)))
  *   val d = constant(Array(Array(1.0, 1.0), Array(0.0, 1.0)))
  *   val e = matmul(c, d)
  *   val result = e.evaluate() // 'result' now holds the result of the matrix multiplication.
  * }}}
  *
  * @param  op    Op whose output this class represents.
  * @param  index Output index.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class Output private(op: Op, index: Int) extends OutputLike {
  /** Graph where the op belongs. */
  override def graph: Graph = op.graph

  /** Name of this op output. This is simply set to `"<op.name>:<index>"`. */
  override def name: String = s"${op.name}:$index"

  /** Data type of this op output. */
  override def dataType: DataType = using(graph.reference) { r =>
    DataType.fromCValue(NativeOp.outputDataType(r.nativeHandle, op.nativeHandle, index))
  }

  /** Device on which this op output will be placed. */
  override def device: String = op.device

  /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input] = using(graph.reference) { _ =>
    val array = NativeOp.consumers(op.nativeHandle, index)
    if (array == null) {
      Array.empty[Input]
    } else {
      array.map(jniOutput => {
        val op = graph.opsCache.getOrElseUpdate(jniOutput.opHandle, Op(graph, jniOutput.opHandle))
        Input(op = op, index = index)
      })
    }
  }

  /** Shape of the tensor that this op output represents. */
  def shape: Shape = {
    val s = using(op.graph.reference)(r => NativeOp.shape(r.nativeHandle, op.nativeHandle, index))
    if (s == null) Shape.unknown() else Shape.fromSeq(s.map(_.toInt))
  }

  /** Rank of the tensor that this op output represents. */
  def rank: Int = shape.rank

  /** Sets the shape of this op output to the provided shape.
    *
    * This method can be useful in cases when shape inference fails, but the shape of the op output is known by the
    * user of the library.
    *
    * @param  shape Shape to use.
    */
  def setShape(shape: Shape): Unit = using(op.graph.reference) { r =>
    NativeOp.setShape(r.nativeHandle, op.nativeHandle, index, shape.asArray.map(_.toLong), shape.rank)
  }

  /** Evaluates this op output.
    *
    * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
    * op output.
    *
    * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
    * the evaluation.
    *
    * @param  feeds   Tensors to feed into the session for this evaluation.
    * @param  session Optional session to use for the evaluation.
    * @return Value of this op output, for this evaluation.
    */
  def evaluate(feeds: Map[Output, Tensor] = Map.empty, session: Session = null): Tensor = {
    val effectiveSession = if (session == null) graph.defaultSession else session
    effectiveSession.run(feeds, this)
  }

  //region Slicing

  // TODO: Maybe add support for a name argument for the constructed op?
  /** Creates an op that slices this op according to the provided indexers.
    *
    * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
    *
    * @param  indexers Sequence of indexers to use.
    * @return Created op.
    */
  def slice(indexers: Indexer*): Output = Indexer.toStridedSlice(indexers: _*)(this)

  //endregion Slicing

  //region Ops

  // TODO: [OPS] Make all ops separate "enriching" classes (e.g., BasicOps, MathOps, etc.).

  private[this] def binaryOperatorHelper(other: Output, operator: (Output, Output) => Output): Output = {
    if (this.dataType.priority >= other.dataType.priority)
      operator(this, Math.cast(other, this.dataType))
    else
      operator(Math.cast(this, other.dataType), other)
  }

  private[this] def divHelper(x: Output, y: Output): Output = {
    if (x.dataType.isFloatingPoint || x.dataType.isComplex)
      Math.divide(x, y)
    else
      Math.truncateDivide(x, y)
  }

  def unary_- : Output = Math.negate(x = this)
  def +(other: Output): Output = binaryOperatorHelper(other, Math.add(_, _))
  def -(other: Output): Output = binaryOperatorHelper(other, Math.subtract(_, _))
  def *(other: Output): Output = binaryOperatorHelper(other, Math.multiply(_, _)) // TODO: [SPARSE]
  def /(other: Output): Output = binaryOperatorHelper(other, divHelper)           // TODO: [SPARSE]
  def %(other: Output): Output = binaryOperatorHelper(other, Math.floorMod(_, _))
  def **(other: Output): Output = binaryOperatorHelper(other, Math.pow(_, _))

  def unary_! : Output = Math.logicalNot(x = this)
  def &&(other: Output): Output = Math.logicalAnd(x = this, y = other)
  def ||(other: Output): Output = Math.logicalOr(x = this, y = other)

  // def ===(other: Output): Output = Math.equal(x = this, y = other)
  // def =!=(other: Output): Output = Math.notEqual(x = this, y = other)
  def <(other: Output): Output = Math.less(x = this, y = other)
  def <=(other: Output): Output = Math.lessEqual(x = this, y = other)
  def >(other: Output): Output = Math.greater(x = this, y = other)
  def >=(other: Output): Output = Math.greaterEqual(x = this, y = other)

  //endregion Ops

  /** Returns the [[Output]] that this [[OutputLike]] object represents. */
  override def toOutput: Output = this

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices = {
    val denseShape = Basic.shape(this, dataType = INT32, optimize = optimize)
    val indices = Math.range(Basic.constant(0), denseShape(0))
    OutputIndexedSlices(indices = indices, values = this, denseShape = denseShape)
  }

  /** Creates an op that slices this op according to the provided indexers.
    *
    * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
    *
    * @param  indexers Sequence of indexers to use.
    * @return Created op.
    */
  def apply(indexers: Indexer*): Output = slice(indexers: _*)

  override def toString: String = {
    if (device != "")
      s"Output(name = $name, shape = $shape, dataType = $dataType, device = $device)"
    else
      s"Output(name = $name, shape = $shape, dataType = $dataType)"
  }

  override def equals(that: Any): Boolean = that match {
    case that: Output => this.op == that.op && this.index == that.index
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    result = prime * result + op.hashCode
    result = prime * result + index
    result
  }
}

object Output {
  private[ops] trait API extends Implicits {
    type OutputLike = ops.OutputLike
    type Output = ops.Output
    type OutputIndexedSlices = ops.OutputIndexedSlices
    type SparseOutput = ops.SparseOutput
  }

  private[ops] trait Implicits {
    implicit def scalaValueToOutput(value: Boolean): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: String): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Float): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Double): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Byte): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Short): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Int): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: Long): Output = Basic.constant(value)
    implicit def scalaValueToOutput(value: UShort): Output = Basic.constant(value)

    implicit def scalaArrayToOutput(value: Array[Boolean]): Output = Basic.constant(value)
    // implicit def scalaArrayToOutput(value: Array[String]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Float]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Double]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Byte]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Short]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Int]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[Long]): Output = Basic.constant(value)
    implicit def scalaArrayToOutput(value: Array[UShort]): Output = Basic.constant(value)
  }

  // TODO: !!!

  private[api] def constantValue(tensor: Output): Option[Tensor] = {
    val value = tensor.op.opType match {
      case "Const" => ??? // TODO: !!! Needs MakeNdArray()
      case "Shape" =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.isFullyDefined)
          Some(Tensor(tensor.dataType, inputShape.asArray.map(Tensor(_))))
        None
      case "Size" =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.isFullyDefined)
          Some(Tensor(INT32, Tensor(inputShape.asArray.product)))
        None
      case "Rank" =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.numElements != -1)
          Some(Tensor(INT32, Tensor(inputShape.numElements)))
        None
      case "Range" =>
        constantValue(tensor.op.inputs(0))
            .flatMap(start => constantValue(tensor.op.inputs(1))
                .flatMap(limit => constantValue(tensor.op.inputs(2))
                    .flatMap(delta => ???))) // TODO: !!! Create tensor range?
      case "Cast" =>
        constantValue(tensor.op.inputs(0)).map(preCast => ???) // TODO: !!! Get data type attribute from op.
      case "Concat" =>
        constantValue(tensor.op.inputs(0)).flatMap(axis => {
          val values = tensor.op.inputs.tail.flatMap(constantValue)
          if (values.contains(None)) {
            None
          } else {
            ??? // TODO: !!! Concatenate tensors.
          }
        })
      case "ConcatV2" =>
        constantValue(tensor.op.inputs(tensor.op.numInputs - 1)).flatMap(axis => {
          val values = tensor.op.inputs.dropRight(1).flatMap(constantValue)
          if (values.contains(None)) {
            None
          } else {
            ??? // TODO: !!! Concatenate tensors.
          }
        })
      case "Pack" =>
        val values = tensor.op.inputs.map(constantValue)
        if (values.contains(None)) {
          None
        } else {
          ??? // TODO: !!! Concatenate tensors.
        }
      case "Fill" =>
        val fillShape = tensor.shape
        val fillValue = constantValue(tensor.op.inputs(0))
        if (fillShape.isFullyDefined && fillValue.isDefined) {
          val value = fillValue.get
          Some(Tensor.fill(value.dataType, fillShape)(value.scalar)(value.dataType.supportedType))
        } else {
          None
        }
      case _ => None
    }
    // If defined, the caller may now depend on the constant value, and so we prevent 'tensor' from being fed.
    if (value.isDefined)
      tensor.graph.preventFeeding(tensor)
    value
  }

  /** Version of [[constantValue]] that returns a [[Shape]].
    *
    * This version should be used when a constant tensor value is interpreted as a (possibly partial) shape (e.g., in
    * the shape function for `reshape`). By explicitly requesting a [[Shape]] as the return value, it is possible to
    * represent unknown dimensions. In contrast, [[constantValue]] is all-or-nothing.
    *
    * @param  tensor One-dimensional tensor to be evaluated.
    * @return [[Shape]] based on the constant value of `tensor`.
    */
  private[api] def constantValueAsShape(tensor: Output): Option[Shape] = {
    // TODO: !!! Do we really need this function?
    val shape = tensor.shape.withRank(1)
    if (shape == Shape(0)) {
      Some(Shape.scalar())
    } else {
      tensor.op.opType match {
        case "Shape" => Some(tensor.op.inputs(0).shape)
        case "Pack" =>
          // 'i' must be a scalar. Attempt to evaluate it.
          val values = tensor.op.inputs.map(i => constantValue(i).map(v => Shape(v.scalar.asInstanceOf[Int])))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case "Concat" =>
          // We assume that 'tensor.op.inputs(0)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          // 'i' must be a vector. Attempt to evaluate it as a shape.
          val values = tensor.op.inputs.tail.map(i => constantValueAsShape(i))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case "ConcatV2" =>
          // We assume that 'tensor.op.inputs(-1)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          // 'i' must be a vector. Attempt to evaluate it as a shape.
          val values = tensor.op.inputs.dropRight(1).map(i => constantValueAsShape(i))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case _ =>
          var returnShape = Shape.unknown(shape(0))
          val valueOption = constantValue(tensor)
          if (valueOption.isDefined) {
            val value = valueOption.get
            require(value.rank == 1, "Only rank-1 tensors can be converted to shapes.")
            val shape = Shape(
              (0 until value.numElements).map(value.getElementAtFlattenedIndex(_).asInstanceOf[Int]): _*)
            returnShape = returnShape.mergeWith(shape)
          }
          Some(returnShape)
      }
    }
  }

  /** Convenient implicit conversion function used to convert op outputs to their corresponding ops for use with the
    * [[Op.createWith]] function, when specifying control dependencies.
    *
    * @param  output Op output.
    * @return Op corresponding to the provided op output.
    */
  implicit def outputToOpImplicitConversion(output: Output): Op = output.op
}

/** Sparse representation of one of the outputs of an `Op`'s computation. of a set of tensor slices at given indices.
  *
  * This class if a simple wrapper for a pair (or a set of three) of [[Output]] objects:
  *   - `indices`: A one-dimensional integer [[Output]] with shape `[D0]`.
  *   - `values`: An [[Output]] of any data type, with shape `[D0, D1, ..., Dn]`.
  *   - `denseShape`: Optionally, an integer [[Output]] with shape `[LARGE0, D1, ..., Dn]`.
  *
  * An [[OutputIndexedSlices]] is typically used to represent a subset of a larger [[Output]], `dense`, of shape
  * `[LARGE0, D1, ..., Dn]`, where `LARGE0 >> D0`. The values in `indices` are the indices in the first dimension of
  * the slices that have been extracted from the larger tensor.
  *
  * The dense [[Output]], `dense`, represented by [[OutputIndexedSlices]], `slices`, has:
  * {{{
  *   dense(slices.indices(i), ::, ::, ...) = slices.values(i, ::, ::, ...)
  * }}}
  *
  * The [[OutputIndexedSlices]] class is used primarily in the definition of gradients for operations that have
  * sparse gradients, such as `gather`.
  *
  * Note that this is different than [[SparseOutput]] which uses multi-dimensional indices and scalar values.
  *
  * @param  indices    Indices along the first dimension of the corresponding dense [[Output]].
  * @param  values     Values corresponding to the provided indices.
  * @param  denseShape Shape of the corresponding dense [[Output]].
  *
  * @author Emmanouil Antonios Platanios
  */
final case class OutputIndexedSlices private(indices: Output, values: Output, denseShape: Output = null)
    extends OutputLike {
  /** Graph that contains `values`, `indices`, and `denseShape`. */
  override def graph: Graph = getGraphFromInputs(Set(values, indices, denseShape))

  /** Name of this op output indexed slices. */
  override def name: String = s"${values.name}[${indices.name}]" +
      (if (denseShape != null) s"(shape = ${denseShape.name})" else "")

  /** Data type of this op output indexed slices. */
  override def dataType: DataType = values.dataType

  /** Device on which these op output indexed slices will be placed. */
  override def device: String = values.device

  /** Op that outputs these indexed slices. */
  override def op: Op = values.op

  /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input] = values.consumers

  /** Returns the [[Output]] that this [[OutputLike]] object represents. */
  override def toOutput: Output = {
    if (denseShape != null)
      throw new IllegalStateException(
        s"Op output conversion requested the conversion of 'OutputIndexedSlices', '$this', which has no dense " +
            s"shape information available.")
    // TODO: Add check for large number of elements (e.g., > 100000000).
    createWith(nameScope = "IndexedSlicesToOutput") {
      Math.unsortedSegmentSum(data = values, segmentIndices = indices, segmentsNumber = denseShape(0))
    }
  }

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices = this

  override def toString: String = {
    s"OutputIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
        s"device = $device)}"
  }
}

/** Represents a sparse op output.
  *
  * TensorFlow represents a sparse tensor as three separate dense tensors: `indices`, `values`, and `denseShape`. In
  * Scala, the three tensors are collected into a `SparseTensor` class for ease of use.  If you have separate
  * `indices`, `values`, and `denseShape` tensors, wrap them in a `SparseTensor` object before passing to the
  * relevant sparse tensor manipulation
  *
  * Concretely, the sparse tensor `SparseTensor(indices, values, denseShape)` comprises the following components,
  * where `N` and `rank` are the number of values and number of dimensions in the `SparseTensor`, respectively:
  *
  *   - `indices`: Two-dimensional `Int64` tensor with shape `[N, rank]`, which specifies the indices of the elements
  * in the sparse tensor that have nonzero values (elements are zero-indexed). For example,
  * `indices = [[1, 3], [2, 4]]` specifies that the elements with indexes `[1, 3]` and `[2, 4]` have nonzero
  * values.
  *   - `values`: One-dimensional tensor of any type, with shape `[N]`, which supplies the values for each element in
  * `indices`. For example, given `indices = [[1, 3], [2, 4]]`, the parameter `values = [18, 3.6]` specifies that
  * element `[1, 3]` of the sparse tensor has a value of `18`, and element `[2, 4]` of the tensor has a value of
  * `3.6`.
  *   - `denseShape`: One-dimensional `Int64` tensor with shape `[rank]`, which specifies the dense shape of the
  * sparse tensor.  For example, `denseShape = [3, 6]` specifies a two-dimensional 3x6 tensor,
  * `denseShape = [2, 3, 4]` specifies a three-dimensional 2x3x4 tensor, and `denseShape = [9]` specifies a
  * one-dimensional tensor with 9 elements.
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
  * {{{
  *   // The sparse tensor:
  *   SparseTensor(indices = Tensor(Tensor(0, 0), Tensor(1, 2)), values = Tensor(1, 2), denseShape = Shape(3, 4))
  *   // represents the dense tensor:
  *   //
  * }}}
  *
  * @param  indices    Two-dimensional `Int64` tensor with shape `[N, rank]`.
  * @param  values     One-dimensional tensor with shape `[N]`.
  * @param  denseShape One-dimensional `Int64` tensor with shape `[rank]`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class SparseOutput(indices: Output, values: Output, denseShape: Output) extends OutputLike {
  // TODO: Add constructor from scala arrays?
  if (indices.dataType != INT64)
    throw InvalidDataTypeException(
      s"Indices cannot have '${indices.dataType}' data type. They have to be 'TFInt64'.")
  if (denseShape.dataType != INT64)
    throw InvalidDataTypeException(
      s"Dense shape cannot have '${denseShape.dataType}' data type. It has to be 'TFInt64'.")
  // TODO: Add a "subShape" method?
  Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
  Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

  /** Graph that contains `values`, `indices`, and `denseShape`. */
  override def graph: Graph = getGraphFromInputs(Set(values, indices, denseShape))

  /** Name of this sparse op output. */
  override def name: String = s"${values.name}[${indices.name}]" +
      (if (denseShape != null) s"(shape = ${denseShape.name})" else "")

  /** Data type of this sparse op output. */
  override def dataType: DataType = values.dataType

  /** Device on which this sparse op output will be placed. */
  override def device: String = values.device

  /** Op that outputs this sparse tensor. */
  override def op: Op = values.op

  /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input] = values.consumers

  /** Gets the [[Shape]] corresponding to the shape of the dense tensor that this sparse tensor represents.
    *
    * @return Dense tensor shape.
    */
  def shape: Shape = Output.constantValueAsShape(denseShape).get

  /** Evaluates this sparse op output.
    *
    * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
    * op output.
    *
    * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
    * the evaluation.
    *
    * @param  feeds   Tensors to feed into the session for this evaluation.
    * @param  session Optional session to use for the evaluation.
    * @return Value of this sparse op output, for this evaluation, represented as tuple containing the indices, the
    *         values, and the dense shape.
    */
  def value(feeds: Map[Output, Tensor] = Map.empty, session: Session = null): (Tensor, Tensor, Tensor) = {
    val effectiveSession = if (session == null) graph.defaultSession else session
    effectiveSession.run(feeds, this)
  }

  override def toOutput: Output = {
    throw new UnsupportedOperationException(s"Cannot convert sparse output '$this' to a dense output.")
  }

  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices = {
    throw new UnsupportedOperationException(s"Cannot convert sparse output '$this' to output indexed slices.")
  }

  override def toString: String = {
    s"OutputIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
        s"device = $device)}"
  }
}

object SparseOutput {
  /** Converts the provided sparse output value to a sparse op output.
    *
    * @param  sparseOutputValue Sparse output value represented as tuple containing the indices, the values, and the
    *                           dense shape.
    * @return Sparse op output.
    */
  private[api] def convertToSparseOutput(sparseOutputValue: (Tensor, Tensor, Tensor)): SparseOutput = {
    SparseOutput(
      Basic.constant(sparseOutputValue._1),
      Basic.constant(sparseOutputValue._2),
      Basic.constant(sparseOutputValue._3))
  }
}
