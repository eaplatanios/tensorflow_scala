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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Indexer, Shape}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.tensors.ops.{Basic => TensorBasic, Math => TensorMath}
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64, IsNumeric}
import org.platanios.tensorflow.api.utilities.using
import org.platanios.tensorflow.jni.{Op => NativeOp}

/** Trait representing outputs of an op's computation.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait OutputLike[+T] extends OutputLikeOrTensorArray[T] {
  /** Graph where the op belongs. */
  def graph: Graph

  /** Name of this op output. */
  def name: String

  /** Data type of this op output. */
  def dataType: DataType[T]

  /** Device on which this op output will be placed. */
  def device: String

  /** Op that generates this output. */
  def op: Op[Seq[Output[Any]], Seq[Output[Any]]]

  /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
  def consumers: Array[Input[Any]]

  /** Returns the [[Output]] that this [[OutputLike]] object represents. */
  def toOutput: Output[T]

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices[T]
}

/** Representation of one of the outputs of an op's computation.
  *
  * An `Output` is a symbolic handle to one of the outputs of an op. It does not hold the values of that op's
  * output, but instead provides a means of computing those values in a TensorFlow `Session`.
  *
  * This class has two primary purposes:
  *
  *   1. An `Output` can be passed as input to another op. This builds a dataflow connection between ops, which
  *      enables TensorFlow to execute an entire graph that represents a large, multi-step computation.
  *   2. After the graph has been launched in a `Session`, the value of an `Output` can be computed by passing
  *      it to `Session.run`.
  *   3. `Output.evaluate` can also be used to compute the value of an [[Output]] If no session is provided,
  *      then the default session is used.
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
final case class Output[+T] private(
    op: Op[Seq[Output[Any]], Seq[Output[Any]]],
    index: Int
) extends OutputLike[T] {
  /** Graph where the op belongs. */
  override def graph: Graph = op.graph

  /** Name of this op output. This is simply set to `"<op.name>:<index>"`. */
  override def name: String = s"${op.name}:$index"

  /** Data type of this op output. */
  override def dataType: DataType[T] = using(graph.reference) { r =>
    DataType.fromCValue(NativeOp.outputDataType(r.nativeHandle, op.nativeHandle, index))
  }

  /** Device on which this op output will be placed. */
  override def device: String = op.device

  /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input[Any]] = using(graph.reference) { _ =>
    val array = NativeOp.consumers(op.nativeHandle, index)
    if (array == null) {
      Array.empty[Input[Any]]
    } else {
      array.map(jniOutput => {
        val op = graph.opsCache.getOrElseUpdate(
          jniOutput.opHandle,
          Op[Seq[Output[Any]], Seq[Output[Any]]](
            graph = graph,
            originalInput = None,
            nativeHandle = jniOutput.opHandle))
        Input(op = op, index = index)
      })
    }
  }

  /** Shape of the tensor that this op output represents. */
  def shape: Shape = {
    val s = using(op.graph.reference)(r => {
      NativeOp.shape(r.nativeHandle, op.nativeHandle, index)
    })
    if (s == null) Shape.unknown() else Shape.fromSeq(s.map(_.toInt))
  }

  /** Rank of the tensor that this op output represents. */
  def rank: Int = {
    shape.rank
  }

  /** Size of the tensor that this op output represents. */
  def size: Long = {
    shape.numElements
  }

  /** Sets the shape of this op output to the provided shape.
    *
    * This method can be useful in cases when shape inference fails, but the shape of the op output is known by the
    * user of the library.
    *
    * @param  shape Shape to use.
    */
  def setShape(shape: Shape): Unit = {
    if (shape.rank != -1) {
      using(op.graph.reference) { r =>
        NativeOp.setShape(
          r.nativeHandle,
          op.nativeHandle,
          index,
          shape.asArray.map(_.toLong),
          shape.rank)
      }
    }
  }

  //region Slicing

  /** Creates an op that slices this op according to the provided indexers.
    *
    * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
    *
    * @param  firstIndexer  First indexer to use.
    * @param  otherIndexers Rest of the indexers to use.
    * @return Created op.
    */
  def apply(
      firstIndexer: Indexer,
      otherIndexers: Indexer*
  ): Output[T] = {
    this.slice(firstIndexer, otherIndexers: _*)
  }

  /** Creates an op that slices this op according to the provided indexers.
    *
    * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
    *
    * @param  firstIndexer  First indexer to use.
    * @param  otherIndexers Rest of the indexers to use.
    * @return Created op.
    */
  def slice(
      firstIndexer: Indexer,
      otherIndexers: Indexer*
  ): Output[T] = {
    val stridedSlice = Indexer.toStridedSlice(firstIndexer, otherIndexers: _*)
    val beginTensor: Output[Int] = stridedSlice._1
    val endTensor: Output[Int] = stridedSlice._2
    val stridesTensor: Output[Int] = stridedSlice._3
    Basic.stridedSlice(
      this, beginTensor, endTensor, stridesTensor, stridedSlice._4,
      stridedSlice._5, stridedSlice._6, stridedSlice._7, stridedSlice._8)
  }

  //endregion Slicing

  /** Returns this output. */
  override def toOutput: Output[T] = {
    this
  }

  /** Returns the output indexed slices object that has the same value as this output.
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return Output indexed slices object that has the same value as this output.
    */
  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices[T] = {
    val denseShape = Basic.shape(this, INT64, optimize = optimize)
    val indices = Math.range(Basic.constant(0L), denseShape(0))
    OutputIndexedSlices(indices = indices, values = this, denseShape = denseShape)
  }

  override def toString: String = {
    if (device != "")
      s"Output(name = $name, shape = $shape, dataType = $dataType, device = $device)"
    else
      s"Output(name = $name, shape = $shape, dataType = $dataType)"
  }

  override def equals(that: Any): Boolean = that match {
    // TODO: !!! [OPS] Find a better way to do this.
    case that: Output[_] => this.op == that.op && this.index == that.index
    case _ => false
  }

  override def hashCode(): Int = {
    // TODO: !!! [OPS] Find a better way to do this.
    val prime = 31
    var result = 1
    result = prime * result + op.hashCode
    result = prime * result + index
    result
  }
}

object Output {
  /** Returns the constant value of the given tensor, if efficiently calculable. */
  private[api] def constantValue[T](output: Output[T]): Option[Tensor[T]] = {
    val value = using(output.graph.reference)(r => {
      val result = NativeOp.tryEvaluateConstant(
        r.nativeHandle, output.op.nativeHandle, output.index)
      if (result == 0L)
        None
      else
        Some(Tensor.fromNativeHandle(result))
    }).orElse(output.op.opType match {
      case "Const" => Option(output.op.tensorAttribute("value")) // TODO: !!! Make more robust.
      case "Shape" =>
        val inputShape = output.op.inputsSeq(0).shape
        if (inputShape.isFullyDefined)
          Some(inputShape.toTensor(output.dataType))
        None
      case "Size" =>
        val inputShape = output.op.inputsSeq(0).shape
        if (inputShape.isFullyDefined)
          Some(Tensor.ofType(INT32, inputShape.asArray.product))
        None
      case "Rank" =>
        val inputShape = output.op.inputsSeq(0).shape
        if (inputShape.numElements != -1)
          Some(Tensor.ofType(INT32, inputShape.numElements.toInt))
        None
      case "Range" =>
        constantValue(output.op.inputsSeq(0))
            .flatMap(start => constantValue(output.op.inputsSeq(1))
                .flatMap(limit => constantValue(output.op.inputsSeq(2))
                    .map(delta => TensorMath.range(
                      start.asInstanceOf[Tensor[Int]],
                      limit.asInstanceOf[Tensor[Int]],
                      delta.asInstanceOf[Tensor[Int]]))))
      case "Cast" =>
        constantValue(output.op.inputsSeq(0)).map(preCast => {
          preCast.cast(output.op.dataTypeAttribute("DstT"))
        })
      case "Concat" =>
        constantValue(output.op.inputsSeq(0)).flatMap(axis => {
          val values = output.op.inputsSeq.tail.map(constantValue)
          if (values.contains(None))
            None
          else
            Some(TensorBasic.concatenate(
              values.map(_.get), axis.asInstanceOf[Tensor[Int]]))
        })
      case "ConcatV2" =>
        constantValue(output.op.inputsSeq(output.op.numInputs - 1)).flatMap(axis => {
          val values = output.op.inputsSeq.dropRight(1).map(constantValue)
          if (values.contains(None))
            None
          else
            Some(TensorBasic.concatenate(
              values.map(_.get), axis.asInstanceOf[Tensor[Int]]))
        })
      case "Pack" =>
        val values = output.op.inputsSeq.map(constantValue)
        if (values.contains(None)) {
          None
        } else {
          Some(TensorBasic.stack(values.map(_.get)))
        }
      case "Fill" =>
        val fillShape = output.shape
        val fillValue = constantValue(output.op.inputsSeq(0))
        if (fillShape.isFullyDefined && fillValue.isDefined) {
          val value = fillValue.get
          Some(Tensor.fill(
            dataType = value.dataType,
            shape = fillShape
          )(value.scalar)(value.dataType.evSupportedType))
        } else {
          None
        }
      // TODO: [OPS] The following two only work for numeric data types.
      //      case "Equal" =>
      //        constantValue(output.op.inputsSeq(0))
      //            .flatMap(value1 => constantValue(output.op.inputsSeq(1))
      //                .map(value2 => TensorMath.equal(value1, value2)))
      //      case "NotEqual" =>
      //        constantValue(output.op.inputsSeq(0))
      //            .flatMap(value1 => constantValue(output.op.inputsSeq(1))
      //                .map(value2 => TensorMath.notEqual(value1, value2)))
      case _ => None
    })
    // If defined, the caller may now depend on the constant value, and so we prevent 'tensor' from being fed.
    if (value.isDefined)
      output.graph.preventFeeding(output)
    value.map(_.asInstanceOf[Tensor[T]])
  }

  /** Version of [[constantValue]] that returns a [[Shape]].
    *
    * This version should be used when a constant tensor value is interpreted as a (possibly partial) shape (e.g., in
    * the shape function for `reshape`). By explicitly requesting a [[Shape]] as the return value, it is possible to
    * represent unknown dimensions. In contrast, [[constantValue]] is all-or-nothing.
    *
    * @param  tensor One-dimensional tensor to be evaluated.
    * @return Shape based on the constant value of `tensor`.
    */
  private[api] def constantValueAsShape[T](tensor: Output[T]): Option[Shape] = {
    if (tensor.rank == 0) {
      // TODO: Handle this better.
      Some(Shape(-1))
    } else if (tensor.shape.withRank(1) == Shape(0)) {
      Some(Shape.scalar())
    } else {
      tensor.op.opType match {
        case "Shape" =>
          val inputShape = tensor.op.inputsSeq(0).shape
          if (inputShape.rank > -1)
            Some(tensor.op.inputsSeq(0).shape)
          else
            None
        case "Pack" =>
          // 'i' must be a scalar. Attempt to evaluate it.
          val values = tensor.op.inputsSeq.map(i => constantValue(i).map(v => Shape(v.scalar.asInstanceOf[Int])))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case "Concat" =>
          // We assume that 'tensor.op.inputs(0)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          // 'i' must be a vector. Attempt to evaluate it as a shape.
          val values = tensor.op.inputsSeq.tail.map(i => constantValueAsShape(i))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case "ConcatV2" =>
          // We assume that 'tensor.op.inputs(-1)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          // 'i' must be a vector. Attempt to evaluate it as a shape.
          val values = tensor.op.inputsSeq.dropRight(1).map(i => constantValueAsShape(i))
          if (values.forall(_.isDefined))
            Some(values.map(_.get).foldLeft(Shape.scalar())((shape, value) => shape.concatenateWith(value)))
          else
            None
        case "StridedSlice" =>
          constantValue(tensor.op.inputsSeq(0))
              .flatMap(begin => constantValue(tensor.op.inputsSeq(1))
                  .flatMap(end => constantValue(tensor.op.inputsSeq(2))
                      .flatMap(strides => {
                        val b = begin(0).scalar.asInstanceOf[Int]
                        val e = end(0).scalar.asInstanceOf[Int]
                        val s = strides(0).scalar.asInstanceOf[Int]
                        val beginMask = tensor.op.longAttribute("begin_mask")
                        val endMask = tensor.op.longAttribute("end_mask")
                        val ellipsisMask = tensor.op.longAttribute("ellipsis_mask")
                        val newAxisMask = tensor.op.longAttribute("new_axis_mask")
                        val shrinkAxisMask = tensor.op.longAttribute("shrink_axis_mask")
                        if (beginMask == 1 ||
                            endMask == 1 ||
                            ellipsisMask == 1 ||
                            newAxisMask == 1 ||
                            shrinkAxisMask == 1 ||
                            (beginMask != 1 && beginMask > 0) ||
                            (endMask != 1 && endMask > 0)) {
                          None
                        } else {
                          val previousShape = constantValueAsShape(tensor.op.inputsSeq(0))
                          previousShape.map(t => Shape(t(b :: s :: e).entriesIterator.map(_.toInt).toArray))
                        }
                      })))
        case _ if tensor.rank == -1 =>
          None
        case _ =>
          var returnShape = Shape.unknown(tensor.shape(0))
          val valueOption = constantValue(tensor)
          if (valueOption.isDefined) {
            val value = valueOption.get.toInt32
            require(value.rank == 1, "Only rank-1 tensors can be converted to shapes.")
            val shape = Shape(
              (0 until value.size.toInt).map(value.getElementAtFlattenedIndex): _*)
            returnShape = returnShape.mergeWith(shape)
          }
          if (returnShape.rank > -1)
            Some(returnShape)
          else
            None
      }
    }
  }
}

/** Sparse representation of a set of tensor slices at given indices.
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
  * @param  indices    Indices along the first dimension of the corresponding dense output.
  * @param  values     Values corresponding to the provided indices.
  * @param  denseShape Shape of the corresponding dense output.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class OutputIndexedSlices[+T](
    indices: Output[Long],
    values: Output[T],
    denseShape: Output[Long] = null
) extends OutputLike[T] {
  /** Graph that contains `indices`, `values`, and `denseShape`. */
  override def graph: Graph = {
    Op.getGraphFromInputs(Set(values.op, indices.op, denseShape.op))
  }

  /** Name of this op output indexed slices. */
  override def name: String = {
    s"${values.name}[${indices.name}]" +
        (if (denseShape != null) s"(shape = ${denseShape.name})" else "")
  }

  /** Data type of these op output indexed slices. */
  override def dataType: DataType[T] = {
    values.dataType
  }

  /** Device on which these op output indexed slices will be placed. */
  override def device: String = {
    values.device
  }

  /** Op that outputs these indexed slices. */
  override def op: Op[Seq[Output[Any]], Seq[Output[Any]]] = {
    values.op
  }

  /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input[Any]] = {
    values.consumers
  }

  /** Gets the [[Shape]] corresponding to the shape of the dense tensor that these indexed slices represent.
    *
    * @return Dense tensor shape.
    */
  def shape: Shape = Output.constantValueAsShape(denseShape).get

  /** Returns the [[Output]] that this [[OutputLike]] object represents. */
  override def toOutput: Output[T] = {
    if (denseShape == null) {
      throw new IllegalStateException(
        s"Conversion of 'OutputIndexedSlices', '$this', " +
            s"which has no dense shape information available, is not possible.")
    }

    val denseShapeValue = Output.constantValueAsShape(denseShape)

    if (denseShapeValue.isEmpty) {
      logger.warn(
        "Converting sparse 'OutputIndexedSlices' to a dense 'Output' of unknown shape. " +
            "This may consume a large amount of memory.")
    } else if (denseShapeValue.get.numElements >= LARGE_SPARSE_TENSOR_SIZE) {
      logger.warn(
        s"Converting sparse 'OutputIndexedSlices' to a dense 'Output' with " +
            s"${denseShapeValue.get.numElements} elements. This may consume a " +
            s"large amount of memory.")
    }

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = new IsNumeric[T] {}

    Op.nameScope(s"${values.op.name}/ToOutput") {
      Math.unsortedSegmentSum(
        data = values,
        segmentIndices = indices,
        segmentsNumber = denseShape.gather(0))
    }
  }

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices[T] = {
    this
  }

  override def toString: String = {
    s"OutputIndexedSlices(" +
        s"indices = ${indices.name}, " +
        s"values = ${values.name}, " +
        s"denseShape = ${denseShape.name}, " +
        s"device = $device)"
  }
}

/** Represents a sparse op output.
  *
  * TensorFlow represents a sparse tensor as three separate dense tensors: `indices`, `values`, and `denseShape`. In
  * Scala, the three tensors are collected into a [[SparseOutput]] class for ease of use.  If you have separate
  * `indices`, `values`, and `denseShape` tensors, wrap them in a `SparseTensor` object before passing to the
  * relevant sparse tensor manipulation.
  *
  * Concretely, the sparse tensor `SparseOutput(indices, values, denseShape)` comprises the following components,
  * where `N` and `rank` are the number of values and number of dimensions in the [[SparseOutput]], respectively:
  *
  *   - `indices`: Two-dimensional tensor with shape `[N, rank]`, which specifies the indices of the elements
  *     in the sparse tensor that have nonzero values (elements are zero-indexed). For example,
  *     `indices = [[1, 3], [2, 4]]` specifies that the elements with indexes `[1, 3]` and `[2, 4]` have nonzero
  *     values.
  *   - `values`: One-dimensional tensor of any type, with shape `[N]`, which supplies the values for each element in
  *     `indices`. For example, given `indices = [[1, 3], [2, 4]]`, the parameter `values = [18, 3.6]` specifies that
  *     element `[1, 3]` of the sparse tensor has a value of `18`, and element `[2, 4]` of the tensor has a value of
  *     `3.6`.
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
  * For example, the sparse tensor `SparseOutput(indices = [[0, 0], [1, 2]], values = [1, 2], denseShape = [3, 4])`,
  * represents the dense tensor `[[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]`.
  *
  * @param  indices    Two-dimensional tensor with shape `[N, rank]`.
  * @param  values     One-dimensional tensor with shape `[N]`.
  * @param  denseShape One-dimensional tensor with shape `[rank]`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class SparseOutput[+T](
    indices: Output[Long],
    values: Output[T],
    denseShape: Output[Long] = null
) extends OutputLike[T] {
  Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
  Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

  /** Graph that contains `indices`, `values`, and `denseShape`. */
  override def graph: Graph = {
    Op.getGraphFromInputs(Set(values.op, indices.op, denseShape.op))
  }

  /** Name of this sparse op output. */
  override def name: String = {
    s"${values.name}[${indices.name}]" +
        (if (denseShape != null) s"(shape = ${denseShape.name})" else "")
  }

  /** Data type of this sparse op output. */
  override def dataType: DataType[T] = {
    values.dataType
  }

  /** Device on which this sparse op output will be placed. */
  override def device: String = {
    values.device
  }

  /** Op that outputs this sparse tensor. */
  override def op: Op[Seq[Output[Any]], Seq[Output[Any]]] = {
    values.op
  }

  /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
  override def consumers: Array[Input[Any]] = {
    values.consumers
  }

  /** Gets the [[Shape]] corresponding to the shape of the dense tensor that this sparse tensor represents.
    *
    * @return Dense tensor shape.
    */
  def shape: Shape = {
    Output.constantValueAsShape(denseShape).get
  }

  /** Returns the [[Output]] that this [[OutputLike]] object represents. */
  override def toOutput: Output[T] = {
    toOutput()
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
    * @param  name            Name for the created op.
    * @return Created op output, with the same data type as `input.values` and shape `input.denseShape`.
    */
  def toOutput[V >: T](
      defaultValue: Output[V] = Basic.zeros(dataType, Shape()),
      validateIndices: Boolean = true,
      name: String = s"${values.op.name}/ToOutput"
  ): Output[T] = {
    Op.Builder[(Output[Long], Output[Long], Output[T], Output[T]), Output[T]](
      opType = "SparseToDense",
      name = name,
      input = (indices, denseShape, values, defaultValue.cast(dataType))
    ).setAttribute("validate_indices", validateIndices)
        .build().output
  }

  /** Returns an [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    *
    * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
    *                  shape of this tensor at graph creation time (instead of execution time), if known.
    * @return [[OutputIndexedSlices]] that has the same value as this [[OutputLike]].
    */
  @throws[UnsupportedOperationException]
  override def toOutputIndexedSlices(optimize: Boolean = true): OutputIndexedSlices[T] = {
    throw new UnsupportedOperationException(
      s"Cannot convert sparse output '$this' to output indexed slices.")
  }

  override def toString: String = {
    s"OutputIndexedSlices(" +
        s"indices = ${indices.name}, " +
        s"values = ${values.name}, " +
        s"denseShape = ${denseShape.name}, " +
        s"device = $device)"
  }
}
