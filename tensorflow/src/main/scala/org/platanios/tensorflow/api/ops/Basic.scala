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

import org.platanios.tensorflow.api.using
import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, InvalidShapeException}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors.{RowMajorOrder, Tensor}
import org.platanios.tensorflow.api.types.{BooleanIsSupportedType => _, IntIsSupportedType => _, StringIsSupportedType => _, _}

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait Basic {
  //region Tensor Creation Ops

  /** Creates an op that returns a constant tensor.
    *
    * The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
    * (optionally) `shape` (see examples below).
    *
    * The argument `value` can be a constant value, or a tensor. If `value` is a one-dimensional tensor, then its length
    * should be equal to the number of elements implied by the `shape` argument (if specified).
    *
    * The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
    *
    * The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not present,
    * the shape of `value` is used.
    *
    * @param  tensor   Constant value.
    * @param  dataType Data type of the resulting tensor. If not provided, its value will be inferred from the type
    *                  of `value`.
    * @param  shape    Shape of the resulting tensor.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
    *                               the provided `shape`.
    */
  @throws[InvalidShapeException]
  def constant(tensor: Tensor, dataType: DataType = null, shape: Shape = null, name: String = "Constant"): Op.Output = {
    val inferredDataType = if (dataType == null) tensor.dataType else dataType
    val inferredShape = if (shape == null) tensor.shape else shape
    val constantTensor = {
      if (inferredDataType != tensor.dataType || inferredShape != tensor.shape) {
        // TODO: !!! Add support for reshaping tensor.
        if (tensor.numElements == 1) {
          Tensor.fill(inferredDataType, inferredShape)(tensor.scalar)(tensor.dataType.supportedType)
        } else {
          if (inferredShape.numElements.get != tensor.shape.numElements.get)
            throw InvalidShapeException(
              s"Shape '${tensor.shape}' tensor is not valid for shape '$inferredShape' constant op creation.")
          val t = Tensor.allocate(inferredDataType, inferredShape, order = RowMajorOrder)
          for ((thisIndex, tensorIndex) <- t.flattenedIndexIterator zip tensor.flattenedIndexIterator)
            t.setElementAtFlattenedIndex(
              thisIndex, tensor.getElementAtFlattenedIndex(tensorIndex))(tensor.dataType.supportedType)
          t
        }
      } else {
        tensor
      }
    }
    using(constantTensor.nativeView) { nativeTensor =>
      Op.Builder(opType = "Const", name = name)
          .setAttribute("value", nativeTensor)
          .setAttribute("dtype", inferredDataType)
          .build().outputs(0)
    }
  }

  /** Creates an op that returns an immutable tensor from the provided memory region.
    *
    * The current implementation memory-maps the tensor from a file.
    *
    * @param  dataType         Data type of the resulting tensor.
    * @param  shape            Shape of the resulting tensor.
    * @param  memoryRegionName Name of the read-only memory region used by the tensor. Please refer to the C++
    *                          `NewReadOnlyMemoryRegionFromFile` function in `tensorflow::Env`.
    * @param  name             Name for the created op.
    * @return Created op output.
    */
  private[ops] def immutableConstant(
      dataType: DataType, shape: Shape, memoryRegionName: String, name: String = "ImmutableConstant"): Op.Output = {
    Op.Builder(opType = "ImmutableConst", name = name)
        .setAttribute("dtype", dataType)
        .setAttribute("shape", shape)
        .setAttribute("memory_region_name", memoryRegionName)
        .build().outputs(0)
  }

  /** Creates an op that returns a tensor with all elements set to zero.
    *
    * This operation returns a tensor of type `dataType` with shape `shape` and all elements set to zero.
    *
    * For example:
    * {{{
    *   zeros(Shape(3, 4), Int32) == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    * }}}
    *
    * @param  shape    Shape of the output tensor.
    * @param  dataType Data type of the output tensor.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def zeros(shape: Shape, dataType: DataType = FLOAT32, name: String = "Zeros"): Op.Output = {
    dataType match {
      case BOOLEAN => constant(Tensor.fill(BOOLEAN, shape)(false), name = name)
      case STRING => constant(Tensor.fill(STRING, shape)(""), name = name)
      case _ => constant(Tensor.fill(dataType, shape)(0), name = name)
    }
  }

  /** Creates an op that returns a tensor of zeros with the same shape and data type as `input`.
    *
    * Given a single tensor (`input`), this op returns a tensor of the same type and shape as `input` but with all
    * elements set to zero. Optionally, you can use `dataType` to specify a new type for the returned tensor.
    *
    * For example:
    * {{{
    *   // 't' is [[1, 2, 3], [4, 5, 6]]
    *   zerosLike(t) == [[0, 0, 0], [0, 0, 0]]
    * }}}
    *
    * @param  input    Input tensor.
    * @param  dataType Data type of the output tensor.
    * @param  optimize Booelan flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def zerosLike(
      input: Op.Output, dataType: DataType = null, optimize: Boolean = true, name: String = "ZerosLike"): Op.Output = {
    val outputDataType = if (dataType != null) dataType else input.dataType
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a zeros tensor independent of the value of 'tensor' since the shape is known statically.
      zeros(input.shape, outputDataType, name)
    } else if (outputDataType != input.dataType) {
      Op.Builder(opType = "ZerosLike", name = name)
          .addInput(Math.cast(input, outputDataType))
          .build().outputs(0)
    } else {
      Op.Builder(opType = "ZerosLike", name = name)
          .addInput(input)
          .build().outputs(0)
    }
  }

  /** Creates an op that returns a tensor with all elements set to one.
    *
    * This operation returns a tensor of type `dataType` with shape `shape` and all elements set to one.
    *
    * For example:
    * {{{
    *   ones(Shape(3, 4), Int32) == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    * }}}
    *
    * @param  shape    Shape of the output tensor.
    * @param  dataType Data type of the output tensor.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def ones(shape: Shape, dataType: DataType = FLOAT32, name: String = "Ones"): Op.Output = {
    dataType match {
      case BOOLEAN => constant(Tensor.fill(BOOLEAN, shape)(true), name = name)
      case _ => constant(Tensor.fill(dataType, shape)(1), name = name)
    }
  }

  /** Creates an op that returns a tensor of ones with the same shape and data type as `input`.
    *
    * Given a single tensor (`input`), this op returns a tensor of the same type and shape as `input` but with all
    * elements set to one. Optionally, you can use `dataType` to specify a new type for the returned tensor.
    *
    * For example:
    * {{{
    *   // 't' is [[1, 2, 3], [4, 5, 6]]
    *   onesLike(t) == [[1, 1, 1], [1, 1, 1]]
    * }}}
    *
    * @param  input    Input tensor.
    * @param  dataType Data type of the output tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def onesLike(
      input: Op.Output, dataType: DataType = null, optimize: Boolean = true, name: String = "OnesLike"): Op.Output = {
    val outputDataType = if (dataType != null) dataType else input.dataType
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a ones tensor independent of the value of 'tensor' since the shape is known statically.
      ones(input.shape, outputDataType, name)
    } else if (outputDataType != input.dataType) {
      Op.Builder(opType = "OnesLike", name = name)
          .addInput(Math.cast(input, outputDataType))
          .build().outputs(0)
    } else {
      Op.Builder(opType = "OnesLike", name = name)
          .addInput(input)
          .build().outputs(0)
    }
  }

  /** Creates an op that returns a tensor filled with the provided scalar value.
    *
    * The op creates a tensor of shape `shape` and fills it with `value`.
    *
    * For example:
    * {{{
    *   fill(Shape(2, 3), 9) == [[9, 9, 9], [9, 9, 9]]
    * }}}
    *
    * @param  shape    Shape of the output tensor.
    * @param  value    Value to fill the output tensor.
    * @param  dataType Optional data type for the created tensor.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def fill(
      shape: Op.Output, value: Op.Output, dataType: DataType = null, name: String = "Fill"): Op.Output = {
    Op.Builder(opType = "Fill", name = name)
        .addInput(shape)
        .addInput(if (dataType == null || dataType == value.dataType) value else Math.cast(value, dataType))
        .build().outputs(0)
  }

  /** Creates a placeholder op for a tensor that will always be fed.
    *
    * IMPORTANT NOTE: This op will produce an error if evaluated. Its value must be fed when using `Session.run`. It is
    * intended as a way to represent a value that will always be fed, and to provide attributes that enable the fed
    * value to be checked at runtime.
    *
    * @param  dataType Data type of the elements in the tensor that will be fed.
    * @param  shape    Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *                  completely unknown.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def placeholder(dataType: DataType, shape: Shape = null, name: String = "Placeholder"): Op.Output = {
    val opBuilder = Op.Builder(opType = "Placeholder", name = name)
        .setAttribute("dtype", dataType)
    if (shape != null)
      opBuilder.setAttribute("shape", shape)
    opBuilder.build().outputs(0)
  }

  /** Creates a placeholder op that passes through `defaultValue` when its input is not fed.
    *
    * @param  defaultValue Default value to pass through when no input is fed for this placeholder.
    * @param  shape        Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *                      completely unknown.
    * @param  name         Name for the created op.
    * @return Created op output.
    */
  def placeholderWithDefault(defaultValue: Tensor, shape: Shape, name: String = "PlaceholderWithDefault"): Op.Output = {
    Op.Builder(opType = "PlaceholderWithDefault", name = name)
        .addInput(Op.createWith(nameScope = name)(constant(tensor = defaultValue, name = "DefaultValue")))
        .setAttribute("shape", shape)
        .build().outputs(0)
  }

  /** Creates a placeholder op for a sparse tensor that will always be fed.
    *
    * IMPORTANT NOTE: This op will produce an error if evaluated. Its value must be fed when using `Session.run`. It is
    * intended as a way to represent a value that will always be fed, and to provide attributes that enable the fed
    * value to be checked at runtime.
    *
    * @param  dataType Data type of the elements in the tensor that will be fed.
    * @param  shape    Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *                  completely unknown. This represents the shape of the dense tensor that corresponds to the sparse
    *                  tensor that this placeholder refers to.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sparsePlaceholder(
      dataType: DataType, shape: Shape = null, name: String = "SparsePlaceholder"): Op.SparseOutput = {
    Op.SparseOutput(
      indices = placeholder(dataType, Shape(-1), name + "/Indices"),
      values = placeholder(INT64, Shape(-1, -1), name + "/Values"),
      denseShape =
          if (shape == null) placeholder(INT64, Shape(-1), name + "/Shape") else constant(shape.toTensor()))
  }

  //endregion Tensor Creation Ops

  //region Tensor Shape Ops

  /** Creates an op that returns the rank of a tensor.
    *
    * The op returns an integer representing the rank of `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *   // 't' has shape [2, 2, 3]
    *   rank(t) == 3
    * }}}
    *
    * Note that the rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number of
    * indices required to uniquely select each element of the tensor. Rank is also known as order, degree, or number of
    * dimensions.
    *
    * @param  input    Tensor whose rank to return.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  rank value that `input` has at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def rank(input: Op.Output, optimize: Boolean = true, name: String = "Rank"): Op.Output = {
    val inputRank = input.shape.rank
    if (optimize && inputRank != -1)
      constant(Tensor.fill(INT32, Shape())(inputRank), name = name)
    else
      Op.Builder(opType = "Rank", name = name)
          .addInput(input)
          .build().outputs(0)
  }

  /** Creates an op that returns the rank of a sparse tensor.
    *
    * The op returns an integer representing the rank of `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *   // 't' has shape [2, 2, 3]
    *   rank(t) == 3
    * }}}
    *
    * Note that the rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number of
    * indices required to uniquely select each element of the tensor. Rank is also known as order, degree, or number of
    * dimensions.
    *
    * @param  input Tensor whose rank to return.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def sparseRank(input: Op.SparseOutput, dataType: DataType = INT32, name: String = "Rank"): Op.Output = {
    size(input.denseShape, dataType, name = name)
  }

  /** Creates an op that returns the size of a tensor.
    *
    * The op returns a number representing the number of elements in `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
    *   size(t) == 12
    * }}}
    *
    * @param  input    Tensor whose size to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  number of elements provided by the shape of that `input` at graph creation time (instead of
    *                  execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def size(
      input: Op.Output, dataType: DataType = INT32, optimize: Boolean = true,
      name: String = "Size"): Op.Output = {
    val inputShape = input.shape
    if (optimize && inputShape.isFullyDefined)
      constant(Tensor.fill(dataType, Shape())(inputShape.numElements.get), name = name)
    else
      Op.Builder(opType = "Size", name = name)
          .addInput(input)
          .setAttribute("out_type", dataType)
          .build().outputs(0)
  }

  /** Creates an op that returns the size of a sparse tensor.
    *
    * The op returns a number representing the number of elements in `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
    *   size(t) == 12
    * }}}
    *
    * @param  input    Tensor whose size to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sparseSize(
      input: Op.SparseOutput, dataType: DataType = INT32, name: String = "SparseSize"): Op.Output = {
    Op.createWith(nameScope = name) {
      Math.product(Math.cast(input.denseShape, dataType), Array(0))
    }
  }

  /** Creates an op that returns the shape of a tensor.
    *
    * This op returns a one-dimensional tensor representing the shape of `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *   shape(t) == [2, 2, 3]
    * }}}
    *
    * @param  input    Tensor whose shape to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  shape of that `input` at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output, which is one-dimensional.
    */
  def shape(
      input: Op.Output, dataType: DataType = INT32, optimize: Boolean = true,
      name: String = "Shape"): Op.Output = {
    val inputShape = input.shape
    if (optimize && inputShape.isFullyDefined)
      constant(inputShape.toTensor(dataType), name = name) // TODO: [OPTIMIZE]
    else
      Op.Builder(opType = "Shape", name = name)
          .addInput(input)
          .setAttribute("out_type", dataType)
          .build().outputs(0)
  }

  /** Creates an op that returns the shape of a sparse tensor.
    *
    * This op returns a one-dimensional tensor representing the shape of `input`.
    *
    * For example:
    * {{{
    *   // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *   shape(t) == [2, 2, 3]
    * }}}
    *
    * @param  input    Tensor whose shape to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @return Created op output, which is one-dimensional.
    */
  def sparseShape(
      input: Op.SparseOutput, dataType: DataType = INT32, name: String = "SparseShape"): Op.Output = {
    Math.cast(input.denseShape, dataType, name = name)
  }

  /** Creates an op that returns the shape of an array of tensors.
    *
    * This op returns an array of one-dimensional tensors, each one representing the shape of the corresponding tensor
    * in `inputs`.
    *
    * @param  inputs   Tensors whose shapes to return.
    * @param  dataType Optional data type to use for the outputs of this op.
    * @return Created op outputs, all of which are one-dimensional.
    */
  def shapeN(
      inputs: Array[Op.Output], dataType: DataType = INT32, name: String = "ShapeN"): Array[Op.Output] = {
    Op.Builder(opType = "Shape", name = name)
        .addInputs(inputs)
        .setAttribute("out_type", dataType)
        .build().outputs
  }

  //endregion Tensor Shape Ops

  //region Tensor Manipulation Ops

  /** Creates an op that returns a tensor with the same shape and contents as the input tensor.
    *
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def identity[T <: Op.OutputLike](input: T, name: String = "Identity"): T = {
    Op.createWithNameScope(nameScope = name, Set(input.op)) {
      input match {
        case i: Op.Output =>
          Op.Builder(opType = "Identity", name = name)
              .addInput(i)
              .build().outputs(0)
        case i: Op.OutputIndexedSlices =>
          val values = identity(i.values, name = "ValuesIdentity")
          val indices = identity(i.indices, name = "IndicesIdentity")
          val denseShape = {
            if (i.denseShape != null)
              identity(i.denseShape, name = "DenseShapeIdentity")
            else
              null
          }
          Op.OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        case i: Op.SparseOutput =>
          val values = identity(i.values, name = "ValuesIdentity")
          val indices = identity(i.indices, name = "IndicesIdentity")
          val denseShape = identity(i.denseShape, name = "DenseShapeIdentity")
          Op.SparseOutput(indices = indices, values = values, denseShape = denseShape)
      }
    }.asInstanceOf[T]
  }

  /** Creates an op that inserts a dimension of size 1 into a tensor's shape.
    *
    * Given an op output `input`, this op inserts a dimension of size 1 at the dimension index `axis` of `input`'s
    * shape. The dimension index `axis` starts at zero; if you specify a negative number for `axis` it is counted
    * backwards from the end.
    *
    * This op is useful if you want to add a batch dimension to a single element. For example, if you have a single
    * image of shape `[height, width, channels]`, you can make it a batch of 1 image with `expandDims(image, 0)`, which
    * will make the shape equal to `[1, height, width, channels]`.
    *
    * For example:
    * {{{
    *   // 't1' is an op output with shape [2]
    *   shape(expandDims(t1, 0)) == [1, 2]
    *   shape(expandDims(t1, 1)) == [2, 1]
    *   shape(expandDims(t1, -1)) == [2, 1]
    *
    *   // 't2' is a tensor of shape [2, 3, 5]
    *   shape(expandDims(t2, 0)) == [1, 2, 3, 5]
    *   shape(expandDims(t2, 2)) == [2, 3, 1, 5]
    *   shape(expandDims(t2, 3)) == [2, 3, 5, 1]
    * }}}
    *
    * This op requires that `-1 - input.shape.rank <= axis <= input.shape.rank`.
    *
    * This op is related to [[squeeze]], which removes dimensions of size 1.
    *
    * @param  input Input tensor.
    * @param  axis  Dimension index at which to expand the shape of `input`.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def expandDims(input: Op.Output, axis: Int, name: String = "ExpandDims"): Op.Output = {
    Op.Builder(opType = "ExpandDims", name = name)
        .addInput(input)
        .addInput(Op.createWith(nameScope = name)(constant(tensor = axis, name = "Axis")))
        .build().outputs(0)
  }

  /** Creates an op that removes dimensions of size 1 from the shape of a tensor.
    *
    * Given a tensor `input`, this op returns a tensor of the same data type, with all dimensions of size 1 removed. If
    * `axes` is specified, then only the dimensions specified by that array will be removed. In that case, all these
    * dimensions need to have size 1.
    *
    * For example:
    * {{{
    *   // 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    *   shape(squeeze(t)) == [2, 3]
    *   shape(squeeze(t, Array(2L, 4L))) ==> [1, 2, 3, 1]
    * }}}
    *
    * @param  input Input tensor.
    * @param  axes  Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
    *               will be squeezed.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def squeeze(input: Op.Output, axes: Array[Int] = null, name: String = "Squeeze"): Op.Output = {
    val builder = Op.Builder(opType = "Squeeze", name = name)
        .addInput(input)
    if (axes != null)
      builder.setAttribute("squeeze_dims", axes.map(_.asInstanceOf[Long]))
    builder.build().outputs(0)
  }

  /** Creates an op that stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
    *
    * The op packs the list of tensors in `inputs` into a tensor with rank one higher than each tensor in `inputs`, by
    * packing them along the `axis` dimension. Given a list of `N` tensors of shape `[A, B, C]`:
    *   - If `axis == 0`, then the output tensor will have shape `[N, A, B, C]`.
    *   - If `axis == 1`, then the output tensor will have shape `[A, N, B, C]`.
    *   - If `axis == -1`, then the output tensor will have shape `[A, B, C, N]`.
    *   - etc.
    *
    * For example:
    * {{{
    *   // 'x' is [1, 4]
    *   // 'y' is [2, 5]
    *   // 'z' is [3, 6]
    *   stack(Array(x, y, z)) == [[1, 4], [2, 5], [3, 6]]         // Packed along the first dimension.
    *   stack(Array(x, y, z), axis = 1) == [[1, 2, 3], [4, 5, 6]] // Packed along the second dimension.
    * }}}
    *
    * This op is the opposite of `unstack`.
    *
    * @param  inputs Input tensors to be stacked.
    * @param  axis   Dimension along which to stack the input tensors.
    * @param  name   Name for the created op.
    * @return Created op output.
    * @throws InvalidShapeException     If the input tensor shapes are not compatible with each other.
    * @throws IndexOutOfBoundsException If `axis` is not within the expected output tensor shape rank.
    */
  @throws[InvalidShapeException]
  @throws[IndexOutOfBoundsException]
  def stack(inputs: Array[Op.Output], axis: Int = 0, name: String = "Stack"): Op.Output = {
    val inputsShape = inputs.head.shape
    inputs.tail.foreach(_.shape.assertIsCompatibleWith(inputsShape))
    if (inputsShape.rank != -1) {
      val expandedRank = inputsShape.rank + 1
      if (axis < -expandedRank || axis >= expandedRank)
        throw new IndexOutOfBoundsException(s"Provided axis, $axis, is not in [${-expandedRank}, $expandedRank).")
    }
    Op.Builder(opType = "Pack", name = name)
        .addInputList(inputs)
        .setAttribute("axis", axis)
        .build().outputs(0)
  }

  /** Creates an op that stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor, in parallel.
    *
    * The op packs the list of tensors in `inputs` into a tensor with rank one higher than each tensor in `inputs`, by
    * packing them along the first dimension. Given a list of `N` tensors of shape `[A, B, C]`, the output tensor will
    * have shape `[N, A, B, C]`.
    *
    * For example:
    * {{{
    *   // 'x' is [1, 4]
    *   // 'y' is [2, 5]
    *   // 'z' is [3, 6]
    *   parallelStack(Array(x, y, z)) == [[1, 4], [2, 5], [3, 6]]
    * }}}
    *
    * The op requires that the shape of all input tensors is known at graph construction time.
    *
    * The difference between `stack` and `parallelStack` is that `stack` requires all of the inputs be computed before
    * the operation will begin executing, but does not require that the input shapes be known during graph construction.
    * `parallelStack` will copy pieces of the input into the output as they become available. In some situations this
    * can provide a performance benefit.
    *
    * @param  inputs Input tensors to be stacked.
    * @param  name   Name for the created op.
    * @return Created op output.
    * @throws InvalidShapeException If the input tensor shapes are not compatible with each other.
    */
  @throws[InvalidShapeException]
  def parallelStack(inputs: Array[Op.Output], name: String = "ParallelStack"): Op.Output = {
    val inputsShape = inputs.head.shape
    inputs.tail.foreach(_.shape.assertIsCompatibleWith(inputsShape))
    val outputShape = Shape(inputs.length).concatenateWith(inputsShape)
    Op.Builder(opType = "ParallelConcat", name = name)
        .addInputList(inputs)
        .setAttribute("shape", outputShape)
        .build().outputs(0)
  }

  /** Creates an op that unpacks the provided dimension of a rank-`R` tensor into a list of rank-`(R-1)` tensors.
    *
    * The op unpacks `number` tensors from `input` by chipping it along the `axis` dimension. If `number == -1` (i.e.,
    * unspecified), its value is inferred from the shape of `input`. If `input.shape(axis)` is not known, then an
    * [[IllegalArgumentException]] is thrown.
    *
    * For example, given a tensor of shape `[A, B, C, D]`:
    *   - If `axis == 0`, then the `i`th tensor in the output is the slice `input(i, ::, ::, ::)` and each tensor in the
    * output will have shape `[B, C, D]`.
    *   - If `axis == 1`, then the `i`th tensor in the output is the slice `input(::, i, ::, ::)` and each tensor in the
    * output will have shape `[A, C, D]`.
    *   - If `axis == -1`, then the `i`th tensor in the output is the slice `input(::, ::, ::, i)` and each tensor in
    * the output will have shape `[A, B, C]`.
    *   - etc.
    *
    * This op is the opposite of `stack`.
    *
    * @param  input  Rank `R > 0` `Tensor` to be unstacked.
    * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
    * @param  axis   Dimension along which to unstack the input tensor.
    * @param  name   Name for the created op.
    * @return Created op outputs.
    * @throws IndexOutOfBoundsException If `axis` is not within the range [-R, R).
    * @throws IllegalArgumentException  If `number` is not specified and its value cannot be inferred.
    */
  @throws[IndexOutOfBoundsException]
  @throws[IllegalArgumentException]
  def unstack(input: Op.Output, number: Int = -1, axis: Int = 0, name: String = "Unstack"): Array[Op.Output] = {
    val num: Int = {
      if (number >= 0) {
        number
      } else {
        val inputShape = input.shape
        val inputShapeRank = inputShape.rank
        if (inputShapeRank != -1 && (axis < -inputShapeRank || axis >= inputShapeRank))
          throw new IndexOutOfBoundsException(
            s"Provided axis, $axis, is not in [${-inputShapeRank}, $inputShapeRank).")
        inputShape(axis)
      }
    }
    if (num == -1)
      throw new IllegalArgumentException(s"Cannot infer number of tensors to unstack from shape '${input.shape}'.")
    Op.Builder(opType = "Unpack", name = name)
        .addInput(input)
        .setAttribute("num", num)
        .setAttribute("axis", axis)
        .build().outputs
  }

  /** Creates an op that concatenates tensors along one dimension.
    *
    * The op concatenates the list of tensors `inputs` along the dimension `axis`. If
    * `inputs(i).shape = [D0, D1, ..., Daxis(i), ..., Dn]`, then the concatenated tensor will have shape
    * `[D0, D1, ..., Raxis, ..., Dn]`, where `Raxis = sum(Daxis(i))`. That is, the data from the input tensors is joined
    * along the `axis` dimension.
    *
    * For example:
    * {{{
    *   // 't1' is equal to [[1, 2, 3], [4, 5, 6]]
    *   // 't2' is equal to [[7, 8, 9], [10, 11, 12]]
    *   concat(Array(t1, t2), 0) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    *   concat(Array(t1, t2), 1) == [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    *
    *   // 't3' has shape [2, 3]
    *   // 't4' has shape [2, 3]
    *   shape(concat(Array(t3, t4), 0)) == [4, 3]
    *   shape(concat(Array(t3, t4), 1)) == [2, 6]
    * }}}
    *
    * Note that, if you want to concatenate along a new axis, it may be better to use the `stack` op instead:
    * {{{
    *   concat(tensors.map(t => expandDims(t, axis)), axis) == stack(tensors, axis)
    * }}}
    *
    * @param  inputs Input tensors to be concatenated.
    * @param  axis   Dimension along which to concatenate the input tensors.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def concatenate(inputs: Seq[Op.Output], axis: Int = 0, name: String = "Concatenate"): Op.Output = {
    val axisConstant = Op.createWith(nameScope = name)(constant(tensor = axis, name = "Axis"))
    if (inputs.length == 1) {
      Op.createWith(nameScope = name)(identity(inputs.head))
    } else {
      Op.Builder(opType = "ConcatV2", name = name)
          .addInputs(inputs)
          .addInput(axisConstant)
          .build().outputs(0)
    }
  }

  /** Creates an op that computes offsets of `concatenate` inputs within its output.
    *
    * For example:
    * {{{
    *   // 'x' is a tensor containing values [2, 2, 7]
    *   // 'y' is a tensor containing values [2, 3, 7]
    *   // 'z' is a tensor containing values [2, 5, 7]
    *   tf.concatenateOffset(Seq(x, y, z), 2) ==> [0, 0, 0], [0, 2, 0], [0, 5, 0]
    * }}}
    *
    * This function is typically used by gradient computations for a `concatenate` op.
    *
    * @param  shapes Sequence of `N` `INT32` vectors representing the shapes of the tensors being concatenated.
    * @param  axis   `INT32` scalar representing the dimension along which to concatenate.
    * @param  name   Name for the created op.
    * @return Sequence of `N` `INT32` vectors representing the starting offset of the input tensors within the
    *         concatenated output.
    * @throws IllegalArgumentException If any of the `shapes` is not an `INT32` vector or if `axis` is not an `INT32`
    *                                  scalar.
    */
  @throws[IllegalArgumentException]
  private[ops] def concatenateOffset(
      shapes: Seq[Op.Output], axis: Op.Output, name: String = "ConcatenateOffset"): Seq[Op.Output] = {
    if (shapes.length < 2)
      throw new IllegalArgumentException(s"At least 2 shapes need to be provided (actual provided = ${shapes.length}).")
    if (shapes.exists(s => s.dataType != INT32 || s.shape.rank > 1))
      throw new IllegalArgumentException("The provided shapes need to be INT32 vectors.")
    if (axis.dataType != INT32 || axis.shape.rank != 0)
      throw new IllegalArgumentException(
        s"The provided axis (dataType = ${axis.dataType}, shape = ${axis.shape}) needs to be an INT32 scalar.")
    Op.Builder(opType = "ConcatOffset", name = name)
        .addInput(axis)
        .addInputs(shapes)
        .build().outputs.toSeq
  }

  /** Creates an op that splits a tensor into sub-tensors.
    *
    * The op splits `input` along dimension `axis` into `numSplits` smaller tensors. It requires that `numSplits` evenly
    * splits `input.shape(axis)`.
    *
    * For example:
    * {{{
    *   // 't' is a tensor with shape [5, 30]
    *   // Split 't' into 3 tensors along dimension 1:
    *   val splits = split(t, numSplits = 3, axis = 1)
    *   shape(splits(0)) == [5, 10]
    *   shape(splits(1)) == [5, 10]
    *   shape(splits(2)) == [5, 10]
    * }}}
    *
    * @param  input     Input tensor to split.
    * @param  numSplits Number of splits to obtain along the `axis` dimension.
    * @param  axis      Dimension along which to split the input tensor.
    * @param  name      Name for the created op.
    * @return Created op outputs.
    */
  def splitEvenly(input: Op.Output, numSplits: Int, axis: Int = 0, name: String = "Split"): Array[Op.Output] = {
    Op.Builder(opType = "Split", name = name)
        .addInput(Op.createWith(nameScope = name)(constant(tensor = axis, name = "Axis")))
        .addInput(input)
        .setAttribute("num_split", numSplits)
        .build().outputs
  }

  /** Creates an op that splits a tensor into sub-tensors.
    *
    * The op splits `input` along dimension `axis` into `splitSizes.length` smaller tensors. The shape of the `i`-th
    * smaller tensor has the same size as the `input` except along dimension `axis` where the size is equal to
    * `splitSizes(i)`.
    *
    * For example:
    * {{{
    *   // 't' is a tensor with shape [5, 30]
    *   // Split 't' into 3 tensors with sizes [4, 5, 11] along dimension 1:
    *   val splits = split(t, splitSizes = [4, 15, 11], axis = 1)
    *   shape(splits(0)) == [5, 4]
    *   shape(splits(1)) == [5, 15]
    *   shape(splits(2)) == [5, 11]
    * }}}
    *
    * @param  input      Input tensor to split.
    * @param  splitSizes Sizes for the splits to obtain.
    * @param  axis       Dimension along which to split the input tensor.
    * @param  name       Name for the created op.
    * @return Created op outputs.
    */
  def split(input: Op.Output, splitSizes: Tensor, axis: Int = 0, name: String = "Split"): Array[Op.Output] = {
    Op.Builder(opType = "SplitV", name = name)
        .addInput(input)
        .addInput(Op.createWith(nameScope = name)(constant(tensor = splitSizes, name = "Sizes")))
        .addInput(Op.createWith(nameScope = name)(constant(tensor = axis, name = "Axis")))
        .build().outputs
  }

  /** Creates an op that tiles the provided input tensor.
    *
    * The op creates a new tensor by replicating `input` `multiples` times. The output tensor's `i`th dimension has
    * `input.shape(i) * multiples(i)` elements, and the values of `input` are replicated `multiples(i)` times along
    * the `i`th dimension. For example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
    *
    * @param  input     Tensor to tile.
    * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
    *                   of `input`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def tile(input: Op.Output, multiples: Op.Output, name: String = "Tile"): Op.Output = {
    Op.Builder(opType = "Tile", name = name)
        .addInput(input)
        .addInput(multiples)
        .build().outputs(0)
  }

  // TODO: Add support for the "pad", the "mirrorPad", and the "meshGrid" ops.
  // TODO: Add support for the "spaceToBatch", the "batchToSpace", the "spaceToDepth", and the "depthToSpace" ops.
  // TODO: Add support for the "extractImagePatches" op (maybe in an "ImageOps" object).

  /** Creates an op that reshapes a tensor.
    *
    * Given `input`, this operation returns a tensor that has the same values as `input` but has shape `shape`. If one
    * component of `shape` is the special value `-1`, then the size of that dimension is computed so that the total size
    * remains constant. In particular, a `shape` of `[-1]` flattens a tensor into a one-dimensional tensor. At most one
    * component of `shape` can be set to `-1`.
    *
    * If `shape` is a one-dimensional or higher tensor, then the operation returns a tensor with shape `shape` filled
    * with the values of `input`. In this case, the number of elements implied by `shape` must be the same as the number
    * of elements in `input`.
    *
    * For example:
    * {{{
    *   // Tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9] => It has shape [9]
    *   reshape(t, [3, 3]) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    *
    *   // Tensor 't' is [[[1, 1], [2, 2]],
    *   //                [[3, 3], [4, 4]]] => It has shape [2, 2, 2]
    *   reshape(t, [2, 4] == [[1, 1, 2, 2],
    *                         [3, 3, 4, 4]]
    *
    *   // Tensor 't' is [[[1, 1, 1],
    *                      [2, 2, 2]],
    *                     [[3, 3, 3],
    *                      [4, 4, 4]],
    *                     [[5, 5, 5],
    *                      [6, 6, 6]]] => It has shape [3, 2, 3]
    *   reshape(t, [-1]) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
    *
    *   // '-1' can also be used to infer the shape. Some examples follow.
    *
    *   // '-1' is inferred to be 9:
    *   reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    *                            [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    *
    *   // '-1' is inferred to be 2:
    *   reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    *                            [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    *
    *   // '-1' is inferred to be 3:
    *   reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
    *                                 [2, 2, 2],
    *                                 [3, 3, 3]],
    *                                [[4, 4, 4],
    *                                 [5, 5, 5],
    *                                 [6, 6, 6]]]
    *
    *   // Tensor 't' is [7]
    *   // An empty shape passed to 'reshape' will result in a scalar
    *   reshape(t, []) == 7
    * }}}
    *
    * @param  input Input tensor.
    * @param  shape Shape of the output tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def reshape(input: Op.Output, shape: Op.Output, name: String = "Reshape"): Op.Output = {
    Op.Builder(opType = "Reshape", name = name)
        .addInput(input)
        .addInput(shape)
        .build().outputs(0)
  }

  /** Creates an op that transposes `input`. THe op permutes the dimensions of `input` according to `permutation`.
    *
    * The returned tensor's dimension `i` will correspond to `input` dimension `permutation(i)`. If `permutation` is not
    * provided, then it is set to `(n - 1, ..., 0)`, where `n` is the rank of the input tensor. Hence by default, this
    * op performs a regular matrix transpose on two-dimensional input tensors.
    *
    * For example:
    * {{{
    *   // Tensor 'x' is [[1, 2, 3], [4, 5, 6]]
    *   transpose(x) == [[1, 4], [2, 5], [3, 6]]
    *   transpose(x, permutation = Array(1, 0)) == [[1, 4], [2, 5], [3, 6]]
    *
    *   // Tensor 'x' is [[[1, 2, 3],
    *   //                 [4, 5, 6]],
    *   //                [[7, 8, 9],
    *   //                 [10, 11, 12]]]
    *   transpose(x, permutation = Array(0, 2, 1)) == [[[1,  4], [2,  5], [3,  6]],
    *                                                  [[7, 10], [8, 11], [9, 12]]]
    *
    * }}}
    *
    * @param  input       Input tensor to transpose.
    * @param  permutation Permutation of the input tensor dimensions.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def transpose(input: Op.Output, permutation: Array[Int] = null, name: String = "Transpose"): Op.Output = {
    if (permutation == null) {
      Op.createWith(nameScope = name) {
        val inputRank = rank(input)
        val reversePermutation = inputRank - constant(1) - Math.range(constant(0), inputRank, constant(1))
        Op.Builder(opType = "Transpose", name = name)
            .addInput(input)
            .addInput(reversePermutation)
            .build().outputs(0)
        // TODO: !!! Set the shape explicitly?
      }
    } else {
      Op.Builder(opType = "Transpose", name = name)
          .addInput(input)
          .addInput(constant(Tensor(permutation.map(Tensor(_)): _*)))
          .build().outputs(0)
    }
  }

  /** Creates an identical op to [[transpose]], except for the fact that the provided `permutation` is itself an op
    * output as opposed to an integer array.
    *
    * @param  input       Input tensor to transpose.
    * @param  permutation `INT32` or `INT64` tensor containing the permutation of the input tensor dimensions.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def transposeDynamic(input: Op.Output, permutation: Op.Output, name: String = "Transpose"): Op.Output = {
    if (permutation.dataType != INT32 && permutation.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${permutation.dataType}' is not supported for the transpose op permutation. " +
            s"Only 'Int32' and 'Int64' are supported.")
    Op.Builder(opType = "Transpose", name = name)
        .addInput(input)
        .addInput(permutation)
        .build().outputs(0)
  }

  /** Creates an op that transposes the last two dimensions of tensor `input`.
    *
    * For example:
    * {{{
    *   // Tensor 'x' is [[1, 2, 3], [4, 5, 6]]
    *   matrixTranspose(x) == [[1, 4], [2, 5], [3, 6]]
    *
    *   // Tensor 'x' has shape [1, 2, 3, 4]
    *   // matrixTranspose(x) has shape [1, 2, 4, 3]
    * }}}
    *
    * Note that [[Math.matMul]] provides named arguments allowing for transposing the matrices involved in the
    * multiplication. This is done with minimal cost, and is preferable to using this function. For example:
    * {{{
    *   matMul(a, b, transposeB = true) // is preferable to:
    *   matMul(a, matrixTranspose(b))
    * }}}
    *
    * @param  input Input tensor to transpose.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def matrixTranspose(input: Op.Output, name: String = "MatrixTranspose"): Op.Output = {
    Op.createWith(nameScope = name) {
      // If we know the number of dimensions statically, we can do two things:
      //   1. Check that `input` is a (batch) matrix.
      //   2. Use a Scala array for the permutation. This preserves static shape information and avoids extra
      //      computation.
      val inputShape = input.shape
      val inputRank = inputShape.rank
      if (inputRank != -1) {
        if (inputRank < 2)
          throw InvalidShapeException(s"'input' should be a (batch) matrix, with rank > 2. Found shape '$inputShape'.")
        val permutation = Range(0, inputRank - 2).toArray ++ Array(inputRank - 1, inputRank - 2)
        transpose(input, permutation)
      } else {
        val inputRank = rank(input)
        val inputRankMinus1 = inputRank - constant(1)
        val inputRankMinus2 = inputRank - constant(2)
        val permutation = concatenate(
          Array(Math.range(constant(0), inputRankMinus2, constant(1)), inputRankMinus1, inputRankMinus2))
        transposeDynamic(input, permutation)
      }
    }
  }

  /** Creates an op that computes the inverse permutation of a tensor.
    *
    * This op computes the inverse of an index permutation. It takes a one-dimensional integer tensor `input`, which
    * represents indices of a zero-based array, and swaps each value with its index position. In other words, for an
    * output tensor `y` and an input tensor `x`, this op computes `y(x(i)) = i`, for `i` in `[0, 1, ..., x.length - 1]`.
    *
    * For example:
    * {{{
    *   // Tensor 't' is [3, 4, 0, 2, 1]
    *   invertPermutation(t) == [2, 4, 3, 0, 1]
    * }}}
    *
    * @param  input One-dimensional `INT32` or `INT64` input tensor
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def invertPermutation(input: Op.Output, name: String = "InvertPermutation"): Op.Output = {
    if (input.dataType != INT32 && input.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${input.dataType}' is not supported for the permutation inversion op input. " +
            s"Only 'Int32' and 'Int64' are supported.")
    if (input.shape.rank != 1 && input.shape.rank != -1)
      throw InvalidShapeException(
        s"Shape '${input.shape}' is not supported for the permutation inversion op input. " +
            s"Only one-dimensional tensors are supported.")
    Op.Builder(opType = "InvertPermutation", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an op that reverses specific dimensions of a tensor.
    *
    * Given an `input` tensor, and an integer array of axes representing the set of dimensions of `input` to reverse,
    * this op reverses each dimension `i` of `input`, for which there exists `j` such that  `axes(j) == i`.
    *
    * `input` can have up to 8 dimensions. The number of dimensions specified in `axes` may be 0 or more entries. If an
    * index is specified more than once, an 'InvalidArgument' error will be raised.
    *
    * For example:
    * {{{
    *   // Tensor 't' is [[[[ 0,  1,  2,  3],
    *   //                  [ 4,  5,  6,  7],
    *   //                  [ 8,  9, 10, 11]],
    *   //                 [[12, 13, 14, 15],
    *   //                  [16, 17, 18, 19],
    *   //                  [20, 21, 22, 23]]]] => It has shape [1, 2, 3, 4]
    *
    *   // 'axes' is [3] or [-1]
    *   reverse(t, axes) == [[[[ 3,  2,  1,  0],
    *                          [ 7,  6,  5,  4],
    *                          [ 11, 10, 9,  8]],
    *                         [[15, 14, 13, 12],
    *                          [19, 18, 17, 16],
    *                          [23, 22, 21, 20]]]]
    *
    *   // 'axes' is [1] or [-3]
    *   reverse(t, axes) == [[[[12, 13, 14, 15],
    *                          [16, 17, 18, 19],
    *                          [20, 21, 22, 23]],
    *                         [[ 0,  1,  2,  3],
    *                          [ 4,  5,  6,  7],
    *                          [ 8,  9, 10, 11]]]]
    *
    *   // 'axes' is [2] or [-2]
    *   reverse(t, axes) == [[[[ 8,  9, 10, 11],
    *                          [ 4,  5,  6,  7],
    *                          [ 0,  1,  2,  3]],
    *                         [[20, 21, 22, 23],
    *                          [16, 17, 18, 19],
    *                          [12, 13, 14, 15]]]]
    * }}}
    *
    * @param  input Input tensor to reverse. It must have rank at most 8.
    * @param  axes  Dimensions of the input tensor to reverse.
    * @param  name  Name for the created op.
    * @return Created op output that has the same shape as `input`.
    */
  def reverse(input: Op.Output, axes: Array[Int], name: String = "Reverse"): Op.Output = {
    Op.Builder(opType = "Reverse", name = name)
        .addInput(input)
        .addInput(Basic.constant(Tensor(axes.map(Tensor(_)): _*)))
        .build().outputs(0)
  }

  /** Creates an identical op to [[reverse]], except for the fact that the provided `axes` are themselves represented as
    * an op output. as opposed to an integer array.
    *
    * @param  input Input tensor to reverse.
    * @param  axes  `INT32` or `INT64` tensor containing the dimensions of the input tensor to reverse.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def reverseDynamic(input: Op.Output, axes: Op.Output, name: String = "Reverse"): Op.Output = {
    if (axes.dataType != INT32 && axes.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${axes.dataType}' is not supported for the reverse op axes. " +
            s"Only 'Int32' and 'Int64' are supported.")
    Op.Builder(opType = "Reverse", name = name)
        .addInput(input)
        .addInput(axes)
        .build().outputs(0)
  }

  /** Creates an op that reverses variable length slices.
    *
    * This op first slices `input` along the dimension `batchAxis`, and for each slice `i`, it reverses the first
    * `sequenceLengths(i)` elements along the dimension `sequenceAxis`.
    *
    * The elements of `sequenceLengths` must obey `sequenceLengths(i) <= input.shape(sequenceAxis)`, and it must be a
    * vector of length `input.shape(batchAxis)`.
    *
    * The output slice `i` along dimension `batchAxis` is then given by input slice `i`, with the first
    * `sequenceLengths(i)` slices along dimension `sequenceAxis` reversed.
    *
    * For example:
    * {{{
    *   // Given:
    *   // sequenceAxis = 1
    *   // batchAxis = 0
    *   // input.shape = [4, 8, ...]
    *   // sequenceLengths = [7, 2, 3, 5]
    *   // slices of 'input' are reversed on 'sequenceAxis', but only up to 'sequenceLengths':
    *   output(0, 0::7, ---) == input(0, 6::-1::, ---)
    *   output(1, 0::2, ---) == input(1, 1::-1::, ---)
    *   output(2, 0::3, ---) == input(2, 2::-1::, ---)
    *   output(3, 0::5, ---) == input(3, 4::-1::, ---)
    *   // while entries past 'sequenceLengths' are copied through:
    *   output(0, 7::, ---) == input(0, 7::, ---)
    *   output(1, 7::, ---) == input(1, 7::, ---)
    *   output(2, 7::, ---) == input(2, 7::, ---)
    *   output(3, 7::, ---) == input(3, 7::, ---)
    *
    *   // In contrast, given:
    *   // sequenceAxis = 0
    *   // batchAxis = 2
    *   // input.shape = [8, ?, 4, ...]
    *   // sequenceLengths = [7, 2, 3, 5]
    *   // slices of 'input' are reversed on 'sequenceAxis', but only up to 'sequenceLengths':
    *   output(0::7, ::, 0, ---) == input(6::-1::, ::, 0, ---)
    *   output(0::2, ::, 1, ---) == input(1::-1::, ::, 1, ---)
    *   output(0::3, ::, 2, ---) == input(2::-1::, ::, 2, ---)
    *   output(0::5, ::, 3, ---) == input(4::-1::, ::, 3, ---)
    *   // while entries past 'sequenceLengths' are copied through:
    *   output(7::, ::, 0, ---) == input(7::, ::, 0, ---)
    *   output(2::, ::, 1, ---) == input(2::, ::, 1, ---)
    *   output(3::, ::, 2, ---) == input(3::, ::, 2, ---)
    *   output(5::, ::, 3, ---) == input(5::, ::, 3, ---)
    * }}}
    *
    * @param  input           Input tensor to reverse.
    * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
    *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
    * @param  sequenceAxis    Tensor dimension which is partially reversed.
    * @param  batchAxis       Tensor dimension along which the reversal is performed.
    * @param  name            Created op name.
    * @return Created op output which has the same shape as `input`.
    */
  def reverseSequence(
      input: Op.Output, sequenceLengths: Op.Output, sequenceAxis: Int, batchAxis: Int = 0,
      name: String = "ReverseSequence"): Op.Output = {
    Op.Builder(opType = "ReverseSequence", name = name)
        .addInput(input)
        .addInput(sequenceLengths)
        .setAttribute("seq_dim", sequenceAxis)
        .setAttribute("batch_dim", batchAxis)
        .build().outputs(0)
  }

  //endregion Tensor Manipulation Ops

  /** Creates an op that returns locations of `true` values in a boolean tensor.
    *
    * The op returns the coordinates of true elements in `input`. The coordinates are returned in a 2-D tensor where the
    * first dimension (rows) represents the number of true elements, and the second dimension (columns) represents the
    * coordinates of the true elements. Note that the shape of the output tensor can vary depending on how many true
    * values there are in `input`. Indices are output in row-major order.
    *
    * For example:
    * {{{
    *   // 'input' tensor is [[true, false]
    *   //                    [true, false]]
    *   // 'input' has two 'true' values and so the output has two coordinates
    *   // 'input' has rank 2 and so each coordinate has two indices
    *   where(input) == [[0, 0],
    *                    [1, 0]]
    *
    *   // `input` tensor is [[[true, false]
    *   //                     [true, false]]
    *   //                    [[false, true]
    *   //                     [false, true]]
    *   //                    [[false, false]
    *   //                     [false, true]]]
    *   // 'input' has 5 'true' values and so the output has 5 coordinates
    *   // 'input' has rank 3 and so each coordinate has three indices
    *   where(input) == [[0, 0, 0],
    *                    [0, 1, 0],
    *                    [1, 0, 1],
    *                    [1, 1, 1],
    *                    [2, 1, 1]]
    * }}}
    *
    * @param  input Input boolean tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def where(input: Op.Output, name: String = "Where"): Op.Output = {
    if (input.dataType != BOOLEAN)
      throw InvalidDataTypeException(
        s"The 'where' op only supports boolean tensors as inputs. It does not support '${input.dataType}' tensors.")
    Op.Builder(opType = "Where", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an op that applies the provided boolean mask to `input`.
    *
    * In general, `0 < mask.rank = K <= tensor.rank`, and `mask`'s shape must match the first `K` dimensions of
    * `tensor`'s shape. We then have: `booleanMask(tensor, mask)(i, j1, --- , jd) = tensor(i1, --- , iK, j1, ---, jd)`,
    * where `(i1, ---, iK)` is the `i`th `true` entry of `mask` (in row-major order).
    *
    * For example:
    * {{{
    *   // 1-D example
    *   tensor = [0, 1, 2, 3]
    *   mask = [True, False, True, False]
    *   booleanMask(tensor, mask) == [0, 2]
    *
    *   // 2-D example
    *   tensor = [[1, 2], [3, 4], [5, 6]]
    *   mask = [True, False, True]
    *   booleanMask(tensor, mask) == [[1, 2], [5, 6]]
    * }}}
    *
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
    * @param  name  Name for the created op output.
    * @return Created op output.
    * @throws InvalidShapeException If the shapes of `input` and `mask` are not compatible.
    */
  @throws[InvalidShapeException]
  def booleanMask(input: Op.Output, mask: Op.Output, name: String = "BooleanMask"): Op.Output = {
    Op.createWithNameScope(name, Set[Op](input.op, mask.op)) {
      val inputShape: Shape = input.shape
      val maskShape: Shape = mask.shape
      val maskRank: Int = maskShape.rank
      if (maskRank < 0)
        throw InvalidShapeException(
          "The rank of the boolean mask must be known, even if some dimension sizes are unknown. For example, " +
              "'Shape(-1)' is fine, but 'Shape.unknown()' is not.")
      if (maskRank == 0)
        throw InvalidShapeException("The boolean mask cannot be a scalar.")
      inputShape(0 :: maskRank).assertIsCompatibleWith(maskShape)
      val dynamicInputShape = shape(input)
      val leadingSize = Math.product(dynamicInputShape(0 :: maskRank), Array(0))
      val reshapedInput = reshape(input, concatenate(Array[Op.Output](leadingSize, dynamicInputShape(maskRank ::)), 0))
      val firstDimension = inputShape(0 :: maskRank).rank
      reshapedInput.setShape(Shape(firstDimension).concatenateWith(inputShape(maskRank ::)))
      gather(reshapedInput, squeeze(where(reshape(mask, Array(-1))), axes = Array(1)))
    }
  }

  // TODO: [OPS] Add support for the "sparseMask" op.

  /** Creates an op that returns a mask tensor representing the first `N` positions of each row of a matrix.
    *
    * For example:
    * {{{
    *   // 'lengths' = [1, 3, 2]
    *   // 'maxLength' = 5
    *   tf.sequenceMask(lengths, maxLength) ==>
    *     [[true, false, false, false, false],
    *      [true,  true,  true, false, false],
    *      [true,  true, false, false, false]]
    * }}}
    *
    * @param  lengths   One-dimensional integer tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @param  dataType  Data type for the output tensor.
    * @param  name      Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If either `lengths` or `maxLength` have invalid rank.
    */
  @throws[IllegalArgumentException]
  def sequenceMask(
      lengths: Op.Output, maxLength: Op.Output = null, dataType: DataType = BOOLEAN,
      name: String = "SequenceMask"): Op.Output = {
    if (lengths.rank != 1)
      throw new IllegalArgumentException(s"'lengths' (shape = ${lengths.shape}) must be a one-dimensional tensor.")
    if (maxLength != null && maxLength.rank != 0)
      throw new IllegalArgumentException(s"'maxLength' (shape = ${maxLength.shape}) must be a scalar tensor.")
    val ops = if (maxLength == null) Set(lengths.op) else Set(lengths.op, maxLength.op)
    Op.createWithNameScope(name, ops) {
      val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
      // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
      // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
      // matrix of size [lengths.shape(0), maxLen].
      val rowVector = Math.range(constant(0, maxLen.dataType), maxLen, constant(1, maxLen.dataType))
      // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
      // into INT32, then so do the elements of 'lengths'.
      val matrix = Math.cast(expandDims(lengths, 1), maxLen.dataType)
      val result = Math.less(rowVector, matrix)
      if (result.dataType == dataType)
        result
      else
        Math.cast(result, dataType)
    }
  }

  /** Creates an op that finds unique elements in a one-dimensional tensor.
    *
    * The op returns a tensor `output` containing all of the unique elements of `input` sorted in the same order that
    * they occur in `input`. This op also returns a tensor `indices` the same size as `input` that contains the
    * index of each value of `input` in the unique output `output`. In other words `output(indices(i)) = input(i)`, for
    * `i` in `[0, 1, ..., input.rank - 1]`.
    *
    * For example:
    * {{{
    *   // Tensor 't' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    *   (output, indices) = unique(t)
    *   // 'output' is [1, 2, 4, 7, 8]
    *   // 'indices' is [0, 0, 1, 2, 2, 2, 3, 4, 4]
    * }}}
    *
    * @param  input One-dimensional input tensor.
    * @param  name  Name for the created op.
    * @return Tuple containing `output` and `indices`.
    */
  def unique(input: Op.Output, name: String = "Unique"): (Op.Output, Op.Output) = {
    if (input.shape.rank != 1 && input.shape.rank != -1)
      throw InvalidShapeException(
        s"Shape '${input.shape}' is not supported for the unique op input. Only one-dimensional tensors are supported.")
    val outputs = Op.Builder(opType = "Unique", name = name)
        .addInput(input)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that finds unique elements in a one-dimensional tensor.
    *
    * The op returns a tensor `output` containing all of the unique elements of `input` sorted in the same order that
    * they occur in `input`. This op also returns a tensor `indices` the same size as `input` that contains the
    * index of each value of `input` in the unique output `output`. Finally, it returns a third tensor `counts` that
    * contains the count of each element of `output` in `input`.
    *
    * For example:
    * {{{
    *   // Tensor 't' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    *   (output, indices, counts) = uniqueWithCounts(t)
    *   // 'output' is [1, 2, 4, 7, 8]
    *   // 'indices' is [0, 0, 1, 2, 2, 2, 3, 4, 4]
    *   // 'counts' is [2, 1, 3, 1, 2]
    * }}}
    *
    * @param  input One-dimensional input tensor.
    * @param  name  Name for the created op.
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts(input: Op.Output, name: String = "UniqueWithCounts"): (Op.Output, Op.Output, Op.Output) = {
    if (input.shape.rank != 1 && input.shape.rank != -1)
      throw InvalidShapeException(
        s"Shape '${input.shape}' is not supported for the unique op input. Only one-dimensional tensors are supported.")
    val outputs = Op.Builder(opType = "UniqueWithCounts", name = name)
        .addInput(input)
        .build().outputs
    (outputs(0), outputs(1), outputs(2))
  }

  /** Creates an op that computes the difference between two lists of numbers or strings.
    *
    * Given a list `x` and a list `y`, this operation returns a list `out` that represents all values that are in `x`
    * but not in `y`. The returned list `output` is sorted in the same order that the numbers appear in `x` (duplicates
    * are preserved). This operation also returns a list `indices` that represents the position of each `out` element in
    * `x`. In other words, `output(i) = x(indices(i))`, for `i` in `[0, 1, ..., output.length - 1]`.
    *
    * For example, given inputs `x = [1, 2, 3, 4, 5, 6]` and `y = [1, 3, 5]`, this op would return
    * `output = [2, 4, 6]` and `indices = [1, 3, 5]`.
    *
    * @param  x             One-dimensional tensor containing the values to keep.
    * @param  y             One-dimensional tensor containing the values to remove.
    * @param  indexDataType Optional data type to use for the output indices of this op. It has to be either `INT32` or
    *                       `INT64`.
    * @param  name          Name for the created op.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff(
      x: Op.Output, y: Op.Output, indexDataType: DataType = INT32,
      name: String = "ListDiff"): (Op.Output, Op.Output) = {
    if (indexDataType != INT32 && indexDataType != INT64)
      throw InvalidDataTypeException(
        s"The index data type cannot be '$indexDataType'. It has to be either 'INT32' or 'INT64'.")
    val outputs = Op.Builder(opType = "ListDiff", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("out_idx", indexDataType)
        .build().outputs
    (outputs(0), outputs(1))
  }

  //region Slice Ops

  /** Creates an op that gathers slices from `input` according to `indices`.
    *
    * `indices` must be an integer tensor of any dimension (usually 0-D or 1-D). The op produces an output tensor with
    * shape `indices.shape + params.shape(1::)`, where:
    * {{{
    *   // Scalar indices
    *   output(::, ---) = input(indices, ::, ---)
    *
    *   // Vector indices
    *   output(i, ::, ---) = input(indices(i), ::, ---)
    *
    *   // Higher rank indices
    *   output(i, ..., j, ::, ---) = input(indices(i, ..., j), ::, ---)
    * }}}
    *
    * If `indices` is a permutation and `indices.length == input.shape(0)`, then this op will permute `input`
    * accordingly.
    *
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def gather(input: Op.Output, indices: Op.Output, name: String = "Gather"): Op.Output = {
    Op.Builder(opType = "Gather", name = name)
        .addInput(input)
        .addInput(indices)
        .build().outputs(0)
  }

  /** Creates an op that gathers values or slices from `input` according to `indices`.
    *
    * `indices` is an integer tensor containing indices into `input`.  The last dimension of `indices` can be equal to
    * at most the rank of `input`, `indices.shape(-1) <= input.rank`. The last dimension of `indices` corresponds to
    * elements (if `indices.shape(-1) == input.rank`), or slices (if `indices.shape(-1) < input.rank`) along dimension
    * `indices.shape(-1)` of `input`. The output has shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    *
    * Some examples follow.
    *
    * Simple indexing into a matrix:
    * {{{
    *   input = [['a', 'b'], ['c', 'd']]
    *   indices = [[0, 0], [1, 1]]
    *   output = ['a', 'd']
    * }}}
    *
    * Slice indexing into a matrix:
    * {{{
    *   input = [['a', 'b'], ['c', 'd']]
    *   indices = [[1], [0]]
    *   output = [['c', 'd'], ['a', 'b']]
    * }}}
    *
    * Indexing into a three-dimensional tensor:
    * {{{
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[1]]
    *   output = [[['a1', 'b1'], ['c1', 'd1']]]
    *
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[0, 1], [1, 0]]
    *   output = [['c0', 'd0'], ['a1', 'b1']]
    *
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[0, 0, 1], [1, 0, 1]]
    *   output = ['b0', 'b1']
    * }}}
    *
    * Batched indexing into a matrix:
    * {{{
    *   input = [['a', 'b'], ['c', 'd']]
    *   indices = [[[0, 0]], [[0, 1]]]
    *   output = [['a'], ['b']]
    * }}}
    *
    * Batched slice indexing into a matrix:
    * {{{
    *   input = [['a', 'b'], ['c', 'd']]
    *   indices = [[[1]], [[0]]]
    *   output = [[['c', 'd']], [['a', 'b']]]
    * }}}
    *
    * Batched indexing into a three-dimensional tensor:
    * {{{
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[[1]], [[0]]]
    *   output = [[[['a1', 'b1'], ['c1', 'd1']]],
    *             [[['a0', 'b0'], ['c0', 'd0']]]]
    *
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    *   output = [[['c0', 'd0'], ['a1', 'b1']],
    *             [['a0', 'b0'], ['c1', 'd1']]]
    *
    *   input = [[['a0', 'b0'], ['c0', 'd0']],
    *            [['a1', 'b1'], ['c1', 'd1']]]
    *   indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    *   output = [['b0', 'b1'], ['d0', 'c1']]
    * }}}
    *
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  name    Name for the created op.
    * @return Created op output that contains the values from `input` gathered from indices given by `indices`, with
    *         shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    */
  def gatherND(input: Op.Output, indices: Op.Output, name: String = "GatherND"): Op.Output = {
    Op.Builder(opType = "GatherNd", name = name)
        .addInput(input)
        .addInput(indices)
        .build().outputs(0)
  }

  // TODO: [OPS] Add support for the "scatterND" op.

  /** Creates an op that returns a slice from `input`.
    *
    * The op output is a tensor with dimensions described by `size`, whose values are extracted from `input`, starting
    * at the offsets in `begin`.
    *
    * Requirements:
    *
    *   - `0 <= begin(i) <= begin(i) + size(i) <= Di, for i in [0, n)`, where `Di` corresponds to the size of
    * the `i`th dimension of `input` and `n` corresponds to the rank of `input`.
    *
    * @param  input Tensor to slice.
    * @param  begin Begin index tensor (must have data type of `INT32` or `INT64`). `begin(i)` specifies the offset into
    *               the `i`th dimension of `input` to slice from.
    * @param  size  Slice size tensor (must have data type of `INT32` or `INT64`). `size(i)` specifies the number of
    *               elements of the `i`th dimension of `input` to slice. If `size(i) == -1`, then all the remaining
    *               elements in dimension `i` are included in the slice (i.e., this is equivalent to setting
    *               `size(i) = input.shape(i) - begin(i)`).
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def slice(input: Op.Output, begin: Op.Output, size: Op.Output, name: String = "Slice"): Op.Output = {
    if (begin.dataType != INT32 && begin.dataType != INT64)
      throw InvalidDataTypeException(
        s"'begin' data type, '${begin.dataType}', is not 'INT32' or 'INT64', as required.")
    if (size.dataType != INT32 && size.dataType != INT64)
      throw InvalidDataTypeException(
        s"'size' data type, '${size.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "Slice", name = name)
        .addInput(input)
        .addInput(begin)
        .addInput(size)
        .build().outputs(0)
  }

  /** Creates an op that returns a strided slice from `input`.
    *
    * Note that most users will want to use the `apply` or the `slice` method of [[Op.Output]] rather than this function
    * directly, as the interface of those methods is much simpler.
    *
    * The goal of the op is to produce a new tensor with a subset of the elements from the `n`-dimensional `input`
    * tensor. The subset is chosen using a sequence of `m` sparse specifications encoded into the arguments of this
    * function. Note that, in some cases, `m` could be equal to `n`, but this need not be the case.
    * Each range specification entry can be one of the following:
    *
    *   - An ellipsis (`---` or `Ellipsis`). Ellipses are used to represent zero or more dimensions of a full-dimension
    * selection and are produced using `ellipsisMask`. For example, `foo(---)` is the identity slice.
    *   - A new axis (`NewAxis`). New axes are used to insert new dimensions of size `1` and are produced using
    * `newAxisMask`. For example, `foo(NewAxis, ---)`, where `foo` has shape `[3, 4]`, produces a new tensor with
    * shape `[1, 3, 4]`.
    *   - A single index (`Index`). This is used to keep only elements that have a given index. For example, if `foo` is
    * a tensor with shape `[5, 6]`, `foo(2, ::)` produces a tensor with shape `[6]`. This is encoded in `begin` and
    * `end` (where `end` has to be equal to `begin + 1`) and in the `shrinkAxisMask` (since an axis is being
    * shrinked).
    *   - A slice (`Slice`). Slices define a range with a `start`, an `end`, and a `step` size. They are used to specify
    * which elements to choose from a given dimension. `step` (sometimes called "stride") can be any integer, but
    * `0`. `begin` is an integer which represents the index of the first value to select, while `end` represents the
    * index of the last value to select (exclusive). The number of values selected in each dimension is
    * `end - begin` if `step > 0` and `begin - end` if `step < 0`. `begin` and `end` can be negative, where `-1`
    * corresponds to the last element, `-2` to the second to last, etc. `beginMask` controls whether to replace the
    * explicitly provided `begin` with an implicit effective value of: `0` if `step > 0`, and `-1` if `step < 0`.
    * `endMask` is analogous, but produces the number required to create the largest open interval. There is
    * currently no way to create begin masks and end masks in the Scala Indexer API. Values of `0` and `-1` should
    * instead be appropriately used for the `begin` value. The `endMask` functionality is not currently supported at
    * all since `foo(0 :: )` should return all elements of `foo`, whereas `foo(0 :: -1)` will return all except the
    * last one.
    *
    * Requirements:
    *
    *   - `0 != strides(i),` for `i` in `[0, m)` (i.e., no stride should be equal to `0`).
    *   - `ellipsisMask` must be a power of two (i.e., only one ellipsis used).
    *
    * Each conceptual range specification is encoded in the op's arguments. The encoding is best understood by
    * considering a non-trivial example. In particular:
    *
    * {{{
    *   // 'foo' is a tensor with shape '[5, 5, 5, 5, 5, 5]'
    *   foo(1, 2 :: 4, NewAxis, ---, 0 :: -1 :: -3, ::) will be encoded as:
    *   begin = [1, 2, x, x, 0, x] // Where "x" denotes that this value is ignored (we usually simply set it to 0)
    *   end = [2, 4, x, x, -3, x]
    *   strides = [1, 1, x, x, -1, 1]
    *   beginMask = 1 << 4 | 1 << 5 = 48
    *   endMask = 1 << 5 = 32
    *   ellipsisMask = 1 << 3 = 8
    *   newAxisMask = 1 << 2 = 4
    *   shrinkAxisMask = 1 << 0 = 1
    *   // The final shape of the slice becomes '[2, 1, 5, 5, 2, 5]'
    * }}}
    *
    * Let us walk step by step through each argument specification in the example slice:
    *
    *   1. The first argument is turned into `begin = 1`, `end = begin + 1 = 2`, `strides = 1`, and the first bit of
    * `shrinkAxisMask` set to `1` (i.e., `shrinkAxisMask |= 1 << 0`). Setting the bit of `shrinkAxisMask` to `1`
    * makes sure this argument is treated differently than `1 :: 2`, which would not shrink the corresponding axis.
    *   2. The second argument contributes `2` to `begin`, `4` to `end`, and `1` to `strides`. All masks have zero bits
    * contributed.
    *   3. The third argument sets the third bit of `newAxisMask` to `1` (i.e., `newAxisMask |= 1 << 2`).
    *   4. The fourth argument sets the fourth bit of `ellipsisMask` to `1` (i.e., `ellipsisMask |= 1 << 3`).
    *   5. The fifth argument contributes `0` to `begin`, `-3` to `end`, and `-1` to `strides`. It shows the use of
    * negative indices. A negative index `i` associated with a dimension that has size `s` is converted to a
    * positive index `s + i`. So `-1` becomes `s - 1` (i.e., the last element index). This conversion is done
    * internally and so `begin`, `end`, and `strides` are allowed to have negative values.
    *   6. The sixth argument indicates that the entire contents of the corresponding dimension are selected. It sets
    * the sixth bit of `beginMask` and `endMask` to `1` (i.e., `beginMask |= 1 << 6` and `endMask |= 1 << 6`).
    *
    * @param  input          Tensor to slice.
    * @param  begin          One-dimensional integer tensor. `begin(i)` specifies the begin offset into the `i`th range
    *                        specification. The exact dimension this corresponds to will be determined by context.
    *                        Out-of-bounds values will be silently clamped. If the `i`th bit of `beginMask` is `1`, then
    *                        `begin(i)` is ignored and the full range of the appropriate dimension is used instead.
    *                        Negative values causes indexing to start from the highest element.
    * @param  end            One-dimensional integer tensor. `end(i)` is like `begin(i)` with the exception that it
    *                        determines the end offset into the `i`th range specification, and that `endMask` is used to
    *                        determine full ranges.
    * @param  strides        One-dimensional integer tensor. `strides(i)` specifies the increment in the `i`th range
    *                        specification after extracting a given element. Negative indices will reverse the original
    *                        order. Out-of-bounds values are clamped to `[0, shape(i)) if slice(i) > 0` or
    *                        `[-1, shape(i) - 1] if slice(i) < 0`.
    * @param  beginMask      Integer value representing a bitmask where bit `i` being `1` means to ignore the begin
    *                        value and instead use the largest interval possible. At runtime `begin(i)` will be replaced
    *                        with `[0, shape(i) - 1) if stride(i) > 0` or `[-1, shape(i) - 1]` if `stride(i) < 0`.
    * @param  endMask        Integer value analogous to `beginMask`, but for specifying the end offset of the slice.
    * @param  ellipsisMask   Integer value representing a bitmask where bit `i` being `1` means that the `i`th position
    *                        is actually an ellipsis. At most one bit can be `1`. If `ellipsisMask == 0`, then an
    *                        implicit ellipsis mask with value `1 << (m + 1)` is provided. This means that
    *                        `foo(3 :: 5) == foo(3 :: 5, ---)`. An ellipsis implicitly creates as many range
    *                        specifications as necessary to fully specify the sliced range for every dimension. For
    *                        example, for a 4-dimensional tensor `foo` the slice `foo(2, ---, 5 :: 8)` implies
    *                        `foo(2, ::, ::, 5 :: 8)`.
    * @param  newAxisMask    Integer value representing a bitmask where bit `i` being `1` means that the `i`th range
    *                        specification creates a new dimension with size `1`. For example,
    *                        `foo(0 :: 4, NewAxis, 0 :: 2)` will produce a tensor with shape `[4, 1, 2]`.
    * @param  shrinkAxisMask Integer value representing a bitmask where bit `i` being `1` means that the `i`th range
    *                        specification should shrink the dimensionality. `begin` and `end` must imply a slice of
    *                        size `1` in the dimension. For example, in `foo(0 :: 4, 3, 0 :: 2)` would result in a
    *                        tensor with shape `[4, 2]`.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def stridedSlice(
      input: Op.Output, begin: Op.Output, end: Op.Output, strides: Op.Output = null, beginMask: Int = 0,
      endMask: Int = 0, ellipsisMask: Int = 0, newAxisMask: Int = 0, shrinkAxisMask: Int = 0,
      name: String = "StridedSlice"): Op.Output = {
    Op.Builder(opType = "StridedSlice", name = name)
        .addInput(input)
        .addInput(begin)
        .addInput(end)
        .addInput(if (strides != null) onesLike(begin, begin.dataType) else strides)
        .setAttribute("begin_mask", beginMask)
        .setAttribute("end_mask", endMask)
        .setAttribute("ellipsis_mask", ellipsisMask)
        .setAttribute("new_axis_mask", newAxisMask)
        .setAttribute("shrink_axis_mask", shrinkAxisMask)
        .build().outputs(0)
  }

  // TODO: Add support for the "stridedSliceAssign" op.

  //endregion Slice Ops

  /** Creates an op that checks a tensor for `NaN` and `Inf` values.
    *
    * When run, reports an `InvalidArgument` error if `input` has any values that are not-a-number (`NaN`) or infinity
    * (`Inf`). Otherwise, it acts as an identity op and passes `input` to the output, as-is.
    *
    * @param  input   Input tensor.
    * @param  message Prefix to print for the error message.
    * @param  name    Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def checkNumerics(input: Op.Output, message: String = "", name: String = "CheckNumerics"): Op.Output = {
    Op.Builder(opType = "CheckNumerics", name = name)
        .addInput(input)
        .setAttribute("message", message)
        .build().outputs(0)
  }

  /** Creates an op that computes the Levenshtein distance between sequences.
    *
    * The op takes variable-length sequences (`hypothesis` and `truth`), each provided as a `SparseTensor`, and computes
    * the Levenshtein distance between them. The op can also normalize the edit distance using the length of `truth` by
    * setting `normalize` to `true`.
    *
    * For example:
    * {{{
    *   // 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
    *   //   [0, 0] = ["a"]
    *   //   [0, 1] = ["b"]
    *   val hypothesis = SparseOutput(Tensor(Tensor(0, 0, 0), Tensor(1, 0, 0)), Tensor("a", "b"), Tensor(2, 1, 1))
    *   // 'truth' is a tensor of shape `[2, 2]` with variable-length values:
    *   //   [0, 0] = []
    *   //   [0, 1] = ["a"]
    *   //   [1, 0] = ["b", "c"]
    *   //   [1, 1] = ["a"]
    *   val truth = tf.SparseOutput(
    *       Tensor(Tensor(0, 1, 0), Tensor(1, 0, 0), Tensor(1, 0, 1), Tensor(1, 1, 0)),
    *       Tensor("a", "b", "c", "a"),
    *       Tensor(2, 2, 2))
    *   val normalize = true
    *
    *   // 'output' is a tensor of shape `[2, 2]` with edit distances normalized by the `truth` lengths, and contains
    *   // the values `[[inf, 1.0], [0.5, 1.0]]`. The reason behind each value is:
    *   //   - (0, 0): no truth,
    *   //   - (0, 1): no hypothesis,
    *   //   - (1, 0): addition,
    *   //   - (1, 1): no hypothesis.
    *   val output = tf.editDistance(hypothesis, truth, normalize)
    * }}}
    *
    * @param  hypothesis Sparse tensor that contains the hypothesis sequences.
    * @param  truth      Sparse tensor that contains the truth sequences.
    * @param  normalize  Optional boolean value indicating whether to normalize the Levenshtein distance by the length
    *                    of `truth`.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def editDistance(
      hypothesis: Op.SparseOutput, truth: Op.SparseOutput, normalize: Boolean = true,
      name: String = "EditDistance"): Op.Output = {
    Op.Builder(opType = "EditDistance", name = name)
        .addInput(hypothesis.indices)
        .addInput(hypothesis.values)
        .addInput(hypothesis.denseShape)
        .addInput(truth.indices)
        .addInput(truth.values)
        .addInput(truth.denseShape)
        .setAttribute("normalize", normalize)
        .build().outputs(0)
  }

  /** Creates an op that returns a one-hot tensor.
    *
    * The locations represented by indices in `indices` take value `onValue`, while all other locations take value
    * `offValue`. `onValue` and `offValue` must have matching data types. If `dataType` is also provided, they must be
    * the same data type as specified by `dataType`.
    *
    * If the input `indices` is rank `N`, the output will have rank `N+1`. The new axis is created at dimension `axis`
    * (which defaults to the last axis).
    *
    * If `indices` is a scalar the output shape will be a vector of length `depth`.
    *
    * If `indices` is a vector of length `features`, the output shape will be:
    *   - `[features, depth]`, if `axis == -1`, and
    *   - `[depth, features]`, if `axis == 0`.
    *
    * If `indices` is a matrix (batch) with shape `[batch, features]`, the output shape will be:
    *   - `[batch, features, depth]`, if `axis == -1`,
    *   - `[batch, depth, features]`, if `axis == 1`, and
    *   - `[depth, batch, features]`, if `axis == 0`.
    *
    * If `dataType` is not provided, the function will attempt to assume the data type of `onValue` or `offValue`, if
    * one or both are passed in. If none of `onValue`, `offValue`, or `dataType` are provided, `dataType` will default
    * to the `FLOAT32` data type.
    *
    * Note: If a non-numeric data type output is desired (e.g., `STRING` or `BOOLEAN`), both `onValue` and `offValue`
    * **must** be provided to `oneHot`.
    *
    * For example:
    * {{{
    *   // 'indices' = [0, 2, -1, 1]
    *   // 'depth' = 3
    *   // 'onValue' = 5.0
    *   // 'offValue' = 0.0
    *   // 'axis' = -1
    *   // The output tensor has shape [4, 3]
    *   tf.oneHot(indices, depth, onValue, offValue, axis) ==>
    *     [[5.0, 0.0, 0.0],  // oneHot(0)
    *      [0.0, 0.0, 5.0],  // oneHot(2)
    *      [0.0, 0.0, 0.0],  // oneHot(-1)
    *      [0.0, 5.0, 0.0]]  // oneHot(1)
    *
    *   // 'indices' = [[0, 2], [1, -1]]
    *   // 'depth' = 3
    *   // 'onValue' = 1.0
    *   // 'offValue' = 0.0
    *   // 'axis' = -1
    *   // The output tensor has shape [2, 2, 3]
    *   tf.oneHot(indices, depth, onValue, offValue, axis) ==>
    *     [[[1.0, 0.0, 0.0],   // oneHot(0)
    *       [0.0, 0.0, 1.0]],  // oneHot(2)
    *      [[0.0, 1.0, 0.0],   // oneHot(1)
    *       [0.0, 0.0, 0.0]]]  // oneHot(-1)
    * }}}
    *
    * @param  indices  Tensor containing the indices for the "on" values.
    * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
    * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
    *                  Defaults to the value `1` with type `dataType`.
    * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
    *                  Defaults to the value `0` with type `dataType`.
    * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
    * @param  dataType Data type of the output tensor. If not provided, the function will attempt to assume the data
    *                  type of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`,
    *                  or `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If the `onValue` data type, the `offValue` data type, and `dataType` are
    *                                  incompatible, or if `indices` or `depth` have invalid data types or shapes.
    */
  @throws[IllegalArgumentException]
  def oneHot(
      indices: Op.Output, depth: Op.Output, onValue: Op.Output = null, offValue: Op.Output = null, axis: Int = -1,
      dataType: DataType = null, name: String = "OneHot"): Op.Output = {
    if (indices.dataType != UINT8 && indices.dataType != INT32 && indices.dataType != INT64)
      throw new IllegalArgumentException(s"The indices data type (${indices.dataType}) must be UINT8, INT32, or INT64.")
    if (depth.dataType != INT32)
      throw new IllegalArgumentException(s"The depth data type (${depth.dataType}) must be INT32.")
    if (depth.shape.rank > 0)
      throw new IllegalArgumentException(s"The depth (shape = ${depth.shape}) must be a scalar tensor.")
    if (onValue != null && onValue.shape.rank > 0)
      throw new IllegalArgumentException(s"The 'on' value (shape = ${onValue.shape}) must be a scalar tensor.")
    if (offValue != null && offValue.shape.rank > 0)
      throw new IllegalArgumentException(s"The 'off' value (shape = ${offValue.shape}) must be a scalar tensor.")
    val inferredDataType = {
      if (dataType != null) {
        dataType
      } else {
        if (onValue != null)
          onValue.dataType
        else if (offValue != null)
          offValue.dataType
        else
          FLOAT32
      }
    }
    if (onValue != null && offValue != null && onValue.dataType != offValue.dataType)
      throw new IllegalArgumentException(
        s"The provided on value data type (${onValue.dataType}) must match " +
            s"the provided off value data type (${offValue.dataType}).")
    Op.createWithNameScope(name, Set(indices.op, depth.op, onValue.op, offValue.op)) {
      val actualOnValue = if (onValue != null) onValue else constant(1, inferredDataType)
      val actualOffValue = if (offValue != null) offValue else constant(1, inferredDataType)
      if (actualOnValue.dataType != inferredDataType)
        throw new IllegalArgumentException(
          s"On value data type (${actualOnValue.dataType}) must match the data type $inferredDataType.")
      if (actualOffValue.dataType != inferredDataType)
        throw new IllegalArgumentException(
          s"Off value data type (${actualOffValue.dataType}) must match the data type $inferredDataType.")
      Op.Builder(opType = "OneHot", name = Op.convertNameScopeToName(Op.currentNameScope))
          .addInput(indices)
          .addInput(depth)
          .addInput(actualOnValue)
          .addInput(actualOffValue)
          .setAttribute("axis", axis)
          .build().outputs(0)
    }
  }

  // TODO: Add support for all the quantization ops.

  //region Broadcasting Ops

  // TODO: Add support for "broadcastShape" (static). Implement the main method in the "Shape" object.

  /** Creates an op that returns the broadcasted dynamic shape between two provided shapes, corresponding to the shapes
    * of the two arguments provided to an op that supports broadcasting.
    *
    * @param  shape1 One-dimensional integer tensor representing the shape of the first argument.
    * @param  shape2 One-dimensional integer tensor representing the shape of the first argument.
    * @param  name   Name for the created op.
    * @return Created op output, which is a one-dimensional integer tensor representing the broadcasted shape.
    */
  def broadcastShapeDynamic(shape1: Op.Output, shape2: Op.Output, name: String = "BroadcastShape"): Op.Output = {
    if (shape1.dataType != INT32 && shape1.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${shape1.dataType}' is not supported for the shape broadcasting op inputs. " +
            s"Only 'INT32' and 'INT64' are supported.")
    if (shape1.shape.rank != 1 && shape1.shape.rank != -1)
      throw InvalidShapeException(
        s"Shape '${shape1.shape}' is not supported for the shape broadcasting op inputs. " +
            s"Only one-dimensional tensors are supported.")
    if (shape2.dataType != INT32 && shape2.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${shape2.dataType}' is not supported for the shape broadcasting op inputs. " +
            s"Only 'INT32' and 'INT64' are supported.")
    if (shape2.shape.rank != 1 && shape2.shape.rank != -1)
      throw InvalidShapeException(
        s"Shape '${shape2.shape}' is not supported for the shape broadcasting op inputs. " +
            s"Only one-dimensional tensors are supported.")
    Op.Builder(opType = "BroadcastArgs", name = name)
        .addInput(shape1)
        .addInput(shape2)
        .build().outputs(0)
  }

  //endregion Broadcasting Ops

  //region Gradients Ops

  /** Creates an op that stops gradient execution, but otherwise acts as an identity op.
    *
    * When executed in a graph, this op outputs its input tensor as-is.
    *
    * When building ops to compute gradients, this op prevents the contribution of its inputs to be taken into account.
    * Normally, the gradient generator adds ops to a graph to compute the derivatives of a specified 'loss' by
    * recursively finding out inputs that contributed to its computation. If you insert this op in the graph its inputs
    * are masked from the gradient generator. They are not taken into account for computing gradients.
    *
    * This is useful any time you want to compute a value with TensorFlow but need to pretend that the value was a
    * constant. Some examples include:
    *
    *   - The *EM* algorithm where the *M-step* should not involve backpropagation through the output of the *E-step*.
    *   - Contrastive divergence training of Boltzmann machines where, when differentiating the energy function, the
    * training must not backpropagate through the graph that generated the samples from the model.
    *   - Adversarial training, where no backprop should happen through the adversarial example generation process.
    *
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def stopGradient(input: Op.Output, name: String = "StopGradient"): Op.Output = {
    Op.Builder(opType = "StopGradient", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an identity op that triggers an error if a gradient is requested.
    *
    * When executed in a graph, this op outputs its input tensor as-is.
    *
    * When building ops to compute gradients, the TensorFlow gradient system ill return an error when trying to lookup
    * the gradient of this op, because no gradient must ever be registered for this function. This op exists to prevent
    * subtle bugs from silently returning unimplemented gradients in some corner cases.
    *
    * @param  input   Input tensor.
    * @param  message Message to print along with the error.
    * @param  name    Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def preventGradient(input: Op.Output, message: String = "", name: String = "PreventGradient"): Op.Output = {
    Op.Builder(opType = "PreventGradient", name = name)
        .addInput(input)
        .setAttribute("message", message)
        .build().outputs(0)
  }

  //endregion Gradients Ops
}

object Basic extends Basic {
  private[api] object Gradients {
    GradientsRegistry.registerNonDifferentiable("Const")
    GradientsRegistry.registerNonDifferentiable("ZerosLike")
    GradientsRegistry.registerNonDifferentiable("OnesLike")
    GradientsRegistry.registerNonDifferentiable("Rank")
    GradientsRegistry.registerNonDifferentiable("Size")
    GradientsRegistry.registerNonDifferentiable("Shape")
    GradientsRegistry.registerNonDifferentiable("ShapeN")
    GradientsRegistry.registerNonDifferentiable("ConcatOffset") // TODO: [OP]
    GradientsRegistry.registerNonDifferentiable("InvertPermutation")
    GradientsRegistry.registerNonDifferentiable("OneHot") // TODO: [OP]
    GradientsRegistry.registerNonDifferentiable("EditDistance") // TODO: [OP]
    GradientsRegistry.registerNonDifferentiable("BroadcastGradientArgs") // TODO: [OP]
    GradientsRegistry.registerNonDifferentiable("StopGradient")

    GradientsRegistry.register("Pack", stackGradient)
    GradientsRegistry.register("Reshape", reshapeGradient)

    private[this] def stackGradient(op: Op, outputGradients: Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
      unstack(
        input = outputGradients.head, number = op.longAttribute("N").toInt,
        axis = op.longAttribute("axis").toInt).toSeq
    }

    private[this] def reshapeGradient(op: Op, outputGradients: Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
      Seq[Op.OutputLike](reshape(outputGradients.head, shape(op.inputs(0))), null)
    }
  }
}
