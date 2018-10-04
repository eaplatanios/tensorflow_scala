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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.NN.CNNDataFormat
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.{executionContext, Tensor}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.InvalidArgumentException
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import org.tensorflow.framework.AttrValue

import scala.language.postfixOps

/** Contains functions for constructing ops related to basic tensor manipulation.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Basic {
  //region Tensor Creation Ops

  /** $OpDocBasicConstant
    *
    * @group BasicOps
    * @param  tensor Constant value.
    * @param  shape  Shape of the resulting tensor.
    * @param  name   Name for the created op.
    * @return Created op output.
    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
    *                               the provided `shape`.
    */
  @throws[InvalidShapeException]
  def constant[T](
      tensor: Tensor[T],
      shape: Shape = null,
      name: String = "Constant"
  ): Output[T] = {
    val inferredShape = if (shape == null) tensor.shape else shape
    val constantTensor = AttrValue.newBuilder()
        .setTensor(Tensor.makeProto[T](tensor, inferredShape))
        .build()
    Op.Builder[Unit, Output[T]](
      opType = "Const",
      name = name,
      input = ()
    ).setAttribute("value", constantTensor)
        .setAttribute("dtype", tensor.dataType)
        .build().output
  }

  /** $OpDocBasicGuaranteeConstant
    *
    * @group BasicOps
    * @param  input Input tensor to guarantee that is constant.
    * @param  name  Name for the created op.
    * @return Created op output which is equal to the input tensor.
    */
  def guaranteeConstant[T](
      input: Output[T],
      name: String = "GuaranteeConstant"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "GuaranteeConst",
      name = name,
      input = input
    ).setGradientFn(identityGradient)
        .build().output
  }

  /** $OpDocBasicImmutableConstant
    *
    * @group BasicOps
    * @param  dataType         Data type of the resulting tensor.
    * @param  shape            Shape of the resulting tensor.
    * @param  memoryRegionName Name of the read-only memory region used by the tensor. Please refer to the C++
    *                          `NewReadOnlyMemoryRegionFromFile` function in `tensorflow::Env`.
    * @param  name             Name for the created op.
    * @return Created op output.
    */
  private[ops] def immutableConstant[T](
      dataType: DataType[T],
      shape: Shape,
      memoryRegionName: String,
      name: String = "ImmutableConstant"
  ): Output[T] = {
    Op.Builder[Unit, Output[T]](
      opType = "ImmutableConst",
      name = name,
      input = ()
    ).setAttribute("dtype", dataType)
        .setAttribute("shape", shape)
        .setAttribute("memory_region_name", memoryRegionName)
        .build().output
  }

  /** $OpDocBasicZeros
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def zeros[T, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Output[I],
      name: String = "Zeros"
  ): Output[T] = {
    // TODO: [OPS] Do we need this import?
    import dataType.evSupportedType
    fill(dataType, shape)(DataType.zero[T], name = name)
  }

  /** $OpDocBasicZerosLike
    *
    * @group BasicOps
    * @param  input    Input tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def zerosLike[T](
      input: Output[T],
      optimize: Boolean = true,
      name: String = "ZerosLike"
  ): Output[T] = {
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a zeros tensor independent of the value of 'tensor' since the shape is known statically.
      zeros(input.dataType, input.shape, name)
    } else {
      Op.Builder[Output[T], Output[T]](
        opType = "ZerosLike",
        name = name,
        input = input
      ).build().output
    }
  }

  /** $OpDocBasicOnes
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def ones[T, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Output[I],
      name: String = "Ones"
  ): Output[T] = {
    // TODO: [OPS] Do we need this import?
    import dataType.evSupportedType
    fill(dataType, shape)(DataType.one[T], name = name)
  }

  /** $OpDocBasicOnesLike
    *
    * @group BasicOps
    * @param  input    Input tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def onesLike[T](
      input: Output[T],
      optimize: Boolean = true,
      name: String = "OnesLike"
  ): Output[T] = {
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a ones tensor independent of the value of 'tensor' since the shape is known statically.
      ones(input.dataType, input.shape, name)
    } else {
      Op.Builder[Output[T], Output[T]](
        opType = "OnesLike",
        name = name,
        input = input
      ).build().output
    }
  }

  /** $OpDocBasicFill
    *
    * @group BasicOps
    * @param  dataType Optional data type for the created tensor.
    * @param  shape    Shape of the output tensor.
    * @param  value    Value to fill the output tensor.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def fill[T, V, I: IsInt32OrInt64](
      dataType: DataType[T],
      shape: Output[I]
  )(
      value: Output[V],
      name: String = "Fill"
  ): Output[T] = {
    val castedValue = value.castTo(dataType)
    Op.Builder[(Output[I], Output[T]), Output[T]](
      opType = "Fill",
      name = name,
      input = (shape, castedValue)
    ).setGradientFn(fillGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def fillGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[I], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[I], Output[T]) = {

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = new IsNumeric[T] {}

    (null, Math.sum(outputGradient))
  }

  /** $OpDocBasicPlaceholder
    *
    * @group BasicOps
    * @param  dataType Data type of the elements in the tensor that will be fed.
    * @param  shape    Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *                  completely unknown.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def placeholder[T](
      dataType: DataType[T],
      shape: Shape = null,
      name: String = "Placeholder"
  ): Output[T] = {
    val opBuilder = Op.Builder[Unit, Output[T]](
      opType = "Placeholder",
      name = name,
      input = ()
    ).setAttribute("dtype", dataType)
    if (shape != null && shape.rank != -1)
      opBuilder.setAttribute("shape", shape)
    opBuilder.build().output
  }

  /** $OpDocBasicPlaceholderWithDefault
    *
    * @group BasicOps
    * @param  default Default value to pass through when no input is fed for this placeholder.
    * @param  shape   Shape of the tensor that will be fed. The shape can be any partially-specified, or even completely
    *                 unknown.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def placeholderWithDefault[T](
      default: Output[T],
      shape: Shape,
      name: String = "PlaceholderWithDefault"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "PlaceholderWithDefault",
      name = name,
      input = default
    ).setAttribute("shape", shape)
        .setGradientFn(identityGradient)
        .build().output
  }

  /** $OpDocBasicSparsePlaceholder
    *
    * @group BasicOps
    * @param  dataType Data type of the elements in the tensor that will be fed.
    * @param  shape    Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *                  completely unknown. This represents the shape of the dense tensor that corresponds to the sparse
    *                  tensor that this placeholder refers to.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sparsePlaceholder[T](
      dataType: DataType[T],
      shape: Shape = null,
      name: String = "SparsePlaceholder"
  ): SparseOutput[T] = {
    val shapeWithDefault = {
      if (shape == null)
        placeholder(INT64, Shape(-1), name + "/Shape")
      else
        constant(shape.toTensor[Long])
    }
    SparseOutput[T](
      indices = placeholder(INT64, Shape(-1, -1), name + "/Indices"),
      values = placeholder(dataType, Shape(-1), name + "/Values"),
      denseShape = shapeWithDefault)
  }

  //endregion Tensor Creation Ops

  //region Tensor Shape Ops

  /** $OpDocBasicRank
    *
    * @group BasicOps
    * @param  input    Tensor whose rank to return.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  rank value that `input` has at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def rank[O <: OutputLike[_]](
      input: O,
      optimize: Boolean = true,
      name: String = "Rank"
  ): Output[Int] = {
    input match {
      case o: Output[_] =>
        val inputRank = o.rank
        if (optimize && inputRank != -1) {
          constant(Tensor.fill[Int](Shape())(inputRank), name = name)
        } else {
          Op.Builder[Output[Any], Output[Int]](
            opType = "Rank",
            name = name,
            input = o
          ).build().output
        }
      case o: OutputIndexedSlices[_] => size(o.denseShape, INT32, optimize = optimize, name = name)
      case o: SparseOutput[_] => size(o.denseShape, INT32, optimize = optimize, name = name)
    }
  }

  /** $OpDocBasicSize
    *
    * @group BasicOps
    * @param  input    Tensor whose size to return.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  number of elements provided by the shape of that `input` at graph creation time (instead of
    *                  execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def size[O <: OutputLike[_], R: IsInt32OrInt64](
      input: O,
      dataType: DataType[R],
      optimize: Boolean = true,
      name: String = "Size"
  ): Output[R] = {
    input match {
      case o: Output[_] =>
        val inputShape = o.shape
        if (optimize && inputShape.isFullyDefined) {
          constant(Tensor.fill[Long](Shape())(inputShape.numElements), name = name).castTo(dataType)
        } else if (optimize && inputShape.rank > -1 && inputShape.asArray.contains(0)) {
          constant(0L, name = name).castTo(dataType)
        } else {
          Op.Builder[Output[Any], Output[R]](
            opType = "Size",
            name = name,
            input = o
          ).setAttribute("out_type", dataType)
              .build().output
        }
      case o: OutputIndexedSlices[_] =>
        Op.nameScope(name) {
          Math.prod(o.denseShape, Seq(0)).castTo(dataType)
        }
      case o: SparseOutput[_] =>
        Op.nameScope(name) {
          Math.prod(o.denseShape, Seq(0)).castTo(dataType)
        }
    }
  }

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input    Tensor whose shape to return.
    * @param  dataType Data type for the resulting tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  shape of that `input` at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output, which is one-dimensional.
    */
  def shape[O <: OutputLike[_], T, I: IsInt32OrInt64](
      input: O,
      dataType: DataType[I],
      optimize: Boolean = true,
      name: String = "Shape"
  ): Output[I] = {
    input match {
      case o: Output[_] =>
        val inputShape = o.shape
        if (optimize && inputShape.isFullyDefined) {
          constant(inputShape.toTensor[Long], name = name).castTo(dataType)
        } else {
          Op.Builder[Output[Any], Output[I]](
            opType = "Shape",
            name = name,
            input = o
          ).setAttribute("out_type", dataType)
              .build().output
        }
      case o: OutputIndexedSlices[_] => o.denseShape.castTo(dataType)
      case o: SparseOutput[_] => o.denseShape.castTo(dataType)
    }
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs   Tensors whose shapes to return.
    * @param  dataType Data type for the resulting tensors.
    * @param  name     Name for the created op.
    * @return Created op outputs, all of which are one-dimensional.
    */
  def shapeN[T, I: IsInt32OrInt64](
      inputs: Seq[Output[T]],
      dataType: DataType[I],
      name: String = "ShapeN"
  ): Seq[Output[I]] = {
    Op.Builder[Seq[Output[T]], Seq[Output[I]]](
      opType = "ShapeN",
      name = name,
      input = inputs
    ).setAttribute("out_type", dataType)
        .build().output
  }

  //endregion Tensor Shape Ops

  //region Tensor Manipulation Ops

  /** $OpDocBasicIdentity
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def identity[T, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "Identity"
  ): OL[T] = {
    Op.nameScope(name) {
      input match {
        case i: Output[T] =>
          Op.Builder[Output[T], Output[T]](
            opType = "Identity",
            name = name,
            input = i
          ).setGradientFn(identityGradient)
              .build().output
        case i: OutputIndexedSlices[T] =>
          val indices = identity(i.indices, name = "IndicesIdentity")
          val values = identity(i.values, name = "ValuesIdentity")
          val denseShape = if (i.denseShape != null) identity(i.denseShape, name = "DenseShapeIdentity") else null
          OutputIndexedSlices[T](indices = indices, values = values, denseShape = denseShape)
        case i: SparseOutput[T] =>
          val indices = identity(i.indices, name = "IndicesIdentity")
          val values = identity(i.values, name = "ValuesIdentity")
          val denseShape = identity(i.denseShape, name = "DenseShapeIdentity")
          SparseOutput[T](indices = indices, values = values, denseShape = denseShape)
      }
    }.asInstanceOf[OL[T]]
  }

  protected def identityGradient[I](
      op: Op[I, I],
      outputGradient: I
  ): I = {
    outputGradient
  }

  // TODO: [BASIC] Add support for "identityN".

  /** $OpDocBasicExpandDims
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  axis  Dimension index at which to expand the shape of `input`.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def expandDims[T, I: IsInt32OrInt64](
      input: Output[T],
      axis: Output[I],
      name: String = "ExpandDims"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "ExpandDims",
      name = name,
      input = (input, axis)
    ).setGradientFn(expandDimsGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  /** $OpDocBasicSqueeze
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  axes  Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
    *               will be squeezed.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def squeeze[T](
      input: Output[T],
      axes: Seq[Int] = null,
      name: String = "Squeeze"
  ): Output[T] = {
    val builder = Op.Builder[Output[T], Output[T]](
      opType = "Squeeze",
      name = name,
      input = input)
    if (axes != null)
      builder.setAttribute("squeeze_dims", axes.map(_.asInstanceOf[Long]).toArray)
    builder
        .setGradientFn(squeezeGradient)
        .build().output
  }

  /** Reshapes the gradient to the shape of the original input. */
  protected def reshapeToInput[T](input: Output[T], gradient: Output[T]): Output[T] = {
    reshape(gradient, shape(input, INT64))
  }

  protected def expandDimsGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reshapeToInput(op.input._1, outputGradient), null)
  }

  protected def squeezeGradient[T](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    reshapeToInput(op.input, outputGradient)
  }

  /** $OpDocBasicStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @param  axis   Dimension along which to stack the input tensors.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def stack[T](
      inputs: Seq[Output[T]],
      axis: Int = 0,
      name: String = "Stack"
  ): Output[T] = {
    Op.Builder[Seq[Output[T]], Output[T]](
      opType = "Pack",
      name = name,
      input = inputs
    ).setAttribute("axis", axis)
        .setGradientFn(stackGradient)
        .build().output
  }

  protected def stackGradient[T](
      op: Op[Seq[Output[T]], Output[T]],
      outputGradient: Output[T]
  ): Seq[Output[T]] = {
    unstack(
      input = outputGradient, number = op.longAttribute("N").toInt,
      axis = op.longAttribute("axis").toInt)
  }

  /** $OpDocBasicParallelStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def parallelStack[T](
      inputs: Seq[Output[T]],
      name: String = "ParallelStack"
  ): Output[T] = {
    val inputsShape = inputs.head.shape
    inputs.tail.foreach(_.shape.assertIsCompatibleWith(inputsShape))
    val outputShape = Shape(inputs.length).concatenateWith(inputsShape)
    Op.Builder[Seq[Output[T]], Output[T]](
      opType = "ParallelConcat",
      name = name,
      input = inputs
    ).setAttribute("shape", outputShape)
        .build().output
  }

  /** $OpDocBasicUnstack
    *
    * @group BasicOps
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
  def unstack[T](
      input: Output[T],
      number: Int = -1,
      axis: Int = 0,
      name: String = "Unstack"
  ): Seq[Output[T]] = {
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
    Op.Builder[Output[T], Seq[Output[T]]](
      opType = "Unpack",
      name = name,
      input = input
    ).setAttribute("num", num)
        .setAttribute("axis", axis)
        .setGradientFn(unstackGradient)
        .build().output
  }

  protected def unstackGradient[T](
      op: Op[Output[T], Seq[Output[T]]],
      outputGradient: Seq[Output[T]]
  ): Output[T] = {
    stack(inputs = outputGradient.map(_.toOutput), axis = op.longAttribute("axis").toInt)
  }

  /** $OpDocBasicConcatenate
    *
    * @group BasicOps
    * @param  inputs Input tensors to be concatenated.
    * @param  axis   Dimension along which to concatenate the input tensors. As in Python, indexing for the axis is
    *                0-based. Positive axes in the range of `[0, rank(values))` refer to the `axis`-th dimension, and
    *                negative axes refer to the `axis + rank(inputs)`-th dimension.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def concatenate[T](
      inputs: Seq[Output[T]],
      axis: Output[Int] = 0,
      name: String = "Concatenate"
  ): Output[T] = {
    if (inputs.length == 1) {
      Op.nameScope(name)(identity(inputs.head))
    } else {
      Op.Builder[(Seq[Output[T]], Output[Int]), Output[T]](
        opType = "ConcatV2",
        name = name,
        input = (inputs, axis)
      ).setGradientFn(concatenateGradient)
          .build().output
    }
  }

  protected def concatenateGradient[T](
      op: Op[(Seq[Output[T]], Output[Int]), Output[T]],
      outputGradient: OutputLike[T]
  ): (Seq[OutputLike[T]], Output[Int]) = {
    if (op.input._1.size == 1) {
      // Degenerate concatenation.
      (Seq(outputGradient), null)
    } else {
      val inputValues = op.input._1
      val outputGradients = outputGradient match {
        case g: Output[T] =>
          var concatenationAxis = op.input._2
          Output.constantValue(concatenationAxis) match {
            case Some(axis) =>
              // If `concatenationAxis` is a constant defined in a different context, then we duplicate it in the
              // current context to avoid passing it through an `enter` node. This is a small optimization in general,
              // but it is required when compiling with XLA, as XLA needs the concatenation op input to be folded into
              // a constant.
              val gradientContext = ControlFlow.getOutputContext(outputGradient.op)
              val axisContext = ControlFlow.getOutputContext(concatenationAxis.op)
              if (axisContext != gradientContext) {
                concatenationAxis = constant(axis)
              }
            case None => ()
          }
          // Using modulus here for convenience since the 'concatenationAxis' value is already verified in the
          // concatenate op implementation to be within the allowed '[-rank, rank)' range.
          val nonNegativeConcatenationAxis = concatenationAxis % rank(inputValues(0))
          // Get the inputs' tensor shapes.
          val shapes = shapeN(inputValues.map(_.toOutput), INT32)
          // The magic number of '16' was found through benchmarking a range of sizes on CPUs and a Maxwell Titan X
          // GPU. A speedup was seen in a large majority of cases when switching implementations at N = 16, but it is
          // possible that there will be a small number of performance regressions.
          if (shapes.length > 16) {
            // Extract the size of each input along the concatenation axis.
            val sizes = squeeze(slice(
              stack(shapes, 1),
              stack[Int](Seq(nonNegativeConcatenationAxis, 0)),
              Tensor(1, -1)))
            split(g, sizes, nonNegativeConcatenationAxis)
          } else {
            val offset = concatenateOffset(nonNegativeConcatenationAxis, shapes)
            offset.zip(shapes).map(t => slice(g, t._1, t._2))
          }
        case g: OutputIndexedSlices[T] =>
          val concatenationAxis = op.input._2
          val staticConcatenationAxis = {
            val axis = Output.constantValue(concatenationAxis)
            if (axis.isEmpty)
              throw new IllegalArgumentException(
                "Can only compute 'OutputIndexedSlices' gradients for the concatenation op when the " +
                    "concatenation axis is statically-known.")
            val realNumericAxis = axis.get.scalar
            if (realNumericAxis < 0) {
              val staticRank = Output.constantValue(rank(inputValues(0)))
              if (staticRank.isEmpty)
                throw new IllegalArgumentException(
                  "Can only compute 'OutputIndexedSlices' gradients for the concatenation op when the " +
                      "first value rank is statically-known.")
              realNumericAxis % staticRank.get.scalar
            } else {
              realNumericAxis
            }
          }
          // Using modulus here for convenience since the 'concatenationAxis' value is already verified in the
          // concatenate op implementation to be within the allowed '[-rank, rank)' range.
          val nonNegativeConcatenationAxis = concatenationAxis % rank(inputValues(0))
          // Get the input tensor shapes.
          val shapes = inputValues.map(shape(_, INT64))
          if (staticConcatenationAxis > 0) {
            // 'nonNegativeConcatenationAxis' > 0. Each input gets OutputIndexedSlices gradients with all the indices,
            // but with the values sliced accordingly. This is like the Output case, except that shape(g.values)(0) is
            // not equal to shape(shapes(i))(0), since only a subset of the axis-0 values are stored.

            // The following creates variables for iteratively slicing a dense gradients tensor.
            // Since shape is 1-D, 'shapeOfShape' is a scalar containing the rank of the inputs.
            val shapeOfShape = shape(shapes(0), INT64)
            // Make a vector of length equal to the input rank, with 0's everywhere and 1 in the concatenation axis index.
            val zero = constant(Tensor(0))
            val mask = concatenate(Seq(
              fill(INT64, expandDims(nonNegativeConcatenationAxis, 0))(zero),
              constant(Tensor(1L)),
              fill(INT64, shapeOfShape - nonNegativeConcatenationAxis.castTo[Long] - ones(INT64, Shape()))(zero)
            ), zero)
            var begin = fill(INT64, shapeOfShape)(zero)
            shapes.map(shape => {
              val newValues = slice(g.values, begin, concatenate[Long](
                Seq(Tensor(-1L), slice(shape, 1, -1)), 0))
              begin = Math.add(begin, shape * mask)
              OutputIndexedSlices(g.indices, newValues, shape)
            })
          } else {
            // 'nonNegativeConcatenationAxis' == 0. Each input gets OutputIndexedSlices gradients but only for the
            // relevant indices.
            var start = zeros(INT64, Shape())
            var end = start
            shapes.map(shape => {
              val shapeConcatenationAxis = gather(shape, nonNegativeConcatenationAxis, axis = 0).castTo[Long]
              end = start + shapeConcatenationAxis
              // Compute the 1-D Output of indices relevant for this input.
              val indicesToSelect = squeeze(
                where(Math.logicalAnd(g.indices >= start, g.indices < end)), axes = Seq(1))
              val newIndices = gather(g.indices, indicesToSelect, axis = 0) - start
              val newValues = gather(g.values, indicesToSelect, axis = 0)
              start = end
              OutputIndexedSlices(newIndices, newValues, shape)
            })
          }
        case _ => throw new IllegalArgumentException(
          "Only 'Output' and 'OutputIndexedSlices' gradients are supported for the concatenation op.")
      }
      (outputGradients, null)
    }
  }

  /** $OpDocBasicConcatenateOffset
    *
    * @group BasicOps
    * @param  axis   Scalar representing the dimension along which to concatenate.
    * @param  shapes Sequence of `N` vectors representing the shapes of the tensors being concatenated.
    * @param  name   Name for the created op.
    * @return Sequence of `N` vectors representing the starting offset of the input tensors within the concatenated
    *         output.
    */
  private[ops] def concatenateOffset(
      axis: Output[Int],
      shapes: Seq[Output[Int]],
      name: String = "ConcatenateOffset"
  ): Seq[Output[Int]] = {
    Op.Builder[(Output[Int], Seq[Output[Int]]), Seq[Output[Int]]](
      opType = "ConcatOffset",
      name = name,
      input = (axis, shapes)
    ).build().output
  }

  /** $OpDocBasicSplitEvenly
    *
    * @group BasicOps
    * @param  input     Input tensor to split.
    * @param  numSplits Number of splits to obtain along the `axis` dimension.
    * @param  axis      Dimension along which to split the input tensor.
    * @param  name      Name for the created op.
    * @return Created op outputs.
    */
  def splitEvenly[T](
      input: Output[T],
      numSplits: Int,
      axis: Output[Int] = 0,
      name: String = "Split"
  ): Seq[Output[T]] = {
    Op.Builder[(Output[Int], Output[T]), Seq[Output[T]]](
      opType = "Split",
      name = name,
      input = (axis, input)
    ).setAttribute("num_split", numSplits)
        .setGradientFn(splitEvenlyGradient)
        .build().output
  }

  protected def splitEvenlyGradient[T](
      op: Op[(Output[Int], Output[T]), Seq[Output[T]]],
      outputGradient: Seq[Output[T]]
  ): (Output[Int], Output[T]) = {
    (null, concatenate(outputGradient.map(_.toOutput), axis = op.input._1))
  }

  /** $OpDocBasicSplit
    *
    * @group BasicOps
    * @param  input      Input tensor to split.
    * @param  splitSizes Sizes for the splits to obtain.
    * @param  axis       Dimension along which to split the input tensor.
    * @param  name       Name for the created op.
    * @return Created op outputs.
    */
  def split[T, I: IsInt32OrInt64](
      input: Output[T],
      splitSizes: Output[I],
      axis: Output[Int] = 0,
      name: String = "Split"
  ): Seq[Output[T]] = {
    val splitSizesShape = splitSizes.shape
    if (splitSizesShape == Shape.unknown())
      throw InvalidArgumentException(s"Cannot infer the number of splits from the shape '$splitSizesShape'.")
    Op.Builder[(Output[T], Output[I], Output[Int]), Seq[Output[T]]](
      opType = "SplitV",
      name = name,
      input = (input, splitSizes, axis)
    ).setAttribute("num_split", splitSizesShape(0))
        .setGradientFn(splitGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def splitGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I], Output[Int]), Seq[Output[T]]],
      outputGradient: Seq[Output[T]]
  ): (Output[T], Output[I], Output[Int]) = {
    (concatenate(outputGradient.map(_.toOutput), axis = op.input._3), null, null)
  }

  /** $OpDocBasicTile
    *
    * @group BasicOps
    * @param  input     Tensor to tile.
    * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
    *                   of `input`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def tile[T, I: IsInt32OrInt64](
      input: Output[T],
      multiples: Output[I],
      name: String = "Tile"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Tile",
      name = name,
      input = (input, multiples)
    ).setGradientFn(tileGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def tileGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: OutputLike[T]
  ): (Output[T], Output[I]) = {
    val inputShape = shape(op.input._1, op.input._2.dataType)
    // We interleave 'multiples' and 'inputShape' to get 'splitShape', reshape the output gradient to 'splitShape',
    // and reduce along all even dimensions (the tiled dimensions) to get the result with shape 'inputShape'.
    // For example:
    //   inputShape = [20, 30, 40]
    //   multiples = [2, 3, 4]
    //   splitShape = [2, 20, 3, 30, 4, 40]
    //   axes = [0, 2, 4]
    val splitShape = reshape(transpose(stack(Seq(op.input._2, inputShape))), Shape(-1))
    val axes = Math.range(0, size(splitShape, INT32), 2)

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = new IsNumeric[T] {}

    // Sum reduces grad along the first dimension for indexed slices.
    val (gradient, processedSplitShape) = outputGradient match {
      case g: OutputIndexedSlices[T] =>
        val gradient = Math.unsortedSegmentSum(
          g.values,
          Math.mod(g.indices.castTo(inputShape.dataType), inputShape(0)),
          inputShape(0))
        (gradient, concatenate(Seq(ones(splitShape.dataType, Shape()), splitShape(1 ::)), axis = 0))
      case g => (g, splitShape)
    }
    val inputGradient = Math.sum(reshape(gradient, processedSplitShape), axes)
    // Fix shape inference.
    inputGradient.setShape(op.input._1.shape)
    (inputGradient, null)
  }

  /** Padding mode. */
  sealed trait PaddingMode {
    /** Creates an op that pads a tensor with zeros.
      *
      * The op pads `input` with values specified by this padding mode, `mode`, according to the `paddings` you specify.
      *
      * `paddings` is an integer tensor with shape `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of
      * `input`, `paddings(D, 0)` indicates how many zeros to add before the contents of `input` in that dimension, and
      * `paddings(D, 1)` indicates how many zeros to add after the contents of `input` in that dimension.
      *
      * The padded size of each dimension `D` of the output is equal to
      * `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
      *
      * @param  input    Input tensor to be padded.
      * @param  paddings Paddings tensor.
      * @param  name     Name for the created op.
      * @return Created op output.
      */
    private[ops] def pad[T, I: IsInt32OrInt64](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T]

    /** Pads a tensor with zeros.
      *
      * The op pads `input` with values specified by this padding mode, `mode`, according to the `paddings` you specify.
      *
      * `paddings` is an integer tensor with shape `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of
      * `input`, `paddings(D, 0)` indicates how many zeros to add before the contents of `input` in that dimension, and
      * `paddings(D, 1)` indicates how many zeros to add after the contents of `input` in that dimension.
      *
      * The padded size of each dimension `D` of the output is equal to
      * `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
      *
      * @param  input    Input tensor to be padded.
      * @param  paddings Paddings tensor.
      * @return Result as a new tensor.
      */
    private[api] def pad[T, I: IsInt32OrInt64](
        input: Tensor[T],
        paddings: Tensor[I]
    ): Tensor[T]
  }

  private[ops] object PaddingMode {
    def fromString(name: String): PaddingMode = name match {
      case "CONSTANT" => ConstantPadding(Some(Tensor(0).reshape(Shape())))
      case "REFLECT" => ReflectivePadding
      case "SYMMETRIC" => SymmetricPadding
      case _ => throw new IllegalArgumentException(s"Invalid padding mode '$name' provided.")
    }
  }

  // TODO: [OPS] Add static data type information for constant padding.

  /** Constant padding mode.
    *
    * The op pads `input` with zeros according to the `paddings` you specify. `paddings` is an integer tensor with shape
    * `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of `input`, `paddings(D, 0)` indicates how many
    * zeros to add before the contents of `input` in that dimension, and `paddings(D, 1)` indicates how many zeros to
    * add after the contents of `input` in that dimension.
    *
    * The padded size of each dimension `D` of the output is equal to
    * `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
    *
    * For example:
    * {{{
    *   // 'input' = [[1, 2, 3], [4, 5, 6]]
    *   // 'paddings' = [[1, 1], [2, 2]]
    *   tf.pad(input, paddings, tf.ConstantPadding(0)) ==>
    *     [[0, 0, 0, 0, 0, 0, 0],
    *      [0, 0, 1, 2, 3, 0, 0],
    *      [0, 0, 4, 5, 6, 0, 0],
    *      [0, 0, 0, 0, 0, 0, 0]]
    * }}}
    */
  case class ConstantPadding(value: Option[Tensor[_]] = None) extends PaddingMode {
    override private[ops] def pad[T, I: IsInt32OrInt64](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      val constantValues = value.map(Basic.constant(_).castTo(input.dataType))
          .getOrElse(Basic.zeros(input.dataType, Shape()))
      Op.Builder[(Output[T], Output[I], Output[T]), Output[T]](
        opType = "PadV2",
        name = name,
        input = (input, paddings, constantValues)
      ).setGradientFn(padGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
          .build().output
    }

    override private[api] def pad[T, I: IsInt32OrInt64](
        input: Tensor[T],
        paddings: Tensor[I]
    ): Tensor[T] = {
      Tensor.fromNativeHandle[T](NativeTensorOpsBasic.padV2(
        executionContext.value.nativeHandle, input.nativeHandle, paddings.nativeHandle,
        value.map(_.castTo(input.dataType)).getOrElse(Tensor.zeros(input.dataType, Shape())).nativeHandle))
    }
  }

  /** Reflective padding mode.
    *
    * The op pads `input` with mirrored values according to the `paddings` you specify. `paddings` is an integer tensor
    * with shape `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of `input`, `paddings(D, 0)`
    * indicates how many values to add before the contents of `input` in that dimension, and `paddings(D, 1)` indicates
    * how many values to add after the contents of `input` in that dimension. Both `paddings(D, 0)` and `paddings(D, 1)`
    * must be no greater than `input.shape(D) - 1`.
    *
    * The padded size of each dimension `D` of the output is equal to
    * `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
    *
    * For example:
    * {{{
    *   // 'input' = [[1, 2, 3], [4, 5, 6]]
    *   // 'paddings' = [[1, 1], [2, 2]]
    *   tf.pad(input, paddings, tf.ReflectivePadding) ==>
    *     [[6, 5, 4, 5, 6, 5, 4],
    *      [3, 2, 1, 2, 3, 2, 1],
    *      [6, 5, 4, 5, 6, 5, 4],
    *      [3, 2, 1, 2, 3, 2, 1]]
    * }}}
    */
  object ReflectivePadding extends PaddingMode {
    override private[ops] def pad[T, I: IsInt32OrInt64](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "MirrorPad",
        name = name,
        input = (input, paddings)
      ).setAttribute("mode", "REFLECT")
          .setGradientFn(mirrorPadGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
          .build().output
    }

    override private[api] def pad[T, I: IsInt32OrInt64](
        input: Tensor[T],
        paddings: Tensor[I]
    ): Tensor[T] = {
      Tensor.fromNativeHandle[T](NativeTensorOpsBasic.mirrorPad(
        executionContext.value.nativeHandle, input.nativeHandle, paddings.nativeHandle,
        "REFLECT".getBytes()))
    }
  }

  /** Symmetric padding mode.
    *
    * The op pads `input` with mirrored values according to the `paddings` you specify. `paddings` is an integer tensor
    * with shape `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of `input`, `paddings(D, 0)`
    * indicates how many values to add before the contents of `input` in that dimension, and `paddings(D, 1)` indicates
    * how many values to add after the contents of `input` in that dimension. Both `paddings(D, 0)` and `paddings(D, 1)`
    * must be no greater than `input.shape(D)`.
    *
    * The padded size of each dimension `D` of the output is equal to
    * `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
    *
    * For example:
    * {{{
    *   // 'input' = [[1, 2, 3], [4, 5, 6]]
    *   // 'paddings' = [[1, 1], [2, 2]]
    *   tf.pad(input, paddings, tf.SymmetricPadding) ==>
    *     [[2, 1, 1, 2, 3, 3, 2],
    *      [2, 1, 1, 2, 3, 3, 2],
    *      [5, 4, 4, 5, 6, 6, 5],
    *      [5, 4, 4, 5, 6, 6, 5]]
    * }}}
    */
  object SymmetricPadding extends PaddingMode {
    override private[ops] def pad[T, I: IsInt32OrInt64](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "MirrorPad",
        name = name,
        input = (input, paddings)
      ).setAttribute("mode", "SYMMETRIC")
          .setGradientFn(mirrorPadGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
          .build().output
    }

    override private[api] def pad[T, I: IsInt32OrInt64](
        input: Tensor[T],
        paddings: Tensor[I]
    ): Tensor[T] = {
      Tensor.fromNativeHandle[T](NativeTensorOpsBasic.mirrorPad(
        executionContext.value.nativeHandle, input.nativeHandle, paddings.nativeHandle,
        "SYMMETRIC".getBytes()))
    }
  }

  /** $OpDocBasicPad
    *
    * @group BasicOps
    * @param  input    Input tensor to be padded.
    * @param  paddings Tensor containing the paddings.
    * @param  mode     Padding mode to use.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def pad[T, I: IsInt32OrInt64](
      input: Output[T],
      paddings: Output[I],
      mode: PaddingMode = ConstantPadding(Some(Tensor(0).reshape(Shape()))),
      name: String = "Pad"
  ): Output[T] = {
    mode.pad(input, paddings, name)
  }

  protected def padGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[T]) = {
    // Pad introduces values around the original tensor, and so the gradient function slices the original shape out of
    // the gradient.
    val x = op.input._1
    val a = op.input._2 // == [rank(x), 2]
    // Take a slice of 'a' (the 1st column: [rank(x), 1]).
    val padBefore = slice(a, Tensor(0, 0), stack[Int](Seq(rank(x), 1)))
    // Make it a one-dimensional tensor and return it.
    val xGradient = slice(outputGradient, reshape(padBefore, Shape(-1)), shape(x, a.dataType))
    (xGradient, null, null)
  }

  protected def mirrorPadGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val gradient = Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "MirrorPadGrad",
      name = "MirrorPadGradient",
      input = (outputGradient, op.input._2)
    ).setAttribute("mode", op.stringAttribute("mode"))
        .setGradientFn(mirrorPadHessian(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
    (gradient, null)
  }

  protected def mirrorPadHessian[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val mode = PaddingMode.fromString(op.stringAttribute("mode"))
    (pad(outputGradient, op.input._2, mode), null)
  }

  /** $OpDocBasicReshape
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  shape Shape of the output tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def reshape[T, I: IsInt32OrInt64](
      input: Output[T],
      shape: Output[I],
      name: String = "Reshape"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Reshape",
      name = name,
      input = (input, shape)
    ).setGradientFn(reshapeGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def reshapeGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reshape(outputGradient, shape(op.input._1, INT64)), null)
  }

  /** $OpDocBasicTranspose
    *
    * @group BasicOps
    * @param  input       Input tensor to transpose.
    * @param  permutation Permutation of the input tensor dimensions.
    * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def transpose[T, I: IsInt32OrInt64](
      input: Output[T],
      permutation: Output[I] = null,
      conjugate: Boolean = false,
      name: String = "Transpose"
  ): Output[T] = {
    val opType = if (conjugate && input.dataType.isComplex) "ConjugateTranspose" else "Transpose"
    if (permutation == null) {
      Op.createWith(nameScope = name) {
        val inputRank = rank(input)
        val reversePermutation = inputRank - constant(1) - Math.range(constant(0), inputRank, constant(1))
        val transposed = Op.Builder[(Output[T], Output[Int]), Output[T]](
          opType = opType,
          name = name,
          input = (input, reversePermutation)
        ).setGradientFn[(Output[T], Output[Int]), Output[T]]({
          if (opType == "Transpose")
            transposeGradient(_, _)(implicitly[IsInt32OrInt64[Int]])
          else
            conjugateTransposeGradient(_, _)(implicitly[IsInt32OrInt64[Int]])
        }).build()
        // Setting the shape explicitly because transpose is not handled by the shape function.
        val inputShape = transposed.input._1.shape
        if (inputShape != null && inputShape.rank != -1)
          transposed.output.setShape(Shape(inputShape.asArray.reverse: _*))
        transposed.output
      }
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = opType,
        name = name,
        input = (input, permutation)
      ).setGradientFn[(Output[T], Output[I]), Output[T]]({
        if (opType == "Transpose")
          transposeGradient(_, _)(implicitly[IsInt32OrInt64[I]])
        else
          conjugateTransposeGradient(_, _)(implicitly[IsInt32OrInt64[I]])
      }).build().output
    }
  }

  protected def transposeGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (transpose(outputGradient, invertPermutation(op.input._2)), null)
  }

  protected def conjugateTransposeGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (transpose(outputGradient, invertPermutation(op.input._2), conjugate = true), null)
  }

  /** $OpDocBasicMatrixTranspose
    *
    * @group BasicOps
    * @param  input     Input tensor to transpose.
    * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
    * @param  name      Name for the created op.
    * @return Created op output.
    * @throws InvalidShapeException If the input tensor has rank <= 2.
    */
  @throws[InvalidShapeException]
  def matrixTranspose[T](
      input: Output[T],
      conjugate: Boolean = false,
      name: String = "MatrixTranspose"
  ): Output[T] = {
    Op.nameScope(name) {
      // If we know the number of dimensions statically, we can do two things:
      //   1. Check that `input` is a (batch) matrix.
      //   2. Use a Scala array for the permutation. This preserves static shape
      //      information and avoids extra computation.
      val inputShape = input.shape
      val inputRank = inputShape.rank
      if (inputRank != -1) {
        val inputRank = input.rank
        if (inputRank < 2)
          throw InvalidShapeException(
            s"'input' should be a (batch) matrix, with rank > 2. Found shape '${input.shape}'.")
        val permutation = Range(0, inputRank - 2).toArray ++ Array(inputRank - 1, inputRank - 2)
        transpose(input, permutation, conjugate)
      } else {
        val inputRank = rank(input)
        val inputRankMinus1 = inputRank - constant(1)
        val inputRankMinus2 = inputRank - constant(2)
        val permutation = concatenate(Seq(
          Math.range(constant(0), inputRankMinus2, constant(1)),
          inputRankMinus1,
          inputRankMinus2))
        transpose(input, permutation, conjugate)
      }
    }
  }

  /** $OpDocBasicInvertPermutation
    *
    * @group BasicOps
    * @param  input One-dimensional input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def invertPermutation[I: IsInt32OrInt64](
      input: Output[I],
      name: String = "InvertPermutation"
  ): Output[I] = {
    Op.Builder[Output[I], Output[I]](
      opType = "InvertPermutation",
      name = name,
      input = input
    ).build().output
  }

  /** $OpDocBasicReverse
    *
    * @group BasicOps
    * @param  input Input tensor to reverse. It must have rank at most 8.
    * @param  axes  Dimensions of the input tensor to reverse.
    * @param  name  Name for the created op.
    * @return Created op output which has the same shape as `input`.
    */
  def reverse[T, I: IsInt32OrInt64](
      input: Output[T],
      axes: Output[I],
      name: String = "Reverse"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "ReverseV2",
      name = name,
      input = (input, if (axes.rank < 1) axes else axes(NewAxis))
    ).setGradientFn(reverseGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def reverseGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reverse(outputGradient, op.input._2), null)
  }

  /** $OpDocBasicReverseSequence
    *
    * @group BasicOps
    * @param  input           Input tensor to reverse.
    * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
    *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
    * @param  sequenceAxis    Tensor dimension which is partially reversed.
    * @param  batchAxis       Tensor dimension along which the reversal is performed.
    * @param  name            Created op name.
    * @return Created op output which has the same shape as `input`.
    */
  def reverseSequence[T, I: IsInt32OrInt64](
      input: Output[T],
      sequenceLengths: Output[I],
      sequenceAxis: Int,
      batchAxis: Int = 0,
      name: String = "ReverseSequence"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "ReverseSequence",
      name = name,
      input = (input, sequenceLengths)
    ).setAttribute("seq_dim", sequenceAxis)
        .setAttribute("batch_dim", batchAxis)
        .setGradientFn(reverseSequenceGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def reverseSequenceGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reverseSequence(
      input = outputGradient,
      sequenceLengths = op.input._2,
      sequenceAxis = op.longAttribute("seq_dim").toInt,
      batchAxis = op.longAttribute("batch_dim").toInt), null)
  }

  /** $OpDocBasicSpaceToBatch
    *
    * @group BasicOps
    * @param  input     Tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  paddings  Two-dimensional tensor with shape `[2, 2]`, containing non-negative integers.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def spaceToBatch[T, I: IsInt32OrInt64](
      input: Output[T],
      blockSize: Int,
      paddings: Output[I],
      name: String = "SpaceToBatch"
  ): Output[T] = {
    val result = spaceToBatchND(input, constant(Tensor(blockSize, blockSize)), paddings, name)
    result.setShape(result.shape.withRank(4))
    result
  }

  /** $OpDocBasicSpaceToBatchND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape Tensor with shape `[M]` whose elements must all be `>= 1`.
    * @param  paddings   Tensor with shape `[M, 2]` whose elements must all be non-negative.
    *                    `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension `i + 1`, which
    *                    corresponds to spatial dimension `i`. It is required that `blockShape(i)` divides
    *                    `inputShape(i + 1) + padStart + padEnd`.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def spaceToBatchND[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      input: Output[T],
      blockShape: Output[I1],
      paddings: Output[I2],
      name: String = "SpaceToBatchND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "SpaceToBatchND",
      name = name,
      input = (input, blockShape, paddings)
    ).setGradientFn(spaceToBatchNDGradient(_, _)(implicitly[IsInt32OrInt64[I1]], implicitly[IsInt32OrInt64[I2]]))
        .build().output
  }

  protected def spaceToBatchNDGradient[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[I2]) = {
    (batchToSpaceND(outputGradient, op.input._2, op.input._3), null, null)
  }

  /** $OpDocBasicBatchToSpace
    *
    * @group BasicOps
    * @param  input     Tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  crops     Tensor with shape `[2, 2]` containing non-negative integers.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def batchToSpace[T, I: IsInt32OrInt64](
      input: Output[T],
      blockSize: Int,
      crops: Output[I],
      name: String = "BatchToSpace"
  ): Output[T] = {
    val result = batchToSpaceND(input, constant(Tensor(blockSize, blockSize)), crops, name)
    result.setShape(result.shape.withRank(4))
    result
  }

  /** $OpDocBasicBatchToSpaceND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape Tensor with shape `[M]` whose elements must all be `>= 1`.
    * @param  crops      Tensor with shape `[M, 2]` whose elements must all be non-negative.
    *                    `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input dimension `i + 1`,
    *                    which corresponds to spatial dimension `i`. It is required that
    *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def batchToSpaceND[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      input: Output[T],
      blockShape: Output[I1],
      crops: Output[I2],
      name: String = "BatchToSpaceND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "BatchToSpaceND",
      name = name,
      input = (input, blockShape, crops)
    ).setGradientFn(batchToSpaceNDGradient(_, _)(implicitly[IsInt32OrInt64[I1]], implicitly[IsInt32OrInt64[I2]]))
        .build().output
  }

  protected def batchToSpaceNDGradient[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[I2]) = {
    (spaceToBatchND(outputGradient, op.input._2, op.input._3), null, null)
  }

  /** $OpDocBasicRequiredSpaceToBatchPaddingsAndCrops
    *
    * @group BasicOps
    * @param  inputShape   Tensor with shape `[N]`.
    * @param  blockShape   Tensor with shape `[N]`.
    * @param  basePaddings Optional tensor with shape `[N, 2]` that specifies the minimum amount of padding to
    *                      use. All elements must be non-negative. Defaults to a tensor containing all zeros.
    * @param  name         Created op name.
    * @return Tuple containing the paddings and crops required.
    */
  def requiredSpaceToBatchPaddingsAndCrops(
      inputShape: Output[Int],
      blockShape: Output[Int],
      basePaddings: Output[Int] = null,
      name: String = "RequiredSpaceToBatchPaddings"
  ): (Output[Int], Output[Int]) = {
    Op.nameScope(name) {
      blockShape.shape.assertFullyDefined()
      blockShape.shape.assertHasRank(1)
      val numBlockDims = blockShape.shape(0)
      if (numBlockDims == 0) {
        (zeros(INT32, Shape(0, 2)), zeros(INT32, Shape(0, 2)))
      } else {
        inputShape.shape.assertIsCompatibleWith(Shape(numBlockDims))
        val actualBasePaddings = {
          if (basePaddings != null) {
            basePaddings.shape.assertIsCompatibleWith(Shape(numBlockDims, 2))
            basePaddings
          } else {
            zeros(INT32, Shape(numBlockDims, 2))
          }
        }
        val cInputShape = Output.constantValue(inputShape)
        val cBlockShape = Output.constantValue(blockShape)
        val cBasePaddings = Output.constantValue(actualBasePaddings)
        if (cInputShape.isDefined && cBlockShape.isDefined && cBasePaddings.isDefined) {
          val ccInputShape = cInputShape.get
          val ccBlockShape = cBlockShape.get
          val ccBasePaddings = cBasePaddings.get
          val padStart = ccBasePaddings(::, 0)
          val originalPadEnd = ccBasePaddings(::, 1)
          val fullInputShape = ccInputShape + padStart + originalPadEnd
          val extraPadEnd = (ccBlockShape - (fullInputShape % ccBlockShape)) % ccBlockShape
          val padEnd = originalPadEnd + extraPadEnd
          val resultPaddings = stack((0 until numBlockDims).map(i => {
            concatenate[Int](Seq(padStart(i), padEnd(i))).toOutput
          }))
          val zero = Tensor.zeros[Int](Shape())
          val resultCrops = stack((0 until numBlockDims).map(i => {
            concatenate[Int](Seq(zero, extraPadEnd(i))).toOutput
          }))
          (resultPaddings, resultCrops)
        } else {
          val padStart = actualBasePaddings(::, 0)
          val originalPadEnd = actualBasePaddings(::, 1)
          val fullInputShape = inputShape + padStart + originalPadEnd
          val extraPadEnd = (blockShape - (fullInputShape % blockShape)) % blockShape
          val padEnd = originalPadEnd + extraPadEnd
          val resultPaddings = stack(
            (0 until numBlockDims).map(i => concatenate(Seq(padStart(i), padEnd(i))).toOutput), name = "Paddings")
          val zero = zeros(padStart.dataType, Shape())
          val resultCrops = stack(
            (0 until numBlockDims).map(i => concatenate(Seq(zero, extraPadEnd(i))).toOutput), name = "Crops")
          (resultPaddings, resultCrops)
        }
      }
    }
  }

  /** $OpDocBasicSpaceToDepth
    *
    * @group BasicOps
    * @param  input      Tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize  Block size which must be greater than `1`.
    * @param  dataFormat Format of the input and output data.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def spaceToDepth[T](
      input: Output[T],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "SpaceToDepth"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "SpaceToDepth",
      name = name,
      input = input
    ).setAttribute("block_size", blockSize.toLong)
        .setAttribute("data_format", dataFormat.name)
        .setGradientFn(spaceToDepthGradient)
        .build().output
  }

  @throws[InvalidArgumentException]
  protected def spaceToDepthGradient[T](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    if (op.stringAttribute("data_format") == "NCHW_VECT_C")
      throw InvalidArgumentException(
        "Cannot compute 'spaceToDepth' gradient with 'NCHW_VECT_C' data format. " +
            "This format requires 'QINT8' data type.")
    depthToSpace(outputGradient, op.longAttribute("block_size").toInt)
  }

  /** $OpDocBasicDepthToSpace
    *
    * @group BasicOps
    * @param  input      Tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize  Block size which must be greater than `1`.
    * @param  dataFormat Format of the input and output data.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def depthToSpace[T](
      input: Output[T],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "DepthToSpace"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "DepthToSpace",
      name = name,
      input = input
    ).setAttribute("block_size", blockSize.toLong)
        .setAttribute("data_format", dataFormat.name)
        .setGradientFn(depthToSpaceGradient)
        .build().output
  }

  @throws[InvalidArgumentException]
  protected def depthToSpaceGradient[T](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    if (op.stringAttribute("data_format") == "NCHW_VECT_C")
      throw InvalidArgumentException(
        "Cannot compute 'spaceToDepth' gradient with 'NCHW_VECT_C' data format. " +
            "This format requires 'QINT8' data type.")
    spaceToDepth(outputGradient, op.longAttribute("block_size").toInt)
  }

  //endregion Tensor Manipulation Ops

  //region Tensor Masking Ops

  /** $OpDocBasicWhere
    *
    * @group BasicOps
    * @param  input Input boolean tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def where[T: IsBooleanOrNumeric](
      input: Output[T],
      name: String = "Where"
  ): Output[Long] = {
    Op.Builder[Output[T], Output[Long]](
      opType = "Where",
      name = name,
      input = input
    ).build().output
  }

  /** $OpDocBasicBooleanMask
    *
    * @group BasicOps
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional tensor, where `K <= N` and `K` must be known statically.
    * @param  name  Name for the created op output.
    * @return Created op output.
    */
  def booleanMask[T](
      input: Output[T],
      mask: Output[Boolean],
      name: String = "BooleanMask"
  ): Output[T] = {
    Op.nameScope(name) {
      val inputShape = input.shape
      val maskShape = mask.shape
      val maskRank = maskShape.rank
      if (maskRank < 0)
        throw InvalidShapeException(
          "The rank of the boolean mask must be known, even if some dimension sizes are unknown. For example, " +
              "'Shape(-1)' is fine, but 'Shape.unknown()' is not.")
      inputShape(0 :: maskRank).assertIsCompatibleWith(maskShape)
      val dynamicInputShape = shape(input, INT64)
      val leadingSize = Math.prod(dynamicInputShape(0 :: maskRank), Seq(0)).reshape(Shape(1))
      val reshapedInput = reshape(input, concatenate(Seq(leadingSize, dynamicInputShape(maskRank ::)), 0))
      val firstDimension = inputShape(0 :: maskRank).numElements.toInt
      if (maskRank >= inputShape.rank)
        reshapedInput.setShape(Shape(firstDimension))
      else
        reshapedInput.setShape(Shape(firstDimension).concatenateWith(inputShape(maskRank ::)))
      gather(reshapedInput, squeeze(where(reshape(mask, Seq(-1))), axes = Seq(1)), axis = 0)
    }
  }

  /** $OpDocBasicSequenceMask
    *
    * @group BasicOps
    * @param  lengths   One-dimensional tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @param  name      Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `maxLength` is not a scalar.
    */
  @throws[IllegalArgumentException]
  def sequenceMask[T: IsIntOrUInt](
      lengths: Output[T],
      maxLength: Output[T] = null,
      name: String = "SequenceMask"
  ): Output[Boolean] = {
    require(maxLength == null || maxLength.rank == -1 || maxLength.rank == 0, "'maxLength' must be a scalar.")
    val ops = if (maxLength == null) Set(lengths.op) else Set(lengths.op, maxLength.op)
    Op.nameScope(name) {
      val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
      // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
      // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
      // matrix of size [lengths.shape(0), maxLen].
      val rowVector = Math.range(Basic.zerosLike(maxLen), maxLen, Basic.onesLike(maxLen))
      // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
      // into INT32, then so do the elements of 'lengths'.
      val matrix = expandDims(lengths, -1).castTo(maxLen.dataType)
      Math.less(rowVector, matrix)
    }
  }

  /** $OpDocBasicIndexedSlicesMask
    *
    * @group BasicOps
    * @param  input       Input indexed slices.
    * @param  maskIndices One-dimensional tensor containing the indices of the elements to mask.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def indexedSlicesMask[T](
      input: OutputIndexedSlices[T],
      maskIndices: Output[Long],
      name: String = "IndexedSlicesMask"
  ): OutputIndexedSlices[T] = {
    Op.nameScope(name) {
      val (outputIndices, toGather) = listDiff(input.indices, maskIndices, INT64)
      val outputValues = gather(input.values, toGather, axis = 0)
      OutputIndexedSlices(indices = outputIndices, values = outputValues, denseShape = input.denseShape)
    }
  }

  //endregion Tensor Masking Ops

  //region Tensor Counting and Set Ops

  /** $OpDocBasicUnique
    *
    * @group BasicOps
    * @param  input           Input tensor.
    * @param  axis            Axis along which to compute the unique values.
    * @param  indicesDataType Data type of the returned indices.
    * @param  name            Name for the created op.
    * @return Tuple containing `output` and `indices`.
    */
  def unique[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      input: Output[T],
      axis: Output[I1],
      indicesDataType: DataType[I2],
      name: String = "Unique"
  ): (Output[T], Output[I2]) = {
    Op.Builder[(Output[T], Output[I1]), (Output[T], Output[I2])](
      opType = "UniqueV2",
      name = name,
      input = (input, axis)
    ).setAttribute("out_idx", indicesDataType)
        .build().output
  }

  /** $OpDocBasicUniqueWithCounts
    *
    * @group BasicOps
    * @param  input           Input tensor.
    * @param  axis            Axis along which to count the unique elements.
    * @param  indicesDataType Data type of the returned indices.
    * @param  name            Name for the created op.
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      input: Output[T],
      axis: Output[I1],
      indicesDataType: DataType[I2],
      name: String = "UniqueWithCounts"
  ): (Output[T], Output[I2], Output[I2]) = {
    Op.Builder[(Output[T], Output[I1]), (Output[T], Output[I2], Output[I2])](
      opType = "UniqueWithCountsV2",
      name = name,
      input = (input, axis)
    ).setAttribute("out_idx", indicesDataType)
        .build().output
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    * @param  x               One-dimensional tensor containing the values to keep.
    * @param  y               One-dimensional tensor containing the values to remove.
    * @param  indicesDataType Data type to use for the output indices of this op.
    * @param  name            Name for the created op.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff[T, I: IsInt32OrInt64](
      x: Output[T],
      y: Output[T],
      indicesDataType: DataType[I],
      name: String = "ListDiff"
  ): (Output[T], Output[I]) = {
    Op.Builder[(Output[T], Output[T]), (Output[T], Output[I])](
      opType = "ListDiff",
      name = name,
      input = (x, y)
    ).setAttribute("out_idx", indicesDataType)
        .build().output
  }

  //endregion Tensor Counting and Set Ops

  //region Tensor Slicing Ops

  /** $OpDocBasicGather
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  axis    Tensor containing the axis along which to gather.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def gather[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      input: Output[T],
      indices: Output[I1],
      axis: Output[I2] = null,
      name: String = "Gather"
  ): Output[T] = {
    if (axis != null) {
      Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
        opType = "GatherV2",
        name = name,
        input = (input, indices, axis)
      ).setGradientFn[(OutputLike[T], Output[I1], Output[I2]), Output[T]]({
        gatherGradient(_, _)(implicitly[IsInt32OrInt64[I1]], implicitly[IsInt32OrInt64[I2]])
      }).build().output
    } else {
      Op.Builder[(Output[T], Output[I1], Output[Int]), Output[T]](
        opType = "GatherV2",
        name = name,
        input = (input, indices, Tensor.zeros[Int](Shape()))
      ).setGradientFn[(OutputLike[T], Output[I1], Output[Int]), Output[T]]({
        gatherGradient(_, _)(implicitly[IsInt32OrInt64[I1]], implicitly[IsInt32OrInt64[Int]])
      }).build().output
    }
  }

  protected def gatherGradient[T, I1: IsInt32OrInt64, I2: IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (OutputLike[T], Output[I1], Output[I2]) = {
    // The input can be large, so we colocate the shape calculation with it.
    // The input can be very large for sparse models and 'shape' raises an exception on the Windows platform whenever
    // any dimension is larger than INT32. 'inputShape' is not used in the optimizer 'applySparse' gradients method
    // and so it's fine to convert it back to INT32 regardless of the truncation.
    val zero = zeros(INT32, Shape())
    val input = op.input._1
    val inputShape = Op.colocateWith(Set(input.op), ignoreExisting = true) {
      shape(input, INT64)
    }
    val indices = op.input._2.castTo[Long]
    val indicesSize = expandDims(size(indices, INT64), zero)
    val axis = op.input._3
    val axisStatic = Output.constantValue(axis)
    // For axis 0 gathers, we build appropriately shaped indexed slices.
    if (axisStatic.map(_.scalar).getOrElse(-1) == 0) {
      val valuesShape = concatenate(Seq(indicesSize, inputShape(1 ::)), zero)
      val values = reshape(outputGradient, valuesShape)
      val reshapedIndices = reshape(indices, indicesSize)
      val gradient = OutputIndexedSlices(indices = reshapedIndices, values = values, denseShape = inputShape)
      (gradient, null, null)
    } else {
      val intInputShape = inputShape.castTo[Int]
      val intIndicesSize = indicesSize.castTo[Int]
      val expandedAxis = axis.castTo[Int].expandDims(zero)
      val outerShape = slice(intInputShape, zero, expandedAxis)
      val outerSize = size(outerShape, INT32)
      val computedShape = slice(intInputShape, expandedAxis, size(inputShape, INT32).expandDims(zero))
      val innerShape = computedShape.slice(1 ::)
      val innerSize = size(innerShape, INT32)
      val outerAxesIndices = Math.range(zero, outerSize)
      val innerAxesIndices = Math.range(outerSize + 1, outerSize + 1 + innerSize)
      val valuesShape = concatenate(Seq(outerShape, intIndicesSize, innerShape), zero)
      val values = reshape(outputGradient, valuesShape)
      val reshapedIndices = reshape(indices, intIndicesSize)
      // We need to sum up every slice `values(..., i, ...)` corresponding to `input(..., indices(i), ...)`. Since
      // `unsortedSegmentSum` does not support an axis parameter, we transpose the gather dimension to the front, and
      // then use `unsortedSegmentSum` to build a `[gatherAxis, outerAxes, innerAxes]` tensor containing all the
      // gradients affecting each index in `gatherAxis` summed up.
      val transposeAxes = concatenate(
        Seq(outerSize.expandDims(zero), outerAxesIndices, innerAxesIndices), zero)
      val valuesTranspose = transpose(values, transposeAxes)
      val numSegments = inputShape.gather(axis)

      // TODO: [TYPES] !!! Super hacky. Remove in the future.
      implicit val ev: IsNumeric[T] = new IsNumeric[T] {}

      val inputGradient = Math.unsortedSegmentSum(valuesTranspose, reshapedIndices, numSegments)
      // We now invert the above transpose by moving dimension 0 back to its original position.
      val transposeAxesInverse = concatenate(Seq(outerAxesIndices + 1, zero, innerAxesIndices), zero)
      val inputGradientTranspose = transpose(inputGradient, transposeAxesInverse)
      (inputGradientTranspose, null, null)
    }
  }

  /** $OpDocBasicGatherND
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  name    Name for the created op.
    * @return Created op output that contains the values from `input` gathered from indices given by `indices`, with
    *         shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    */
  def gatherND[T, I: IsInt32OrInt64](
      input: Output[T],
      indices: Output[I],
      name: String = "GatherND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "GatherNd",
      name = name,
      input = (input, indices)
    ).setGradientFn[(OutputLike[T], Output[I]), Output[T]]({
      gatherNDGradient(_, _)(implicitly[IsInt32OrInt64[I]])
    }).build().output
  }

  protected def gatherNDGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (OutputLike[T], Output[I]) = {
    val indices = op.input._2
    val inputShape = shape(op.input._1, indices.dataType)
    if (indices.rank == 2 && indices.shape(-1) == 1) {
      (OutputIndexedSlices(
        indices = Basic.squeeze(indices, axes = Seq(-1)).castTo[Long],
        values = outputGradient,
        denseShape = inputShape.castTo[Long]), null)
    } else {
      (scatterND(indices, outputGradient, inputShape), null)
    }
  }

  /** $OpDocBasicScatterND
    *
    * @group BasicOps
    * @param  indices Indices tensor.
    * @param  updates Updates to scatter into the output tensor.
    * @param  shape   One-dimensional tensor specifying the shape of the output tensor.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def scatterND[T, I: IsInt32OrInt64](
      indices: Output[I],
      updates: Output[T],
      shape: Output[I],
      name: String = "ScatterND"
  ): Output[T] = {
    Op.Builder[(Output[I], Output[T], Output[I]), Output[T]](
      opType = "ScatterNd",
      name = name,
      input = (indices, updates, shape)
    ).setGradientFn(scatterNDGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def scatterNDGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[I], Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[I], Output[T], Output[I]) = {
    (null, gatherND(outputGradient, op.input._1), null)
  }

  /** $OpDocBasicSlice
    *
    * @group BasicOps
    * @param  input Tensor to slice.
    * @param  begin Begin index tensor. `begin(i)` specifies the offset into the `i`th dimension of `input` to slice
    *               from.
    * @param  size  Slice size tensor. `size(i)` specifies the number of elements of the `i`th dimension of `input` to
    *               slice. If `size(i) == -1`, then all the remaining elements in dimension `i` are included in the
    *               slice (i.e., this is equivalent to setting `size(i) = input.shape(i) - begin(i)`).
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  private[ops] def slice[T, I: IsInt32OrInt64](
      input: Output[T],
      begin: Output[I],
      size: Output[I],
      name: String = "Slice"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I], Output[I]), Output[T]](
      opType = "Slice",
      name = name,
      input = (input, begin, size)
    ).setGradientFn(sliceGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def sliceGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[I]) = {
    // Create an N x 2 padding where the first column represents how many zeros are to be prepended for each
    // dimension, and the second column indicates how many zeros are to be appended. The number of zeros to append
    // corresponds to the shape of the input elementwise-subtracted by both the begin vector and sizes vector. Some
    // more reshaping is needed to assemble this tensor with the right dimensions.
    val inputVector = op.input._1
    val beginVector = op.input._2
    val dataType = beginVector.dataType
    val inputRank = rank(inputVector)
    val padShape = stack[Int](Seq(inputRank, 1))
    val beforePad = reshape(beginVector, padShape)
    val afterPad = reshape(shape(inputVector, dataType) - shape(op.output, dataType) - beginVector, padShape)
    val paddings = concatenate(Seq(beforePad, afterPad), axis = 1)
    (pad(outputGradient, paddings), null, null)
  }

  /** $OpDocBasicStridedSlice
    *
    * @group BasicOps
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
  def stridedSlice[T, I: IsInt32OrInt64](
      input: Output[T],
      begin: Output[I],
      end: Output[I],
      strides: Output[I] = null,
      beginMask: Long = 0,
      endMask: Long = 0,
      ellipsisMask: Long = 0,
      newAxisMask: Long = 0,
      shrinkAxisMask: Long = 0,
      name: String = "StridedSlice"
  ): Output[T] = {
    val stridesWithDefault = if (strides != null) onesLike(begin) else strides
    Op.Builder[(Output[T], Output[I], Output[I], Output[I]), Output[T]](
      opType = "StridedSlice",
      name = name,
      input = (input, begin, end, stridesWithDefault)
    ).setAttribute("begin_mask", beginMask)
        .setAttribute("end_mask", endMask)
        .setAttribute("ellipsis_mask", ellipsisMask)
        .setAttribute("new_axis_mask", newAxisMask)
        .setAttribute("shrink_axis_mask", shrinkAxisMask)
        .setGradientFn(stridedSliceGradient(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
  }

  protected def stridedSliceGradient[T, I: IsInt32OrInt64](
      op: Op[(Output[T], Output[I], Output[I], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[I], Output[I]) = {
    val inputShape = shape(op.input._1, op.input._2.dataType)
    val gradient = Op.Builder[(Output[I], Output[I], Output[I], Output[I], Output[T]), Output[T]](
      opType = "StridedSliceGrad",
      name = "StridedSliceGradient",
      input = (inputShape, op.input._2, op.input._3, op.input._4, outputGradient)
    ).setAttribute("begin_mask", op.longAttribute("begin_mask"))
        .setAttribute("end_mask", op.longAttribute("end_mask"))
        .setAttribute("ellipsis_mask", op.longAttribute("ellipsis_mask"))
        .setAttribute("new_axis_mask", op.longAttribute("new_axis_mask"))
        .setAttribute("shrink_axis_mask", op.longAttribute("shrink_axis_mask"))
        .setGradientFn(stridedSliceHessian(_, _)(implicitly[IsInt32OrInt64[I]]))
        .build().output
    (gradient, null, null, null)
  }

  protected def stridedSliceHessian[T, I: IsInt32OrInt64](
      op: Op[(Output[I], Output[I], Output[I], Output[I], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[I], Output[I], Output[I], Output[I], Output[T]) = {
    val gradient = stridedSlice(
      input = outputGradient,
      begin = op.input._2,
      end = op.input._3,
      strides = op.input._4,
      beginMask = op.longAttribute("begin_mask").toInt,
      endMask = op.longAttribute("end_mask").toInt,
      ellipsisMask = op.longAttribute("ellipsis_mask").toInt,
      newAxisMask = op.longAttribute("new_axis_mask").toInt,
      shrinkAxisMask = op.longAttribute("shrink_axis_mask").toInt)
    (null, null, null, null, gradient)
  }

  // TODO: [OPS] Move 'stridedSliceAssign' to the variables package.

  /** $OpDocBasicStridedSliceAssign
    *
    * @group BasicOps
    * @param  input          Resource whose slice is being assigned `value`.
    * @param  value          Value to assign to the slice of `input`.
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
  private[api] def stridedSliceAssign[T, I: IsInt32OrInt64](
      input: Output[Long],
      value: Output[T],
      begin: Output[I],
      end: Output[I],
      strides: Output[I] = null,
      beginMask: Int = 0,
      endMask: Int = 0,
      ellipsisMask: Int = 0,
      newAxisMask: Int = 0,
      shrinkAxisMask: Int = 0,
      name: String = "StridedSliceAssign"
  ): Output[T] = {
    val stridesWithDefault = if (strides != null) onesLike(begin) else strides
    Op.Builder[(Output[Long], Output[I], Output[I], Output[I], Output[T]), Output[T]](
      opType = "ResourceStridedSliceAssign",
      name = name,
      input = (input, begin, end, stridesWithDefault, value)
    ).setAttribute("begin_mask", beginMask)
        .setAttribute("end_mask", endMask)
        .setAttribute("ellipsis_mask", ellipsisMask)
        .setAttribute("new_axis_mask", newAxisMask)
        .setAttribute("shrink_axis_mask", shrinkAxisMask)
        .build().output
  }

  //endregion Tensor Slicing Ops

  //region Tensor Ungrouped Ops

  /** $OpDocBasicCheckNumerics
    *
    * @group BasicOps
    * @param  input   Input tensor.
    * @param  message Prefix to print for the error message.
    * @param  name    Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def checkNumerics[T: IsDecimal](
      input: Output[T],
      message: String = "",
      name: String = "CheckNumerics"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "CheckNumerics",
      name = name,
      input = input
    ).setAttribute("message", message)
        .setGradientFn(checkNumericsGradient(_, _)(implicitly[IsDecimal[T]]))
        .build().output
  }

  protected def checkNumericsGradient[T: IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    checkNumerics(outputGradient, "Not-a-number (NaN) or infinity (Inf) values detected in the gradient.")
  }

  /** $OpDocBasicEditDistance
    *
    * @group BasicOps
    * @param  hypothesis Sparse tensor that contains the hypothesis sequences.
    * @param  truth      Sparse tensor that contains the truth sequences.
    * @param  normalize  Optional boolean value indicating whether to normalize the Levenshtein distance by the length
    *                    of `truth`.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def editDistance[T](
      hypothesis: SparseOutput[T],
      truth: SparseOutput[T],
      normalize: Boolean = true,
      name: String = "EditDistance"
  ): Output[Float] = {
    Op.Builder[(SparseOutput[T], SparseOutput[T]), Output[Float]](
      opType = "EditDistance",
      name = name,
      input = (hypothesis, truth)
    ).setAttribute("normalize", normalize)
        .build().output
  }

  /** $OpDocBasicOneHot
    *
    * @group BasicOps
    * @param  indices  Tensor containing the indices for the "on" values.
    * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
    * @param  dataType Data type of the output tensor. If not provided, the function will attempt to assume the data
    *                  type of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`,
    *                  or `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
    * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
    *                  Defaults to the value `1` with type `dataType`.
    * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
    *                  Defaults to the value `0` with type `dataType`.
    * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def oneHot[T, I: IsInt32OrInt64OrUInt8](
      indices: Output[I],
      depth: Output[Int],
      dataType: DataType[T],
      onValue: Output[T] = null,
      offValue: Output[T] = null,
      axis: Int = -1,
      name: String = "OneHot"
  ): Output[T] = {
    Op.nameScope(name) {
      val actualOnValue = if (onValue != null) onValue.castTo(dataType) else ones(dataType, Shape())
      val actualOffValue = if (offValue != null) offValue.castTo(dataType) else zeros(dataType, Shape())
      Op.Builder[(Output[I], Output[Int], Output[T], Output[T]), Output[T]](
        opType = "OneHot",
        name = name,
        input = (indices, depth, actualOnValue, actualOffValue)
      ).setAttribute("axis", axis)
          .build().output
    }
  }

  //endregion Tensor Ungrouped Ops

  // TODO: [OPS] Add support for all the quantization ops.

  //region Tensor Broadcasting Ops

  /** $OpDocBasicBroadcastGradientArguments
    *
    * @group BasicOps
    * @param  shape1 First operand shape.
    * @param  shape2 Second operand shape.
    * @param  name   Name for the created op.
    * @return Tuple containing two op outputs, each containing the reduction indices for the corresponding op.
    */
  def broadcastGradientArguments[I: IsInt32OrInt64](
      shape1: Output[I],
      shape2: Output[I],
      name: String = "BroadcastGradientArguments"
  ): (Output[I], Output[I]) = {
    Op.Builder[(Output[I], Output[I]), (Output[I], Output[I])](
      opType = "BroadcastGradientArgs",
      name = name,
      input = (shape1, shape2)
    ).build().output
  }

  /** $OpDocBasicBroadcastTo
    *
    * @group BasicOps
    * @param  value Tensor to broadcast.
    * @param  shape Shape to broadcast the provided tensor to.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def broadcastTo[T, I: IsInt32OrInt64](
      value: Output[T],
      shape: Output[I],
      name: String = "BroadcastTo"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "BroadcastTo",
      name = name,
      input = (value, shape)
    ).build().output
  }

  // TODO: Add support for "broadcastShape" (static). Implement the main method in the "Shape" class.

  /** $OpDocBasicBroadcastShape
    *
    * @group BasicOps
    * @param  shape1 One-dimensional tensor representing the shape of the first argument.
    * @param  shape2 One-dimensional tensor representing the shape of the first argument.
    * @param  name   Name for the created op.
    * @return Created op output, which is a one-dimensional integer tensor representing the broadcasted shape.
    */
  def broadcastShapeDynamic[I: IsInt32OrInt64](
      shape1: Output[I],
      shape2: Output[I],
      name: String = "BroadcastShape"
  ): Output[I] = {
    Op.Builder[(Output[I], Output[I]), Output[I]](
      opType = "BroadcastArgs",
      name = name,
      input = (shape1, shape2)
    ).build().output
  }

  /** $OpDocBasicMeshGrid
    *
    * @group BasicOps
    * @param  inputs               Sequence containing `N` input rank-`1` tensors.
    * @param  useCartesianIndexing If `true` (the default value), the broadcasting instructions for the first two
    *                              dimensions are swapped.
    * @param  name                 Name for the created op.
    * @return Created op outputs, each with rank `N`.
    */
  def meshGrid[T: IsNotQuantized](
      inputs: Seq[Output[T]],
      useCartesianIndexing: Boolean = true,
      name: String = "MeshGrid"
  ): Seq[Output[T]] = {
    Op.nameScope(name) {
      val rank = inputs.length
      val (outputs, shapes) = {
        // Prepare reshape by inserting dimensions with size 1 where needed.
        val outputs = inputs.zipWithIndex.map(i => {
          val shape = Shape.fromSeq(Seq.fill(i._2)(1) ++ (-1 +: Seq.fill(rank - i._2 - 1)(1)))
          reshape(i._1, shape)
        })
        // Create parameters for broadcasting each tensor to the full size.
        val shapes = inputs.map(size(_, INT64))
        if (useCartesianIndexing) {
          outputs.zip(shapes).zipWithIndex.map(o => o._2 match {
            case 0 =>
              val outputsShape = Shape.fromSeq(Seq(1, -1) ++ Seq.fill(rank - 2)(1))
              (reshape(o._1._1, outputsShape), shapes(1))
            case 1 =>
              val outputsShape = Shape.fromSeq(Seq(-1, 1) ++ Seq.fill(rank - 2)(1))
              (reshape(o._1._1, outputsShape), shapes(0))
            case _ => o._1
          }).unzip
        } else {
          (outputs, shapes)
        }
      }
      // TODO: [OPS] Improve performance with a broadcast.
      val multiplicativeFactor = fill(inputs.head.dataType, stack(shapes))(1)
      outputs.map(Math.multiply(_, multiplicativeFactor))
    }
  }

  //endregion Tensor Broadcasting Ops

  //region Tensor Gradient Ops

  /** $OpDocBasicStopGradient
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def stopGradient[T](
      input: Output[T],
      name: String = "StopGradient"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "StopGradient",
      name = name,
      input = input
    ).build().output
  }

  /** $OpDocBasicPreventGradient
    *
    * @group BasicOps
    * @param  input   Input tensor.
    * @param  message Message to print along with the error.
    * @param  name    Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  def preventGradient[T](
      input: Output[T],
      message: String = "",
      name: String = "PreventGradient"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "PreventGradient",
      name = name,
      input = input
    ).setAttribute("message", message)
        .setGradientFn(preventGradientGradient)
        .build().output
  }

  @throws[IllegalArgumentException]
  protected def preventGradientGradient[T](
      op: Op[Output[T], Output[T]],
      outputGradients: Output[T]
  ): Output[T] = {
    throw new IllegalArgumentException(
      s"Gradient explicitly disabled. Reason: ${op.stringAttribute("message")}.")
  }

  //endregion Tensor Gradient Ops
}

object Basic extends Basic {
  private[ops] trait Implicits {
    implicit def outputConvertibleToBasicOps[OC, T](
        value: OC
    )(implicit f: OC => Output[T]): BasicOps[T] = {
      new BasicOps(f(value))
    }

    implicit class BasicOps[T](val output: Output[T]) {
      //region Tensor Manipulation Ops

      /** $OpDocBasicIdentity
        *
        * @group BasicOps
        * @return Created op output.
        */
      def identity: Output[T] = {
        Basic.identity(output)
      }

      /** $OpDocBasicExpandDims
        *
        * @group BasicOps
        * @param  axis Dimension index at which to expand the shape of this tensor.
        * @return Result as a new tensor.
        */
      def expandDims(axis: Output[Int]): Output[T] = {
        Basic.expandDims(output, axis)
      }

      /** $OpDocBasicSqueeze
        *
        * @group BasicOps
        * @param  axes Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
        *              will be squeezed.
        * @return Result as a new tensor.
        */
      def squeeze(axes: Seq[Int] = null): Output[T] = {
        Basic.squeeze(output, axes)
      }

      /** $OpDocBasicUnstack
        *
        * @group BasicOps
        * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
        * @param  axis   Dimension along which to unstack the input tensor.
        * @return Result as a new tensor.
        */
      def unstack(number: Int = -1, axis: Int = 0): Seq[Output[T]] = {
        Basic.unstack(output, number, axis)
      }

      /** $OpDocBasicSplitEvenly
        *
        * @group BasicOps
        * @param  numSplits Number of splits to obtain along the `axis` dimension.
        * @param  axis      Dimension along which to split the input tensor.
        * @return Result as a sequence of new tensors.
        */
      def splitEvenly(
          numSplits: Int,
          axis: Output[Int] = 0
      ): Seq[Output[T]] = {
        Basic.splitEvenly(output, numSplits, axis)
      }

      /** $OpDocBasicSplit
        *
        * @group BasicOps
        * @param  splitSizes Sizes for the splits to obtain.
        * @param  axis       Dimension along which to split the input tensor.
        * @return Result as a new tensor.
        */
      def split[I: IsInt32OrInt64](
          splitSizes: Output[I],
          axis: Output[Int] = 0
      ): Seq[Output[T]] = {
        Basic.split(output, splitSizes, axis)
      }

      /** $OpDocBasicTile
        *
        * @group BasicOps
        * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the
        *                   rank of `input`.
        * @return Result as a new tensor.
        */
      def tile[I: IsInt32OrInt64](multiples: Output[I]): Output[T] = {
        Basic.tile(output, multiples)
      }

      /** $OpDocBasicPad
        *
        * @group BasicOps
        * @param  paddings Tensor containing the paddings.
        * @param  mode     Padding mode to use.
        * @return Result as a new tensor.
        */
      def pad[I: IsInt32OrInt64](
          paddings: Output[I],
          mode: PaddingMode = ConstantPadding(Some(Tensor(0)))
      ): Output[T] = {
        Basic.pad(output, paddings, mode)
      }

      /** $OpDocBasicReshape
        *
        * @group BasicOps
        * @param  shape Shape of the output tensor.
        * @return Result as a new tensor.
        */
      def reshape[I: IsInt32OrInt64](shape: Output[I]): Output[T] = {
        Basic.reshape(output, shape)
      }

      /** $OpDocBasicTranspose
        *
        * @group BasicOps
        * @param  permutation Permutation of the input tensor dimensions.
        * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def transpose[I: IsInt32OrInt64](
          permutation: Output[I] = null,
          conjugate: Boolean = false
      ): Output[T] = {
        Basic.transpose(output, permutation, conjugate)
      }

      /** $OpDocBasicMatrixTranspose
        *
        * @group BasicOps
        * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def matrixTranspose(conjugate: Boolean = false): Output[T] = {
        Basic.matrixTranspose(output, conjugate)
      }

      /** $OpDocBasicInvertPermutation
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def invertPermutation(implicit ev: IsInt32OrInt64[T]): Output[T] = {
        Basic.invertPermutation(output)
      }

      /** $OpDocBasicReverse
        *
        * @group BasicOps
        * @param  axes Dimensions of the input tensor to reverse.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverse[I: IsInt32OrInt64](axes: Output[I]): Output[T] = {
        Basic.reverse(output, axes)
      }

      /** $OpDocBasicReverseSequence
        *
        * @group BasicOps
        * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
        *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
        * @param  sequenceAxis    Tensor dimension which is partially reversed.
        * @param  batchAxis       Tensor dimension along which the reversal is performed.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverseSequence[I: IsInt32OrInt64](
          sequenceLengths: Output[I],
          sequenceAxis: Int,
          batchAxis: Int = 0
      ): Output[T] = {
        Basic.reverseSequence(output, sequenceLengths, sequenceAxis, batchAxis)
      }

      /** $OpDocBasicSpaceToBatch
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def spaceToBatch[I: IsInt32OrInt64](
          blockSize: Int,
          paddings: Output[I]
      ): Output[T] = {
        Basic.spaceToBatch(output, blockSize, paddings)
      }

      /** $OpDocBasicSpaceToBatchND
        *
        * @group BasicOps
        * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
        * @param  paddings   Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
        *                    `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension `i + 1`, which
        *                    corresponds to spatial dimension `i`. It is required that `blockShape(i)` divides
        *                    `inputShape(i + 1) + padStart + padEnd`.
        * @return Result as a new tensor.
        */
      def spaceToBatchND[I1: IsInt32OrInt64, I2: IsInt32OrInt64](
          blockShape: Output[I1],
          paddings: Output[I2]
      ): Output[T] = {
        Basic.spaceToBatchND(output, blockShape, paddings)
      }

      /** $OpDocBasicBatchToSpace
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  crops     `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def batchToSpace[I: IsInt32OrInt64](
          blockSize: Int,
          crops: Output[I]
      ): Output[T] = {
        Basic.batchToSpace(output, blockSize, crops)
      }

      /** $OpDocBasicBatchToSpaceND
        *
        * @group BasicOps
        * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
        * @param  crops      Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
        *                    `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input dimension `i + 1`,
        *                    which corresponds to spatial dimension `i`. It is required that
        *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
        * @return Result as a new tensor.
        */
      def batchToSpaceND[I1: IsInt32OrInt64, I2: IsInt32OrInt64](
          blockShape: Output[I1],
          crops: Output[I2]
      ): Output[T] = {
        Basic.batchToSpaceND(output, blockShape, crops)
      }

      /** $OpDocBasicSpaceToDepth
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def spaceToDepth(
          blockSize: Int,
          dataFormat: CNNDataFormat = CNNDataFormat.default
      ): Output[T] = {
        Basic.spaceToDepth(output, blockSize, dataFormat)
      }

      /** $OpDocBasicDepthToSpace
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def depthToSpace(
          blockSize: Int,
          dataFormat: CNNDataFormat = CNNDataFormat.default
      ): Output[T] = {
        Basic.depthToSpace(output, blockSize, dataFormat)
      }

      //endregion Tensor Manipulation Ops

      //region Tensor Masking Ops

      /** $OpDocBasicWhere
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def where(implicit ev: IsBooleanOrNumeric[T]): Output[Long] = {
        Basic.where(output)
      }

      /** $OpDocBasicBooleanMask
        *
        * @group BasicOps
        * @param  mask `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
        * @return Result as a new tensor.
        */
      def booleanMask(mask: Output[Boolean]): Output[T] = {
        Basic.booleanMask(output, mask)
      }

      /** $OpDocBasicSequenceMask
        *
        * @group BasicOps
        * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum
        *                   value in this tensor.
        * @return Result as a new tensor.
        */
      def sequenceMask(
          maxLength: Output[T] = null
      )(implicit ev: IsIntOrUInt[T]): Output[Boolean] = {
        Basic.sequenceMask(output, maxLength)
      }

      //endregion Tensor Masking Ops

      //region Tensor Counting and Set Ops

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @return Tuple containing `output` and `indices`.
        */
      def unique[I1: IsInt32OrInt64](
          axis: Output[I1]
      ): (Output[T], Output[Int]) = {
        Basic.unique(output, axis, indicesDataType = INT32)
      }

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output` and `indices`.
        */
      def unique[I1: IsInt32OrInt64, I2: IsInt32OrInt64](
          axis: Output[I1],
          indicesDataType: DataType[I2]
      ): (Output[T], Output[I2]) = {
        Basic.unique(output, axis, indicesDataType)
      }

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts[I1: IsInt32OrInt64](
          axis: Output[I1]
      ): (Output[T], Output[Int], Output[Int]) = {
        Basic.uniqueWithCounts(output, axis, indicesDataType = INT32)
      }

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts[I1: IsInt32OrInt64, I2: IsInt32OrInt64](
          axis: Output[I1],
          indicesDataType: DataType[I2]
      ): (Output[T], Output[I2], Output[I2]) = {
        Basic.uniqueWithCounts(output, axis, indicesDataType)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other One-dimensional tensor containing the values to remove.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff(other: Output[T]): (Output[T], Output[Int]) = {
        Basic.listDiff(output, other, INT32)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other           One-dimensional tensor containing the values to remove.
        * @param  indicesDataType Data type to use for the output indices of this op.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff[I: IsInt32OrInt64](
          other: Output[T],
          indicesDataType: DataType[I]
      ): (Output[T], Output[I]) = {
        Basic.listDiff(output, other, indicesDataType)
      }

      //endregion Tensor Counting and Set Ops

      //region Tensor Slicing Ops

      /** $OpDocBasicGather
        *
        * @group BasicOps
        * @param  indices Tensor containing indices to gather.
        * @param  axis    Tensor containing the axis along which to gather.
        * @return Result as a new tensor.
        */
      def gather[I: IsInt32OrInt64](
          indices: Output[I],
          axis: Output[I] = null
      ): Output[T] = {
        Basic.gather(output, indices, axis)
      }

      /** $OpDocBasicGatherND
        *
        * @group BasicOps
        * @param  indices Tensor containing indices to gather.
        * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
        *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
        */
      def gatherND[I: IsInt32OrInt64](indices: Output[I]): Output[T] = {
        Basic.gatherND(output, indices)
      }

      //endregion Tensor Slicing Ops

      //region Tensor Ungrouped Ops

      /** $OpDocBasicCheckNumerics
        *
        * @group BasicOps
        * @param  message Prefix to print for the error message.
        * @return Result as a new tensor which has the same value as the input tensor.
        */
      def checkNumerics(
          message: String = ""
      )(implicit ev: IsDecimal[T]): Output[T] = {
        Basic.checkNumerics(output)
      }

      /** $OpDocBasicOneHot
        *
        * @group BasicOps
        * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
        * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
        *                  Defaults to the value `1` with type `dataType`.
        * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
        *                  Defaults to the value `0` with type `dataType`.
        * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
        * @param  dataType Data type of the output tensor. If not provided, the function will attempt to assume the data
        *                  type of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`,
        *                  or `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
        * @return Created op output.
        */
      def oneHot[R](
          depth: Output[Int],
          dataType: DataType[R],
          onValue: Output[R] = null,
          offValue: Output[R] = null,
          axis: Int = -1
      )(implicit ev: IsInt32OrInt64OrUInt8[T]): Output[R] = {
        Basic.oneHot(output, depth, dataType, onValue, offValue, axis)
      }

      //endregion Tensor Ungrouped Ops

      //region Tensor Broadcasting Ops

      /** $OpDocBasicBroadcastTo
        *
        * @group BasicOps
        * @param  shape Shape to broadcast the provided tensor to.
        * @return Created op output.
        */
      def broadcastTo[I: IsInt32OrInt64](
          shape: Output[I]
      ): Output[T] = {
        Basic.broadcastTo(output, shape)
      }

      //endregion Tensor Broadcasting Ops

      //region Tensor Gradient Ops

      /** $OpDocBasicStopGradient
        *
        * @group BasicOps
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def stopGradient(): Output[T] = {
        Basic.stopGradient(output)
      }

      /** $OpDocBasicPreventGradient
        *
        * @group BasicOps
        * @param  message Message to print along with the error.
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def preventGradient(message: String = ""): Output[T] = {
        Basic.preventGradient(output, message)
      }

      //endregion Tensor Gradient Ops
    }
  }

  /** @define OpDocBasicConstant
    *   The `constant` op returns a constant tensor.
    *
    *   The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
    *   (optionally) `shape` (see examples below).
    *
    *   The argument `value` can be a constant value, or a tensor. If `value` is a one-dimensional tensor, then its
    *   length should be equal to the number of elements implied by the `shape` argument (if specified).
    *
    *   The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
    *
    *   The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not
    *   present, the shape of `value` is used.
    *
    * @define OpDocBasicGuaranteeConstant
    *   The `guaranteeConstant` op gives a guarantee to the TensorFlow runtime that the input tensor is a constant. The
    *   runtime is then free to make optimizations based on this. The op only accepts value-typed tensors as inputs and
    *   rejects resource variable handles. It returns the input tensor without modification.
    *
    * @define OpDocBasicImmutableConstant
    *   The `immutableConstant` op returns an immutable tensor from the provided memory region.
    *
    *   The current implementation memory-maps the tensor from a file.
    *
    * @define OpDocBasicZeros
    *   The `zeros` op returns a tensor of type `dataType` with shape `shape` and all elements set to zero.
    *
    *   For example:
    *   {{{
    *      zeros(INT32, Shape(3, 4)) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    *   }}}
    *
    * @define OpDocBasicZerosLike
    *   The `zerosLike` op returns a tensor of zeros with the same shape and data type as `input`.
    *
    *   Given a single tensor (`input`), the op returns a tensor of the same type and shape as `input` but with all
    *   elements set to zero. Optionally, you can use `dataType` to specify a new type for the returned tensor.
    *
    *   For example:
    *   {{{*   // 't' is [[1, 2, 3], [4, 5, 6]]
    *      zerosLike(t) ==> [[0, 0, 0], [0, 0, 0]]
    *   }}}
    *
    * @define OpDocBasicOnes
    *   The    `ones` op returns a tensor of type `dataType` with shape `shape` and all elements set to one.
    *
    *   For example:
    *   {{{
    *      ones(INT32, Shape(3, 4)) ==> [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    *   }}}
    *
    * @define OpDocBasicOnesLike
    *   The `onesLike` op returns a tensor of ones with the same shape and data type as `input`.
    *
    *   Given a single tensor (`input`), the op returns a tensor of the same type and shape as `input` but with all
    *   elements set to one. Optionally, you can use `dataType` to specify a new type for the returned tensor.
    *
    *   For example:
    *   {{{
    *      // 't' is [[1, 2, 3], [4, 5, 6]]
    *      onesLike(t) ==> [[1, 1, 1], [1, 1, 1]]
    *   }}}
    *
    * @define OpDocBasicFill
    *   The `fill` op returns a tensor filled with the provided scalar value.
    *
    *   The op creates a tensor of shape `shape` and fills it with `value`.
    *
    *   For example:
    *   {{{
    *      fill(Shape(2, 3), 9) ==> [[9, 9, 9], [9, 9, 9]]
    *   }}}
    *
    * @define OpDocBasicPlaceholder
    *   The `placeholder` op returns a placeholder for a tensor that will always be fed.
    *
    *   '''IMPORTANT NOTE:''' This op will produce an error if evaluated. Its value must be fed when using
    *   `Session.run`. It is intended as a way to represent a value that will always be fed, and to provide attributes
    *   that enable the fed value to be checked at runtime.
    *
    * @define OpDocBasicPlaceholderWithDefault
    *   The `placeholderWithDefault` op returns a placeholder op that passes through a defult value when its input is
    *   not fed.
    *
    * @define OpDocBasicSparsePlaceholder
    *   The `sparsePlaceholder` op returns a placeholder for a sparse tensor that will always be fed.
    *
    *   '''IMPORTANT NOTE:''' This op will produce an error if evaluated. Its value must be fed when using
    *   `Session.run`. It is intended as a way to represent a value that will always be fed, and to provide attributes
    *   that enable the fed value to be checked at runtime.
    *
    * @define OpDocBasicRank
    *   The `rank` op returns the rank of a tensor.
    *
    *   The op returns an integer representing the rank of `input`.
    *
    *   For example:
    *   {{{
    *      // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *      // 't' has shape [2, 2, 3]
    *      rank(t) ==> 3
    *   }}}
    *
    *   Note that the rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number of
    *   indices required to uniquely select each element of the tensor. Rank is also known as order, degree, or number
    *   of dimensions.
    *
    * @define OpDocBasicSize
    *   The `size` op returns the size of a tensor.
    *
    *   The op returns a number representing the number of elements in `input`.
    *
    *   For example:
    *   {{{
    *      // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
    *      size(t) ==> 12
    *   }}}
    *
    * @define OpDocBasicShape
    *   The `shape` op returns the shape of a tensor.
    *
    *   The op returns a one-dimensional tensor representing the shape of `input`.
    *
    *   For example:
    *   {{{
    *      // 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    *      shape(t) ==> [2, 2, 3]
    *   }}}
    *
    * @define OpDocBasicShapeN
    *   The `shapeN` op returns the shape of an array of tensors.
    *
    *   The op returns an array of one-dimensional tensors, each one representing the shape of the corresponding tensor
    *   in `inputs`.
    *
    * @define OpDocBasicIdentity
    *   The `identity` op returns a tensor with the same shape and contents as the input tensor.
    *
    * @define OpDocBasicExpandDims
    *   The `expandDims` op inserts a dimension of size 1 into the tensor's shape and returns the result as a new
    *   tensor.
    *
    *   Given a tensor `input`, the op inserts a dimension of size 1 at the dimension index `axis` of the tensor's
    *   shape. The dimension index `axis` starts at zero; if you specify a negative number for `axis` it is counted
    *   backwards from the end.
    *
    *   This op is useful if you want to add a batch dimension to a single element. For example, if you have a single
    *   image of shape `[height, width, channels]`, you can make it a batch of 1 image with `expandDims(image, 0)`,
    *   which will make the shape equal to `[1, height, width, channels]`.
    *
    *   For example:
    *   {{{*   // 't1' is a tensor of shape [2]
    *      t1.expandDims(0).shape == Shape(1, 2)
    *      t1.expandDims(1).shape == Shape(2, 1)
    *      t1.expandDims(-1).shape == Shape(2, 1)
    *
    *      // 't2' is a tensor of shape [2, 3, 5]
    *      t2.expandDims(0).shape == Shape(1, 2, 3, 5)
    *      t2.expandDims(2).shape == Shape(2, 3, 1, 5)
    *      t2.expandDims(3).shape == Shape(2, 3, 5, 1)
    *   }}}
    *
    *   This op requires that `-1 - input.rank <= axis <= input.rank`.
    *
    *   This is related to `squeeze`, which removes dimensions of size 1.
    *
    * @define OpDocBasicSqueeze
    *   The `squeeze` op removes dimensions of size 1 from the shape of a tensor and returns the result as a new tensor.
    *
    *   Given a tensor `input`, the op returns a tensor of the same data type, with all dimensions of size 1 removed.
    *   If `axes` is specified, then only the dimensions specified by that array will be removed. In that case, all
    *   these dimensions need to have size 1.
    *
    *   For example:
    *   {{{
    *      // 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    *      t.squeeze().shape == Shape(2, 3)
    *      t.squeeze(Array(2, 4)).shape == Shape(1, 2, 3, 1)
    *   }}}
    *
    * @define OpDocBasicStack
    *   The `stack` op stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
    *
    *   The op packs the list of tensors in `inputs` into a tensor with rank one higher than each tensor in `inputs`, by
    *   packing them along the `axis` dimension. Given a list of `N` tensors of shape `[A, B, C]`:
    *   - If `axis == 0`, then the output tensor will have shape `[N, A, B, C]`.
    *   - If `axis == 1`, then the output tensor will have shape `[A, N, B, C]`.
    *   - If `axis == -1`, then the output tensor will have shape `[A, B, C, N]`.
    *   - etc.
    *
    *   For example:
    *   {{{
    *      // 'x' is [1, 4]
    *      // 'y' is [2, 5]
    *      // 'z' is [3, 6]
    *      stack(Array(x, y, z)) ==> [[1, 4], [2, 5], [3, 6]]          // Packed along the first dimension.
    *      stack(Array(x, y, z), axis = 1) ==> [[1, 2, 3], [4, 5, 6]]  // Packed along the second dimension.
    *   }}}
    *
    *   This op is the opposite of `unstack`.
    *
    * @define OpDocBasicParallelStack
    *   The `parallelStack` op stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor, in parallel.
    *
    *   The op packs the list of tensors in `inputs` into a tensor with rank one higher than each tensor in `inputs`, by
    *   packing them along the first dimension. Given a list of `N` tensors of shape `[A, B, C]`, the output tensor will
    *   have shape `[N, A, B, C]`.
    *
    *   For example:
    *   {{{
    *      // 'x' is [1, 4]
    *      // 'y' is [2, 5]
    *      // 'z' is [3, 6]
    *      parallelStack(Array(x, y, z)) ==> [[1, 4], [2, 5], [3, 6]]
    *   }}}
    *
    *   The op requires that the shape of all input tensors is known at graph construction time.
    *
    *   The difference between `stack` and `parallelStack` is that `stack` requires all of the inputs be computed before
    *   the operation will begin executing, but does not require that the input shapes be known during graph
    *   construction. `parallelStack` will copy pieces of the input into the output as they become available. In some
    *   situations this can provide a performance benefit.
    *
    * @define OpDocBasicUnstack
    *   The `unstack` op unpacks the provided dimension of a rank-`R` tensor into a list of rank-`(R-1)` tensors.
    *
    *   The op unpacks `number` tensors from `input` by chipping it along the `axis` dimension. If `number == -1` (i.e.,
    *   unspecified), its value is inferred from the shape of `input`. If `input.shape(axis)` is not known, then an
    *   [[IllegalArgumentException]] is thrown.
    *
    *   For example, given a tensor of shape `[A, B, C, D]`:
    *   - If `axis == 0`, then the `i`th tensor in the output is the slice `input(i, ::, ::, ::)` and each tensor in the
    *     output will have shape `[B, C, D]`.
    *   - If `axis == 1`, then the `i`th tensor in the output is the slice `input(::, i, ::, ::)` and each tensor in the
    *     output will have shape `[A, C, D]`.
    *   - If `axis == -1`, then the `i`th tensor in the output is the slice `input(::, ::, ::, i)` and each tensor in
    *     the output will have shape `[A, B, C]`.
    *   - etc.
    *
    *   This op is the opposite of `stack`.
    *
    * @define OpDocBasicConcatenate
    *   The `concatenate` op concatenates tensors along one dimension.
    *
    *   The op concatenates the list of tensors `inputs` along the dimension `axis`. If
    *   `inputs(i).shape = [D0, D1, ..., Daxis(i), ..., Dn]`, then the concatenated tensor will have shape
    *   `[D0, D1, ..., Raxis, ..., Dn]`, where `Raxis = sum(Daxis(i))`. That is, the data from the input tensors is
    *   joined along the `axis` dimension.
    *
    *   For example:
    *   {{{
    *      // 't1' is equal to [[1, 2, 3], [4, 5, 6]]
    *      // 't2' is equal to [[7, 8, 9], [10, 11, 12]]
    *      concatenate(Array(t1, t2), 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    *      concatenate(Array(t1, t2), 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    *
    *      // 't3' has shape [2, 3]
    *      // 't4' has shape [2, 3]
    *      concatenate(Array(t3, t4), 0).shape ==> [4, 3]
    *      concatenate(Array(t3, t4), 1).shape ==> [2, 6]
    *   }}}
    *
    *   Note that, if you want to concatenate along a new axis, it may be better to use the `stack` op instead:
    *   {{{
    *      concatenate(tensors.map(t => expandDims(t, axis)), axis) == stack(tensors, axis)
    *   }}}
    *
    * @define OpDocBasicConcatenateOffset
    *   The `concatenateOffset` op computes offsets of `concatenate` inputs within its output.
    *
    *   For example:
    *   {{{
    *      // 'x' is a tensor containing values [2, 2, 7]
    *      // 'y' is a tensor containing values [2, 3, 7]
    *      // 'z' is a tensor containing values [2, 5, 7]
    *      concatenateOffset(Seq(x, y, z), 2) ==> [0, 0, 0], [0, 2, 0], [0, 5, 0]
    *   }}}
    *
    *   This function is typically used by gradient computations for a `concatenate` op.
    *
    * @define OpDocBasicSplitEvenly
    *   The `splitEvenly` op splits a tensor into sub-tensors.
    *
    *   The op splits `input` along dimension `axis` into `numSplits` smaller tensors. It requires that `numSplits`
    *   evenly splits `input.shape(axis)`.
    *
    *   For example:
    *   {{{
    *      // 't' is a tensor with shape [5, 30]
    *      // Split 't' into 3 tensors along dimension 1:
    *      val splits = split(t, numSplits = 3, axis = 1)
    *      splits(0).shape ==> [5, 10]
    *      splits(1).shape ==> [5, 10]
    *      splits(2).shape ==> [5, 10]
    *   }}}
    *
    * @define OpDocBasicSplit
    *   The `split` op splits a tensor into sub-tensors.
    *
    *   The op splits `input` along dimension `axis` into `splitSizes.length` smaller tensors. The shape of the `i`-th
    *   smaller tensor has the same size as the `input` except along dimension `axis` where the size is equal to
    *   `   splitSizes(i)`.
    *
    *   For example:
    *   {{{
    *      // 't' is a tensor with shape [5, 30]
    *      // Split 't' into 3 tensors with sizes [4, 5, 11] along dimension 1:
    *      val splits = split(t, splitSizes = [4, 15, 11], axis = 1)
    *      splits(0).shape ==> [5, 4]
    *      splits(1).shape ==> [5, 15]
    *      splits(2).shape ==> [5, 11]
    *   }}}
    *
    * @define OpDocBasicTile
    *   The `tile` op tiles the provided input tensor.
    *
    *   The op creates a new tensor by replicating `input` `multiples` times. The output tensor's `i`th dimension has
    *   `input.shape(i) * multiples(i)` elements, and the values of `input` are replicated `multiples(i)` times along
    *   the `i`th dimension. For example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
    *
    * @define OpDocBasicPad
    *   The `pad` op pads a tensor with zeros.
    *
    *   The op pads `input` with values specified by the padding mode, `mode`, according to the `paddings` you specify.
    *
    *   `paddings` is an integer tensor with shape `[n, 2]`, where `n` is the rank of `input`. For each dimension `D` of
    *   `input`, `paddings(D, 0)` indicates how many zeros to add before the contents of `input` in that dimension, and
    *   `paddings(D, 1)` indicates how many zeros to add after the contents of `input` in that dimension.
    *
    *   If `mode` is [[ReflectivePadding]] then both `paddings(D, 0)` and `paddings(D, 1)` must be no greater than
    *   `input.shape(D) - 1`. If `mode` is [[SymmetricPadding]] then both `paddings(D, 0)` and `paddings(D, 1)` must be
    *   no greater than `input.shape(D)`.
    *
    *   The padded size of each dimension `D` of the output is equal to
    *   `paddings(D, 0) + input.shape(D) + paddings(D, 1)`.
    *
    *   For example:
    *   {{{
    *     // 'input' = [[1, 2, 3], [4, 5, 6]]
    *     // 'paddings' = [[1, 1], [2, 2]]
    *
    *     pad(input, paddings, ConstantPadding(0)) ==>
    *       [[0, 0, 0, 0, 0, 0, 0],
    *        [0, 0, 1, 2, 3, 0, 0],
    *        [0, 0, 4, 5, 6, 0, 0],
    *        [0, 0, 0, 0, 0, 0, 0]]
    *
    *     pad(input, paddings, ReflectivePadding) ==>
    *       [[6, 5, 4, 5, 6, 5, 4],
    *        [3, 2, 1, 2, 3, 2, 1],
    *        [6, 5, 4, 5, 6, 5, 4],
    *        [3, 2, 1, 2, 3, 2, 1]]
    *
    *     pad(input, paddings, SymmetricPadding) ==>
    *       [[2, 1, 1, 2, 3, 3, 2],
    *        [2, 1, 1, 2, 3, 3, 2],
    *        [5, 4, 4, 5, 6, 6, 5],
    *        [5, 4, 4, 5, 6, 6, 5]]
    *   }}}
    *
    * @define OpDocBasicReshape
    *   The `reshape` op reshapes a tensor.
    *
    *   Given `input`, the op returns a tensor that has the same values as `input` but has shape `shape`. If one
    *   component of `shape` is the special value `-1`, then the size of that dimension is computed so that the total
    *   size remains constant. In particular, a `shape` of `[-1]` flattens a tensor into a one-dimensional tensor. At
    *   most one component of `shape` can be set to `-1`.
    *
    *   If `shape` is a one-dimensional or higher tensor, then the operation returns a tensor with shape `shape` filled
    *   with the values of `input`. In this case, the number of elements implied by `shape` must be the same as the
    *   number of elements in `input`.
    *
    *   For example:
    *   {{{
    *     // Tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9] => It has shape [9]
    *     reshape(t, [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    *
    *     // Tensor 't' is [[[1, 1], [2, 2]],
    *     //                [[3, 3], [4, 4]]] => It has shape [2, 2, 2]
    *     reshape(t, [2, 4] ==> [[1, 1, 2, 2],
    *                            [3, 3, 4, 4]]
    *
    *     // Tensor 't' is [[[1, 1, 1],
    *                        [2, 2, 2]],
    *                       [[3, 3, 3],
    *                        [4, 4, 4]],
    *                       [[5, 5, 5],
    *                        [6, 6, 6]]] => It has shape [3, 2, 3]
    *     reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
    *
    *     // '-1' can also be used to infer the shape. Some examples follow.
    *
    *     // '-1' is inferred to be 9:
    *     reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    *                              [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    *
    *     // '-1' is inferred to be 2:
    *     reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    *                              [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    *
    *     // '-1' is inferred to be 3:
    *     reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
    *                                   [2, 2, 2],
    *                                   [3, 3, 3]],
    *                                  [[4, 4, 4],
    *                                   [5, 5, 5],
    *                                   [6, 6, 6]]]
    *
    *     // Tensor 't' is [7]
    *     // An empty shape passed to 'reshape' will result in a scalar
    *     reshape(t, []) ==> 7
    *   }}}
    *
    * @define OpDocBasicTranspose
    *   The `transpose` op permutes the dimensions of a tensor according to a provided permutation.
    *
    *   The returned tensor's dimension `i` will correspond to `input` dimension `permutation(i)`. If `permutation` is
    *   not provided, then it is set to `(n - 1, ..., 0)`, where `n` is the rank of the input tensor. Hence by default,
    *   the op performs a regular matrix transpose on two-dimensional input tensors.
    *
    *   For example:
    *   {{{
    *     // Tensor 'x' is [[1, 2, 3], [4, 5, 6]]
    *     transpose(x) ==> [[1, 4], [2, 5], [3, 6]]
    *     transpose(x, permutation = Array(1, 0)) ==> [[1, 4], [2, 5], [3, 6]]
    *
    *     // Tensor 'x' is [[[1, 2, 3],
    *     //                 [4, 5, 6]],
    *     //                [[7, 8, 9],
    *     //                 [10, 11, 12]]]
    *     transpose(x, permutation = Array(0, 2, 1)) ==> [[[1,  4], [2,  5], [3,  6]],
    *                                                     [[7, 10], [8, 11], [9, 12]]]
    * }}}
    *
    * @define OpDocBasicMatrixTranspose
    *   The `matrixTranpose` op transposes the last two dimensions of tensor `input`.
    *
    *   For example:
    *   {{{
    *     // Tensor 'x' is [[1, 2, 3], [4, 5, 6]]
    *     matrixTranspose(x) ==> [[1, 4], [2, 5], [3, 6]]
    *
    *     // Tensor 'x' has shape [1, 2, 3, 4]
    *     matrixTranspose(x).shape ==> [1, 2, 4, 3]
    *   }}}
    *
    *   Note that [[Math.matmul]] provides named arguments allowing for transposing the matrices involved in the
    *   multiplication. This is done with minimal cost, and is preferable to using this function. For example:
    *   {{{
    *     matmul(a, b, transposeB = true) // is preferable to:
    *     matmul(a, matrixTranspose(b))
    *   }}}
    *
    * @define OpDocBasicInvertPermutation
    *   The `invertPermutation` op computes the inverse permutation of a tensor.
    *
    *   This op computes the inverse of an index permutation. It takes a one-dimensional integer tensor `input`, which
    *   represents indices of a zero-based array, and swaps each value with its index position. In other words, for an
    *   output tensor `y` and an input tensor `x`, this op computes `y(x(i)) = i`, for `i` in
    *   `[0, 1, ..., x.length - 1]`.
    *
    *   For example:
    *   {{{
    *     // Tensor 't' is [3, 4, 0, 2, 1]
    *     invertPermutation(t) ==> [2, 4, 3, 0, 1]
    *   }}}
    *
    * @define OpDocBasicReverse
    *   The `reverse` op reverses specific dimensions of a tensor.
    *
    *   Given an `input` tensor, and an integer array of axes representing the set of dimensions of `input` to reverse,
    *   this op reverses each dimension `i` of `input`, for which there exists `j` such that  `axes(j) == i`.
    *
    *   `input` can have up to 8 dimensions. The number of dimensions specified in `axes` may be 0 or more entries. If
    *   an index is specified more than once, an 'InvalidArgument' error will be raised.
    *
    *   For example:
    *   {{{
    *     // Tensor 't' is [[[[ 0,  1,  2,  3],
    *     //                  [ 4,  5,  6,  7],
    *     //                  [ 8,  9, 10, 11]],
    *     //                 [[12, 13, 14, 15],
    *     //                  [16, 17, 18, 19],
    *     //                  [20, 21, 22, 23]]]] => It has shape [1, 2, 3, 4]
    *
    *     // 'axes' is [3] or [-1]
    *     reverse(t, axes) ==> [[[[ 3,  2,  1,  0],
    *                             [ 7,  6,  5,  4],
    *                             [ 11, 10, 9,  8]],
    *                            [[15, 14, 13, 12],
    *                             [19, 18, 17, 16],
    *                             [23, 22, 21, 20]]]]
    *
    *     // 'axes' is [1] or [-3]
    *     reverse(t, axes) ==> [[[[12, 13, 14, 15],
    *                             [16, 17, 18, 19],
    *                             [20, 21, 22, 23]],
    *                            [[ 0,  1,  2,  3],
    *                             [ 4,  5,  6,  7],
    *                             [ 8,  9, 10, 11]]]]
    *
    *     // 'axes' is [2] or [-2]
    *     reverse(t, axes) ==> [[[[ 8,  9, 10, 11],
    *                             [ 4,  5,  6,  7],
    *                             [ 0,  1,  2,  3]],
    *                            [[20, 21, 22, 23],
    *                             [16, 17, 18, 19],
    *                             [12, 13, 14, 15]]]]
    *   }}}
    *
    * @define OpDocBasicReverseSequence
    *   The `reverseSequence` op reverses variable length slices.
    *
    *   The op first slices `input` along the dimension `batchAxis`, and for each slice `i`, it reverses the first
    *   `sequenceLengths(i)` elements along the dimension `sequenceAxis`.
    *
    *   The elements of `sequenceLengths` must obey `sequenceLengths(i) <= input.shape(sequenceAxis)`, and it must be a
    *   vector of length `input.shape(batchAxis)`.
    *
    *   The output slice `i` along dimension `batchAxis` is then given by input slice `i`, with the first
    *   `sequenceLengths(i)` slices along dimension `sequenceAxis` reversed.
    *
    *   For example:
    *   {{{
    *     // Given:
    *     // sequenceAxis = 1
    *     // batchAxis = 0
    *     // input.shape = [4, 8, ...]
    *     // sequenceLengths = [7, 2, 3, 5]
    *     // slices of 'input' are reversed on 'sequenceAxis', but only up to 'sequenceLengths':
    *     output(0, 0::7, ---) == input(0, 6::-1::, ---)
    *     output(1, 0::2, ---) == input(1, 1::-1::, ---)
    *     output(2, 0::3, ---) == input(2, 2::-1::, ---)
    *     output(3, 0::5, ---) == input(3, 4::-1::, ---)
    *     // while entries past 'sequenceLengths' are copied through:
    *     output(0, 7::, ---) == input(0, 7::, ---)
    *     output(1, 7::, ---) == input(1, 7::, ---)
    *     output(2, 7::, ---) == input(2, 7::, ---)
    *     output(3, 7::, ---) == input(3, 7::, ---)
    *
    *     // In contrast, given:
    *     // sequenceAxis = 0
    *     // batchAxis = 2
    *     // input.shape = [8, ?, 4, ...]
    *     // sequenceLengths = [7, 2, 3, 5]
    *     // slices of 'input' are reversed on 'sequenceAxis', but only up to 'sequenceLengths':
    *     output(0::7, ::, 0, ---) == input(6::-1::, ::, 0, ---)
    *     output(0::2, ::, 1, ---) == input(1::-1::, ::, 1, ---)
    *     output(0::3, ::, 2, ---) == input(2::-1::, ::, 2, ---)
    *     output(0::5, ::, 3, ---) == input(4::-1::, ::, 3, ---)
    *     // while entries past 'sequenceLengths' are copied through:
    *     output(7::, ::, 0, ---) == input(7::, ::, 0, ---)
    *     output(2::, ::, 1, ---) == input(2::, ::, 1, ---)
    *     output(3::, ::, 2, ---) == input(3::, ::, 2, ---)
    *     output(5::, ::, 3, ---) == input(5::, ::, 3, ---)
    *   }}}
    *
    * @define OpDocBasicSpaceToBatch
    *   The `spaceToBatch` op zero-pads and then rearranges (permutes) blocks of spatial data into batches.
    *
    *   More specifically, the op outputs a copy of the input tensor where values from the `height` and `width`
    *   dimensions are moved to the `batch` dimension. After the zero-padding, both `height` and `width` of the input
    *   must be divisible by `blockSize` (which must be greater than `1`). This is the reverse functionality to that of
    *   [[batchToSpace]].
    *
    *   `input` is a `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    *
    *   `paddings` has shape `[2, 2]`. It specifies the padding of the input with zeros across the spatial dimensions as
    *   follows: `paddings = [[padTop, padBottom], [padLeft, padRight]]`. The effective spatial dimensions of the
    *   zero-padded input tensor will be:
    *     - `heightPad = padTop + height + padBottom`
    *     - `widthPad = padLeft + width + padRight`
    *
    *   `blockSize` indicates the block size:
    *     - Non-overlapping blocks of size `blockSize x blockSize` in the height and width dimensions are rearranged
    *       into the batch dimension at each location.
    *     - The batch dimension size of the output tensor is `batch * blockSize * blockSize`.
    *     - Both `heightPad` and `widthPad` must be divisible by `blockSize`.
    *
    *   The shape of the output will be:
    *   `[batch * blockSize * blockSize, heightPad / blockSize, widthPad / blockSize, depth]`
    *
    *   Some examples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *     // blockSize = 2
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatch(input, blockSize, paddings) ==> [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]  (shape = [4, 1, 1, 1])
    *
    *     // === Example #2 ===
    *     // input = [[[[1, 2, 3], [4,   5,  6]],
    *     //           [[7, 8, 9], [10, 11, 12]]]]  (shape = [1, 2, 2, 3])
    *     // blockSize = 2
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatch(input, blockSize, paddings) ==>
    *       [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [4, 1, 1, 3])
    *
    *     // === Example #3 ===
    *     // input = [[[[ 1],  [2],  [3],  [ 4]],
    *     //           [[ 5],  [6],  [7],  [ 8]],
    *     //           [[ 9], [10], [11],  [12]],
    *     //           [[13], [14], [15],  [16]]]]  (shape = [1, 4, 4, 1])
    *     // blockSize = 2
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatch(input, blockSize, paddings) ==>
    *       [[[[1], [3]], [[ 9], [11]]],
    *        [[[2], [4]], [[10], [12]]],
    *        [[[5], [7]], [[13], [15]]],
    *        [[[6], [8]], [[14], [16]]]]  (shape = [4, 2, 2, 1])
    *
    *     // === Example #4 ===
    *     // input = [[[[ 1],  [2],  [3],  [ 4]],
    *     //           [[ 5],  [6],  [7],  [ 8]]],
    *     //          [[[ 9], [10], [11],  [12]],
    *     //           [[13], [14], [15],  [16]]]]  (shape = [2, 2, 4, 1])
    *     // blockSize = 2
    *     // paddings = [[0, 0], [2, 0]]
    *     spaceToBatch(input, blockSize, paddings) ==>
    *       [[[[0], [1], [3]]], [[[0], [ 9], [11]]],
    *        [[[0], [2], [4]]], [[[0], [10], [12]]],
    *        [[[0], [5], [7]]], [[[0], [13], [15]]],
    *        [[[0], [6], [8]]], [[[0], [14], [16]]]]  (shape = [8, 1, 3, 1])
    *   }}}
    *
    * @define OpDocBasicSpaceToBatchND
    *   The `spaceToBatchND` op divides "spatial" dimensions `[1, ..., M]` of `input` into a grid of blocks with shape
    *   `blockShape`, and interleaves these blocks with the "batch" dimension (`0`) such that, in the output, the
    *   spatial dimensions `[1, ..., M]` correspond to the position within the grid, and the batch dimension combines
    *   both the position within a spatial block and the original batch position. Prior to division into blocks, the
    *   spatial dimensions of the input are optionally zero padded according to `paddings`. This is the reverse
    *   functionality to that of [[batchToSpaceND]].
    *
    *   `input` is an `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *   `spatialShape` has `M` dimensions.
    *
    *   The op is equivalent to the following steps:
    *     1. Zero-pad the st of shape `paddedShape`.
    *     2. Reshape `padded` to `reshapedPadded` of shape:
    *        {{{
    *          [batch] +
    *          [[paddedShape(1) / blockShape(0), blockShape(0), ..., paddedShape(M) / blockShape(M-1), blockShape(M-1)]` +
    *          remainingShape
    *        }}}
    *     3. Permute the dimensions of `reshapedPadded` to produce `permutedReshapedPadded` of shape:
    *        {{{
    *          blockShape +
    *          [batch] +
    *          [paddedShape(1) / blockShape(0), ..., paddedShape(M) / blockShape(M-1)] +
    *          remainingShape
    *        }}}
    *     4. Reshape `permutedReshapedPadded` to flatten `blockShape` into the batch dimension, producing an output
    *        tensor of shape:
    *        {{{
    *          [batch *   product(blockShape)] +
    *          [paddedShape(1) / blockShape(0), ..., paddedShape(M) / blockShape(M-1)] +
    *          remainingShape
    *        }}}
    *
    *   Among others, this op is useful for reducing atrous convolution to regular convolution.
    *
    *   Some examples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *     // blockShape = [2, 2]
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatchND(input, blockShape, paddings) ==>
    *       [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]  (shape = [4, 1, 1, 1])
    *
    *     // === Example #2 ===
    *     // input = [[[[1, 2, 3], [4, 5, 6]],
    *     //           [[7, 8, 9], [10, 11, 12]]]]  (shape = [1, 2, 2, 3])
    *     // blockShape = [2, 2]
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatchND(input, blockShape, paddings) ==>
    *       [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [4, 1, 1, 3])
    *
    *     // === Example #3 ===
    *     // input = [[[[ 1],  [2],  [3],  [ 4]],
    *     //           [[ 5],  [6],  [7],  [ 8]],
    *     //           [[ 9], [10], [11],  [12]],
    *     //           [[13], [14], [15],  [16]]]]  (shape = [1, 4, 4, 1])
    *     // blockShape = [2, 2]
    *     // paddings = [[0, 0], [0, 0]]
    *     spaceToBatchND(input, blockShape, paddings) ==>
    *       [[[[1], [3]], [[ 9], [11]]],
    *        [[[2], [4]], [[10], [12]]],
    *        [[[5], [7]], [[13], [15]]],
    *        [[[6], [8]], [[14], [16]]]]  (shape = [4, 2, 2, 1])
    *
    *     // === Example #4 ===
    *     // input = [[[[ 1],  [2],  [3],  [ 4]],
    *     //           [[ 5],  [6],  [7],  [ 8]]],
    *     //          [[[ 9], [10], [11],  [12]],
    *     //           [[13], [14], [15],  [16]]]]  (shape = [2, 2, 4, 1])
    *     // blockShape = [2, 2]
    *     // paddings = [[0, 0], [2, 0]]
    *     spaceToBatchND(input, blockShape, paddings) ==>
    *       [[[[0], [1], [3]]], [[[0], [ 9], [11]]],
    *        [[[0], [2], [4]]], [[[0], [10], [12]]],
    *        [[[0], [5], [7]]], [[[0], [13], [15]]],
    *        [[[0], [6], [8]]], [[[0], [14], [16]]]]  (shape = [8, 1, 3, 1])
    *   }}}
    *
    * @define OpDocBasicBatchToSpace
    *   The `batchToSpace` op rearranges (permutes) data from batches into blocks of spatial data, followed by cropping.
    *
    *   More specifically, the op outputs a copy of the input tensor where values from the `batch` dimension are moved
    *   in spatial blocks to the `height` and `width` dimensions, followed by cropping along the `height` and `width`
    *   dimensions. This is the reverse functionality to that of [[spaceToBatch]].
    *
    *   `input` is a `4`-dimensional input tensor with shape
    *   `[batch * blockSize * blockSize, heightPad / blockSize, widthPad / blockSize, depth]`.
    *
    *   `crops` has shape `[2, 2]`. It specifies how many elements to crop from the intermediate result across the
    *   spatial dimensions as follows: `crops = [[cropTom, cropBottom], [cropLeft, cropRight]]`. The shape of the output
    *   will be: `[batch, heightPad - cropTom - cropBottom, widthPad - cropLeft - cropRight, depth]`.
    *
    *   Some examples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]  (shape = [4, 1, 1, 1])
    *     // blockSize = 2
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpace(input, blockSize, crops) ==> [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *
    *     // === Example #2 ===
    *     // input = [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [4, 1, 1, 3])
    *     // blockSize = 2
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpace(input, blockSize, crops) ==>
    *       [[[[1, 2, 3], [4,   5,  6]],
    *         [[7, 8, 9], [10, 11, 12]]]]  (shape = [1, 2, 2, 3])
    *
    *     // === Example #3 ===
    *     // input = [[[[1], [3]], [[ 9], [11]]],
    *     //          [[[2], [4]], [[10], [12]]],
    *     //          [[[5], [7]], [[13], [15]]],
    *     //          [[[6], [8]], [[14], [16]]]]  (shape = [4, 2, 2, 1])
    *     // blockSize = 2
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpace(input, blockSize, crops) ==>
    *       [[[[ 1],  [2],  [3],  [ 4]],
    *         [[ 5],  [6],  [7],  [ 8]],
    *         [[ 9], [10], [11],  [12]],
    *         [[13], [14], [15],  [16]]]]  (shape = [1, 4, 4, 1])
    *
    *     // === Example #4 ===
    *     // input = [[[[0], [1], [3]]], [[[0], [ 9], [11]]],
    *     //          [[[0], [2], [4]]], [[[0], [10], [12]]],
    *     //          [[[0], [5], [7]]], [[[0], [13], [15]]],
    *     //          [[[0], [6], [8]]], [[[0], [14], [16]]]]  (shape = [8, 1, 3, 1])
    *     // blockSize = 2
    *     // crops = [[0, 0], [2, 0]]
    *     batchToSpace(input, blockSize, crops) ==>
    *       [[[[ 1],  [2],  [3],  [ 4]],
    *         [[ 5],  [6],  [7],  [ 8]]],
    *        [[[ 9], [10], [11],  [12]],
    *         [[13], [14], [15],  [16]]]]  (shape = [2, 2, 4, 1])
    *   }}}
    *
    * @define OpDocBasicBatchToSpaceND
    *   The `batchToSpaceND` op reshapes the "batch" dimension `0` into `M + 1` dimensions of shape
    *   `blockShape + [batch]` and interleaves these blocks back into the grid defined by the spatial dimensions
    *   `[1, ..., M]`, to obtain a result with the same rank as the input. The spatial dimensions of this intermediate
    *   result are then optionally cropped according to `crops` to produce the output. This is the reverse functionality
    *   to that of [[spaceToBatchND]].
    *
    *   `input` is an `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *   `spatialShape` has `M` dimensions.
    *
    *   The op is equivalent to the following steps:
    *     1. Reshape `input` to `reshaped` of shape:
    *        {{{
    *          [blockShape(0), ..., blockShape(M-1),
    *          batch / product(blockShape),
    *          inputShape(1), ..., inputShape(N-1)]
    *        }}}
    *     2. Permute dimensions of `reshaped` to produce `permuted` of shape:
    *        {{{
    *          [batch / product(blockShape),
    *          inputShape(1), blockShape(0),
    *          ...,
    *          inputShape(N-1), blockShape(M-1),
    *          inputShape(M+1),
    *          ...,
    *          inputShape(N-1)]
    *        }}}
    *     3. Reshape `permuted` to produce `reshapedPermuted` of shape:
    *        {{{
    *          [batch / product(blockShape),
    *          inputShape(1) * blockShape(0),
    *          ...,
    *          inputShape(M) * blockShape(M-1),
    *          ...,
    *          inputShape(M+1),
    *          ...,
    *          inputShape(N-1)]
    *        }}}
    *     4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted` according to `crops` to produce
    *        the output of shape:
    *        {{{
    *          [batch / product(blockShape),
    *           inputShape(1) * blockShape(0) - crops(0, 0) - crops(0, 1),
    *          ...,
    *          inputShape(M) * blockShape(M-1) - crops(M-1, 0) - crops(M-1, 1),
    *          inputShape(M+1),
    *          ...,
    *          inputShape(N-1)]
    *        }}}
    *
    *   Some exaples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]  (shape = [4, 1, 1, 1])
    *     // blockShape = [2, 2]
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpaceND(input, blockShape, crops) ==> [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *
    *     // === Example #2 ===
    *     // input = [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [4, 1, 1, 3])
    *     // blockShape = [2, 2]
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpaceND(input, blockShape, crops) ==>
    *       [[[[1, 2, 3], [ 4,  5,  6]],
    *         [[7, 8, 9], [10, 11, 12]]]]  (shape = [1, 2, 2, 3])
    *
    *     // === Example #3 ===
    *     // input = [[[[1], [3]], [[ 9], [11]]],
    *     //          [[[2], [4]], [[10], [12]]],
    *     //          [[[5], [7]], [[13], [15]]],
    *     //          [[[6], [8]], [[14], [16]]]]  (shape = [4, 2, 2, 1])
    *     // blockShape = [2, 2]
    *     // crops = [[0, 0], [0, 0]]
    *     batchToSpaceND(input, blockShape, crops) ==>
    *       [[[[ 1],  [2],  [3],  [ 4]],
    *         [[ 5],  [6],  [7],  [ 8]],
    *         [[ 9], [10], [11],  [12]],
    *         [[13], [14], [15],  [16]]]]  (shape = [1, 4, 4, 1])
    *
    *     // === Example #4 ===
    *     // input = [[[[0], [1], [3]]], [[[0], [ 9], [11]]],
    *     //          [[[0], [2], [4]]], [[[0], [10], [12]]],
    *     //          [[[0], [5], [7]]], [[[0], [13], [15]]],
    *     //          [[[0], [6], [8]]], [[[0], [14], [16]]]]  (shape = [8, 1, 3, 1])
    *     // blockShape = [2, 2]
    *     // crops = [[0, 0], [2, 0]]
    *     batchToSpaceND(input, blockShape, crops) ==>
    *       [[[[[ 1],  [2],  [3],  [ 4]],
    *          [[ 5],  [6],  [7],  [ 8]]],
    *         [[[ 9], [10], [11],  [12]],
    *          [[13], [14], [15],  [16]]]]  (shape = [2, 2, 4, 1])
    *   }}}
    *
    * @define OpDocBasicRequiredSpaceToBatchPaddingsAndCrops
    *   The `requiredSpaceToBatchPaddingsAndCrops` op calculates the paddings and crops required to make `blockShape`
    *   divide `inputShape`.
    *
    *   This function can be used to calculate a suitable `paddings`/`crops` argument for use with the
    *   [[spaceToBatchND]]/[[batchToSpaceND]] functions.
    *
    *   The returned tensors, `paddings` and `crops` satisfy:
    *     - `paddings(i, 0) == basePaddings(i, 0)`,
    *     - `0 <= paddings(i, 1) - basePaddings(i, 1) < blockShape(i)`,
    *     - `(inputShape(i) + paddings(i, 0) + paddings(i, 1)) % blockShape(i) == 0`,
    *     - `crops(i, 0) == 0`, and
    *     - `crops(i, 1) == paddings(i, 1) - basePaddings(i, 1)`.
    *
    * @define OpDocBasicSpaceToDepth
    *   The `spaceToDepth` op that rearranges blocks of spatial data, into depth.
    *
    *   More specifically, the op outputs a copy of the input tensor where values from the `height` and `width`
    *   dimensions are moved to the `depth` dimension. `blockSize` indicates the input block size and how the data is
    *   moved:
    *     - Non-overlapping blocks of size `blockSize x blockSize` in the height and width dimensions are rearranged
    *       into the depth dimension at each location.
    *     - The depth of the output tensor is `inputDepth * blockSize * blockSize`.
    *     - The input tensor's `height` and `width` must be divisible by `blockSize`.
    *
    *   That is, assuming that `input` is in the shape `[batch, height, width, depth]`, the shape of the output will be:
    *   `[batch, height / blockSize, width / blockSize, depth * block_size * block_size]`.
    *
    *   This op is useful for resizing the activations between convolutions (but keeping all data), e.g., instead of
    *   pooling. It is also useful for training purely convolutional models.
    *
    *   Some examples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *     // blockSize = 2
    *     spaceToDepth(input, blockSize) ==> [[[[1, 2, 3, 4]]]]  (shape = [1, 1, 1, 4])
    *
    *     // === Example #2 ===
    *     // input =  [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [1, 2, 2, 3])
    *     // blockSize = 2
    *     spaceToDepth(input, blockSize) ==>
    *       [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]  (shape = [1, 1, 1, 12])
    *
    *     // === Example #3 ===
    *     // input = [[[[ 1], [ 2], [ 5], [ 6]],
    *     //           [[ 3], [ 4], [ 7], [ 8]],
    *     //           [[ 9], [10], [13], [14]],
    *     //           [[11], [12], [15], [16]]]]  (shape = [1, 4, 4, 1])
    *     // blockSize = 2
    *     spaceToDepth(input, blockSize) ==>
    *       [[[[ 1,  2,  3,  4],
    *          [ 5,  6,  7,  8]],
    *         [[ 9, 10, 11, 12],
    *          [13, 14, 15, 16]]]]  (shape = [1, 2, 2, 4])
    *   }}}
    *
    * @define OpDocBasicDepthToSpace
    *   The `depthToSpace` op rearranges data from depth into blocks of spatial data.
    *
    *   More specifically, the op outputs a copy of the input tensor where values from the `depth` dimension are moved
    *   in spatial blocks to the `height` and `width` dimensions. `blockSize` indicates the input block size and how the
    *   data us moved:
    *     - Chunks of data of size `blockSize * blockSize` from depth are rearranged into non-overlapping blocks of size
    *       `blockSize x blockSize`.
    *     - The width the output tensor is `inputDepth * blockSize`, whereas the height is `inputHeight * blockSize`.
    *     - The depth of the input tensor must be divisible by `blockSize * blockSize`.
    *
    *   That is, assuming that `input` is in the shape `[batch, height, width, depth]`, the shape of the output will be:
    *   `[batch, height * blockSize, width * blockSize, depth / (block_size * block_size)]`.
    *
    *   This op is useful for resizing the activations between convolutions (but keeping all data), e.g., instead of
    *   pooling. It is also useful for training purely convolutional models.
    *
    *   Some examples:
    *   {{{
    *     // === Example #1 ===
    *     // input = [[[[1, 2, 3, 4]]]]  (shape = [1, 1, 1, 4])
    *     // blockSize = 2
    *     depthToSpace(input, blockSize) ==> [[[[1], [2]], [[3], [4]]]]  (shape = [1, 2, 2, 1])
    *
    *     // === Example #2 ===
    *     // input =  [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]  (shape = [1, 1, 1, 12])
    *     // blockSize = 2
    *     depthToSpace(input, blockSize) ==>
    *       [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]  (shape = [1, 2, 2, 3])
    *
    *     // === Example #3 ===
    *     // input = [[[[ 1,  2,  3,  4],
    *     //            [ 5,  6,  7,  8]],
    *     //           [[ 9, 10, 11, 12],
    *     //            [13, 14, 15, 16]]]]  (shape = [1, 2, 2, 4])
    *     // blockSize = 2
    *     depthToSpace(input, blockSize) ==>
    *       [[[[ 1], [ 2], [ 5], [ 6]],
    *         [[ 3], [ 4], [ 7], [ 8]],
    *         [[ 9], [10], [13], [14]],
    *         [[11], [12], [15], [16]]]]  (shape = [1, 4, 4, 1,])
    *   }}}
    *
    * @define OpDocBasicWhere
    *   The `where` op returns locations of `true` values in a boolean tensor.
    *
    *   The op returns the coordinates of true elements in `input`. The coordinates are returned in a 2-D tensor where
    *   the first dimension (rows) represents the number of true elements, and the second dimension (columns) represents
    *   the coordinates of the true elements. Note that the shape of the output tensor can vary depending on how many
    *   true values there are in `input`. Indices are output in row-major order.
    *
    *   For example:
    *   {{{
    *     // 'input' tensor is [[true, false]
    *     //                    [true, false]]
    *     // 'input' has two 'true' values and so the output has two coordinates
    *     // 'input' has rank 2 and so each coordinate has two indices
    *     where(input) ==> [[0, 0],
    *                       [1, 0]]
    *
    *     // `input` tensor is [[[true, false]
    *     //                     [true, false]]
    *     //                    [[false, true]
    *     //                     [false, true]]
    *     //                    [[false, false]
    *     //                     [false, true]]]
    *     // 'input' has 5 'true' values and so the output has 5 coordinates
    *     // 'input' has rank 3 and so each coordinate has three indices
    *     where(input) ==> [[0, 0, 0],
    *                       [0, 1, 0],
    *                       [1, 0, 1],
    *                       [1, 1, 1],
    *                       [2, 1, 1]]
    *   }}}
    *
    * @define OpDocBasicBooleanMask
    *   The `booleanMask` op applies the provided boolean mask to `input`.
    *
    *   In general, `0 < mask.rank = K <= tensor.rank`, and `mask`'s shape must match the first `K` dimensions of
    *   `tensor`'s shape. We then have:
    *   `booleanMask(tensor, mask)(i, j1, --- , jd) = tensor(i1, --- , iK, j1, ---, jd)`, where `(i1, ---, iK)` is the
    *   `i`th `true` entry of `mask` (in row-major order).
    *
    *   For example:
    *   {{{
    *     // 1-D example
    *     tensor = [0, 1, 2, 3]
    *     mask = [True, False, True, False]
    *     booleanMask(tensor, mask) ==> [0, 2]
    *
    *     // 2-D example
    *     tensor = [[1, 2], [3, 4], [5, 6]]
    *     mask = [True, False, True]
    *     booleanMask(tensor, mask) ==> [[1, 2], [5, 6]]
    *   }}}
    *
    * @define OpDocBasicSequenceMask
    *   The `sequenceMask` op returns a mask tensor representing the first `N` positions of each row of a matrix.
    *
    *   For example:
    *   {{{
    *     // 'lengths' = [1, 3, 2]
    *     // 'maxLength' = 5
    *     sequenceMask(lengths, maxLength) ==>
    *       [[true, false, false, false, false],
    *        [true,  true,  true, false, false],
    *        [true,  true, false, false, false]]
    *   }}}
    *
    * @define OpDocBasicIndexedSlicesMask
    *   The `indexedSlicesMask` op masks elements of indexed slices tensors.
    *
    *   Given an indexed slices tensor instance `input`, this function returns another indexed slices tensor
    *   that contains a subset of the slices of `input`. Only the slices at indices not specified in `maskIndices` are
    *   returned.
    *
    *   This is useful when you need to extract a subset of slices from an indexed slices tensor.
    *
    *   For example:
    *   {{{
    *     // 'input' contains slices at indices [12, 26, 37, 45] from a large tensor with shape [1000, 10]
    *     input.indices ==> [12, 26, 37, 45]
    *     input.values.shape ==> [4, 10]
    *
    *     // `output` will be the subset of `input` slices at its second and third indices, and so we want to mask its
    *     // first and last indices (which are at absolute indices 12 and 45)
    *     val output = tf.indexedSlicesMask(input, [12, 45])
    *     output.indices ==> [26, 37]
    *     output.values.shape ==> [2, 10]
    *   }}}
    *
    * @define OpDocBasicUnique
    *   The `unique` op finds unique elements in a one-dimensional tensor.
    *
    *   The op returns a tensor `output` containing all of the unique elements of `input` sorted in the same order that
    *   they occur in `input`. This op also returns a tensor `indices` the same size as `input` that contains the
    *   index of each value of `input` in the unique output `output`. In other words `output(indices(i)) = input(i)`,
    *   for `i` in `[0, 1, ..., input.rank - 1]`.
    *
    *   For example:
    *   {{{
    *     // Tensor 't' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    *     val (output, indices) = unique(t)
    *     // 'output' is [1, 2, 4, 7, 8]
    *     // 'indices' is [0, 0, 1, 2, 2, 2, 3, 4, 4]
    *   }}}
    *
    * @define OpDocBasicUniqueWithCounts
    *   The `uniqueWithCounts` finds unique elements in a one-dimensional tensor.
    *
    *   The op returns a tensor `output` containing all of the unique elements of `input` sorted in the same order that
    *   they occur in `input`. This op also returns a tensor `indices` the same size as `input` that contains the
    *   index of each value of `input` in the unique output `output`. Finally, it returns a third tensor `counts` that
    *   contains the count of each element of `output` in `input`.
    *
    *   For example:
    *   {{{
    *     // Tensor 't' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    *     val (output, indices, counts) = uniqueWithCounts(t)
    *     // 'output' is [1, 2, 4, 7, 8]
    *     // 'indices' is [0, 0, 1, 2, 2, 2, 3, 4, 4]
    *     // 'counts' is [2, 1, 3, 1, 2]
    *   }}}
    *
    * @define OpDocBasicListDiff
    *   The `listDiff` op computes the difference between two lists of numbers or strings.
    *
    *   Given a list `x` and a list `y`, the op returns a list `out` that represents all values that are in `x` but not
    *   in `y`. The returned list `output` is sorted in the same order that the numbers appear in `x` (duplicates are
    *   preserved). The op also returns a list `indices` that represents the position of each `out` element in `x`. In
    *   other words, `output(i) = x(indices(i))`, for `i` in `[0, 1, ..., output.length - 1]`.
    *
    *   For example, given inputs `x = [1, 2, 3, 4, 5, 6]` and `y = [1, 3, 5]`, this op would return
    *   `output = [2, 4, 6]` and `indices = [1, 3, 5]`.
    *
    * @define OpDocBasicGather
    *   The `gather` op gathers slices from `input` axis `axis`, according to `indices`.
    *
    *   `indices` must be an integer tensor of any dimension (usually 0-D or 1-D). The op produces an output tensor with
    *   shape `input.shape[::axis] + indices.shape + input.shape(axis + 1::)`, where:
    *   {{{
    *     // Scalar indices (output has rank = rank(input) - 1)
    *     output(a_0, ..., a_n, b_0, ..., b_n) = input(a_0, ..., a_n, indices, b_0, ..., b_n)
    *
    *     // Vector indices (output has rank = rank(input))
    *     output(a_0, ..., a_n, i, b_0, ..., b_n) = input(a_0, ..., a_n, indices(i), b_0, ..., b_n)
    *
    *     // Higher rank indices (output has rank = rank(input) + rank(indices) - 1)
    *     output(a_0, ..., a_n, i, ..., j, b_0, ..., b_n) = input(a_0, ..., a_n, indices(i, ..., j), b_0, ..., b_n)
    *   }}}
    *
    *   If `indices` is a permutation and `indices.length == input.shape(0)`, then this op will permute `input`
    *   accordingly.
    *
    * @define OpDocBasicGatherND
    *   The `gatherND` op gathers values or slices from `input` according to `indices`.
    *
    *   `indices` is an integer tensor containing indices into `input`.  The last dimension of `indices` can be equal to
    *   at most the rank of `input`, `indices.shape(-1) <= input.rank`. The last dimension of `indices` corresponds to
    *   elements (if `indices.shape(-1) == input.rank`), or slices (if `indices.shape(-1) < input.rank`) along dimension
    *   `indices.shape(-1)` of `input`. The output has shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    *
    *   Some examples follow.
    *
    *   Simple indexing into a matrix:
    *   {{{
    *     input   = [['a', 'b'], ['c', 'd']]
    *     indices = [[0, 0], [1, 1]]
    *     output  = ['a', 'd']
    *   }}}
    *
    *   Slice indexing into a matrix:
    *   {{{
    *     input   = [['a', 'b'], ['c', 'd']]
    *     indices = [[1], [0]]
    *     output  = [['c', 'd'], ['a', 'b']]
    *   }}}
    *
    *   Indexing into a three-dimensional tensor:
    *   {{{
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[1]]
    *     output  = [[['a1', 'b1'], ['c1', 'd1']]]
    *
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[0, 1], [1, 0]]
    *     output  = [['c0', 'd0'], ['a1', 'b1']]
    *
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[0, 0, 1], [1, 0, 1]]
    *     output  = ['b0', 'b1']
    *   }}}
    *
    *   Batched indexing into a matrix:
    *   {{{
    *     input   = [['a', 'b'], ['c', 'd']]
    *     indices = [[[0, 0]], [[0, 1]]]
    *     output  = [['a'], ['b']]
    *   }}}
    *
    *   Batched slice indexing into a matrix:
    *   {{{
    *     input   = [['a', 'b'], ['c', 'd']]
    *     indices = [[[1]], [[0]]]
    *     output  = [[['c', 'd']], [['a', 'b']]]
    *   }}}
    *
    *   Batched indexing into a three-dimensional tensor:
    *   {{{
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[[1]], [[0]]]
    *     output  = [[[['a1', 'b1'], ['c1', 'd1']]],
    *                [[['a0', 'b0'], ['c0', 'd0']]]]
    *
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    *     output  = [[['c0', 'd0'], ['a1', 'b1']],
    *               [['a0', 'b0'], ['c1', 'd1']]]
    *
    *     input   = [[['a0', 'b0'], ['c0', 'd0']],
    *                [['a1', 'b1'], ['c1', 'd1']]]
    *     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    *     output  = [['b0', 'b1'], ['d0', 'c1']]
    *   }}}
    *
    * @define OpDocBasicScatterND
    *   The `scatterND` op scatters `updates` into a new (initially zero-valued) tensor, according to `indices`.
    *
    *   The op creates a new tensor by applying sparse `updates` to individual values or slices within a zero-valued
    *   tensor of the given `shape`, according to indices. It is the inverse of the [[gatherND]] op, which extracts
    *   values or slices from a given tensor.
    *
    *   '''WARNING:''' The order in which the updates are applied is non-deterministic, and so the output will be
    *   non-deterministic if `indices` contains duplicates.
    *
    *   `indices` is an integer tensor containing indices into a new tensor of shape `shape`. The last dimension of
    *   `indices` can be at most the rank of `shape`: `indices.shape(-1) <= shape.rank`. The last dimension of `indices`
    *   corresponds to indices into elements (if `indices.shape(-1) == shape.rank`) or slices (if
    *   `indices.shape(-1) < shape.rank`) along dimension `indices.shape(-1)` of `shape`.
    *
    *   `updates` is a tensor with shape `indices.shape(::-1) + shape(indices.shape(-1)::)`.
    *
    *   The simplest form of scatter is to insert individual elements in a tensor by index. For example, say we want to
    *   insert `4` scattered elements in a rank-`1` tensor with `8` elements.
    *
    *   <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    *     <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
    *   </div>
    *
    *   In Scala, this scatter operation would look like this:
    *   {{{
    *     val indices = constant(Tensor(Tensor(4), Tensor(3), Tensor(1), Tensor(7)))
    *     val updates = constant(Tensor(9, 10, 11, 12))
    *     val shape = constant(Tensor(8))
    *     scatterND(indices, updates, shape) ==> [0, 11, 0, 10, 9, 0, 0, 12]
    *   }}}
    *
    *   We can also, insert entire slices of a higher rank tensor all at once. For example, say we want to insert two
    *   slices in the first dimension of a rank-`3` tensor with two matrices of new values.
    *
    *   <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    *     <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
    *   </div>
    *
    *   In Scala, this scatter operation would look like this:
    *   {{{
    *     val indices = constant(Tensor(Tensor(0), Tensor(2)))
    *     val updates = constant(Tensor(Tensor(Tensor(5, 5, 5, 5), Tensor(6, 6, 6, 6),
    *                                          Tensor(7, 7, 7, 7), Tensor(8, 8, 8, 8))
    *                                   Tensor(Tensor(5, 5, 5, 5), Tensor(6, 6, 6, 6),
    *                                          Tensor(7, 7, 7, 7), Tensor(8, 8, 8, 8))))
    *     val shape = constant(Tensor(4, 4, 4))
    *     scatterND(indices, updates, shape) ==>
    *       [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    *        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    *        [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    *        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    *   }}}
    *
    * @define OpDocBasicSlice
    *   The `slice` op returns a slice from `input`.
    *
    *   The op output is a tensor with dimensions described by `size`, whose values are extracted from `input`, starting
    *   at the offsets in `begin`.
    *
    *   Requirements:
    *
    *     - `0 <= begin(i) <= begin(i) + size(i) <= Di, for i in [0, n)`, where `Di` corresponds to the size of
    *       the `i`th dimension of `input` and `n` corresponds to the rank of `input`.
    *
    * @define OpDocBasicStridedSlice
    *   The `stridedSlice` op  returns a strided slice from `input`.
    *
    *   Note that most users will want to use the `apply` or the `slice` method of tensors rather than this function
    *   directly, as the interface of those methods is much simpler.
    *
    *   The goal of the op is to produce a new tensor with a subset of the elements from the `n`-dimensional `input`
    *   tensor. The subset is chosen using a sequence of `m` sparse specifications encoded into the arguments of this
    *   function. Note that, in some cases, `m` could be equal to `n`, but this need not be the case.
    *   Each range specification entry can be one of the following:
    *
    *     - An ellipsis (`---` or `Ellipsis`). Ellipses are used to represent zero or more dimensions of a
    *       full-dimension selection and are produced using `ellipsisMask`. For example, `foo(---)` is the identity
    *       slice.
    *     - A new axis (`NewAxis`). New axes are used to insert new dimensions of size `1` and are produced using
    *       `newAxisMask`. For example, `foo(NewAxis, ---)`, where `foo` has shape `[3, 4]`, produces a new tensor with
    *       shape `[1, 3, 4]`.
    *     - A single index (`Index`). This is used to keep only elements that have a given index. For example, if `foo`
    *       is a tensor with shape `[5, 6]`, `foo(2, ::)` produces a tensor with shape `[6]`. This is encoded in `begin`
    *       and `end` (where `end` has to be equal to `begin + 1`) and in the `shrinkAxisMask` (since an axis is being
    *       shrinked).
    *     - A slice (`Slice`). Slices define a range with a `start`, an `end`, and a `step` size. They are used to
    *       specify which elements to choose from a given dimension. `step` (sometimes called "stride") can be any
    *       integer, but `0`. `begin` is an integer which represents the index of the first value to select, while `end`
    *       represents the index of the last value to select (exclusive). The number of values selected in each
    *       dimension is `end - begin` if `step > 0` and `begin - end` if `step < 0`. `begin` and `end` can be negative,
    *       where `-1` corresponds to the last element, `-2` to the second to last, etc. `beginMask` controls whether to
    *       replace the explicitly provided `begin` with an implicit effective value of: `0` if `step > 0`, and `-1` if
    *       `step < 0`. `endMask` is analogous, but produces the number required to create the largest open interval.
    *       There is currently no way to create begin masks and end masks in the Scala Indexer API. Values of `0` and
    *       `-1` should instead be appropriately used for the `begin` value. The `endMask` functionality is not
    *       currently supported at all since `foo(0 :: )` should return all elements of `foo`, whereas `foo(0 :: -1)`
    *       will return all except the last one.
    *
    *   Requirements:
    *
    *     - `0 != strides(i),` for `i` in `[0, m)` (i.e., no stride should be equal to `0`).
    *     - `ellipsisMask` must be a power of two (i.e., only one ellipsis used).
    *
    *   Each conceptual range specification is encoded in the op's arguments. The encoding is best understood by
    *   considering a non-trivial example. In particular:
    *
    *   {{{
    *     // 'foo' is a tensor with shape '[5, 5, 5, 5, 5, 5]'
    *     foo(1, 2 :: 4, NewAxis, ---, 0 :: -1 :: -3, ::) will be encoded as:
    *     begin = [1, 2, x, x, 0, x] // Where "x" denotes that this value is ignored (we usually simply set it to 0)
    *     end = [2, 4, x, x, -3, x]
    *     strides = [1, 1, x, x, -1, 1]
    *     beginMask = 1 << 4 | 1 << 5 = 48
    *     endMask = 1 << 5 = 32
    *     ellipsisMask = 1 << 3 = 8
    *     newAxisMask = 1 << 2 = 4
    *     shrinkAxisMask = 1 << 0 = 1
    *     // The final shape of the slice becomes '[2, 1, 5, 5, 2, 5]'
    *   }}}
    *
    *   Let us walk step by step through each argument specification in the example slice:
    *
    *     1. The first argument is turned into `begin = 1`, `end = begin + 1 = 2`, `strides = 1`, and the first bit of
    *        `shrinkAxisMask` set to `1` (i.e., `shrinkAxisMask |= 1 << 0`). Setting the bit of `shrinkAxisMask` to `1`
    *        makes sure this argument is treated differently than `1 :: 2`, which would not shrink the corresponding
    *        axis.
    *     2. The second argument contributes `2` to `begin`, `4` to `end`, and `1` to `strides`. All masks have zero
    *        bits contributed.
    *     3. The third argument sets the third bit of `newAxisMask` to `1` (i.e., `newAxisMask |= 1 << 2`).
    *     4. The fourth argument sets the fourth bit of `ellipsisMask` to `1` (i.e., `ellipsisMask |= 1 << 3`).
    *     5. The fifth argument contributes `0` to `begin`, `-3` to `end`, and `-1` to `strides`. It shows the use of
    *        negative indices. A negative index `i` associated with a dimension that has size `s` is converted to a
    *        positive index `s + i`. So `-1` becomes `s - 1` (i.e., the last element index). This conversion is done
    *        internally and so `begin`, `end`, and `strides` are allowed to have negative values.
    *     6. The sixth argument indicates that the entire contents of the corresponding dimension are selected. It sets
    *        the sixth bit of `beginMask` and `endMask` to `1` (i.e., `beginMask |= 1 << 6` and `endMask |= 1 << 6`).
    *
    * @define OpDocBasicStridedSliceAssign
    *   The `stridedSliceAssign` op assigns a value to a slice of `input`.
    *
    *   Note that, currently, `input` is required to be a resource. The arguments of this function work in the same way
    *   as the corresponding arguments of `stridedSlice`;
    *
    *   '''NOTE:''' The created op currently does not support broadcasting and so `value`'s shape must be equal to the
    *   shape produced by the slice of `input`.
    *
    * @define OpDocBasicCheckNumerics
    *   The `checkNumerics` op checks a tensor for `NaN` and `Inf` values.
    *
    *   When run, reports an `InvalidArgument` error if `input` has any values that are not-a-number (`NaN`) or infinity
    *   (`Inf`). Otherwise, it acts as an identity op and passes `input` to the output, as-is.
    *
    * @define OpDocBasicEditDistance
    *   The `editDistance` op computes the Levenshtein distance between sequences.
    *
    *   The op takes variable-length sequences (`hypothesis` and `truth`), each provided as a `SparseTensor`, and
    *   computes the Levenshtein distance between them. The op can also normalize the edit distance using the length of
    *   `truth` by setting `normalize` to `true`.
    *
    *   For example:
    *   {{{
    *     // 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
    *     //   [0, 0] = ["a"]
    *     //   [0, 1] = ["b"]
    *     val hypothesis = SparseOutput(Tensor(Tensor(0, 0, 0), Tensor(1, 0, 0)), Tensor("a", "b"), Tensor(2, 1, 1))
    *     // 'truth' is a tensor of shape `[2, 2]` with variable-length values:
    *     //   [0, 0] = []
    *     //   [0, 1] = ["a"]
    *     //   [1, 0] = ["b", "c"]
    *     //   [1, 1] = ["a"]
    *     val truth = SparseOutput(
    *         Tensor(Tensor(0, 1, 0), Tensor(1, 0, 0), Tensor(1, 0, 1), Tensor(1, 1, 0)),
    *         Tensor("a", "b", "c", "a"),
    *         Tensor(2, 2, 2))
    *     val normalize = true
    *
    *     // 'output' is a tensor of shape `[2, 2]` with edit distances normalized by the `truth` lengths, and contains
    *     // the values `[[inf, 1.0], [0.5, 1.0]]`. The reason behind each value is:
    *     //   - (0, 0): no truth,
    *     //   - (0, 1): no hypothesis,
    *     //   - (1, 0): addition,
    *     //   - (1, 1): no hypothesis.
    *     val output = editDistance(hypothesis, truth, normalize)
    *   }}}
    *
    * @define OpDocBasicOneHot
    *   The `oneHot` op returns a one-hot tensor.
    *
    *   The locations represented by indices in `indices` take value `onValue`, while all other locations take value
    *   `offValue`. `onValue` and `offValue` must have matching data types. If `dataType` is also provided, they must be
    *   the same data type as specified by `dataType`.
    *
    *   If the input `indices` is rank `N`, the output will have rank `N+1`. The new axis is created at dimension `axis`
    *   (which defaults to the last axis).
    *
    *   If `indices` is a scalar the output shape will be a vector of length `depth`.
    *
    *   If `indices` is a vector of length `features`, the output shape will be:
    *     - `[features, depth]`, if `axis == -1`, and
    *     - `[depth, features]`, if `axis == 0`.
    *
    *   If `indices` is a matrix (batch) with shape `[batch, features]`, the output shape will be:
    *     - `[batch, features, depth]`, if `axis == -1`,
    *     - `[batch, depth, features]`, if `axis == 1`, and
    *     - `[depth, batch, features]`, if `axis == 0`.
    *
    *   If `dataType` is not provided, the function will attempt to assume the data type of `onValue` or `offValue`, if
    *   one or both are passed in. If none of `onValue`, `offValue`, or `dataType` are provided, `dataType` will default
    *   to the `FLOAT32` data type.
    *
    *   Note: If a non-numeric data type output is desired (e.g., `STRING` or `BOOLEAN`), both `onValue` and `offValue`
    *   **must**   be provided to `oneHot`.
    *
    *   For example:
    *   {{{
    *     // 'indices' = [0, 2, -1, 1]
    *     // 'depth' = 3
    *     // 'onValue' = 5.0
    *     // 'offValue' = 0.0
    *     // 'axis' = -1
    *     // The output tensor has shape [4, 3]
    *     oneHot(indices, depth, onValue, offValue, axis) ==>
    *       [[5.0, 0.0, 0.0],  // oneHot(0)
    *        [0.0, 0.0, 5.0],  // oneHot(2)
    *        [0.0, 0.0, 0.0],  // oneHot(-1)
    *        [0.0, 5.0, 0.0]]  // oneHot(1)
    *
    *     // 'indices' = [[0, 2], [1, -1]]
    *     // 'depth' = 3
    *     // 'onValue' = 1.0
    *     // 'offValue' = 0.0
    *     // 'axis' = -1
    *     // The output tensor has shape [2, 2, 3]
    *     oneHot(indices, depth, onValue, offValue, axis) ==>
    *       [[[1.0, 0.0, 0.0],   // oneHot(0)
    *         [0.0, 0.0, 1.0]],  // oneHot(2)
    *        [[0.0, 1.0, 0.0],   // oneHot(1)
    *         [0.0, 0.0, 0.0]]]  // oneHot(-1)
    *   }}}
    *
    * @define OpDocBasicBroadcastGradientArguments
    *   The `broadcastGradientArguments` op returns the reduction indices for computing the gradients of `shape0`
    *   `[operator]` `shape1` with broadcasting.
    *
    *   This is typically used by gradient computations for broadcasting operations.
    *
    * @define OpDocBasicBroadcastTo
    *   The `broadcastTo` op returns a tensor with its shape broadcast to the provided shape. Broadcasting is the
    *   process of making arrays to have compatible shapes for arithmetic operations. Two shapes are compatible if for
    *   each dimension pair they are either equal or one of them is one. When trying to broadcast a tensor to a shape,
    *   the op starts with the trailing dimension, and works its way forward.
    *
    *   For example:
    *   {{{
    *     val x = tf.constant(Tensor(1, 2, 3))
    *     val y = tf.broadcastTo(x, Seq(3, 3))
    *     y ==> [[1, 2, 3],
    *            [1, 2, 3],
    *            [1, 2, 3]]
    *   }}}
    *   In the above example, the input tensor with the shape of `[1, 3]` is broadcasted to the output tensor with a
    *   shape of `[3, 3]`.
    *
    * @define OpDocBasicBroadcastShape
    *   The `broadcastShape` op returns the broadcasted dynamic shape between two provided shapes, corresponding to the
    *   shapes of the two arguments provided to an op that supports broadcasting.
    *
    * @define OpDocBasicMeshGrid
    *   The `meshGrid` op broadcasts parameters for evaluation on an `N`-dimensional grid.
    *
    *   Given `N` one-dimensional coordinate arrays `inputs`, the op returns a list, `outputs`, of `N`-dimensional
    *   coordinate arrays for evaluating expressions on an `N`-dimensional grid.
    *
    *   '''NOTE:''' If `useCartesianIndexing` is set to `true` (the default value), the broadcasting instructions for
    *   the first two dimensions are swapped.
    *
    *   For example:
    *   {{{
    *     // 'x' = [1, 2, 3]
    *     // 'y' = [4, 5, 6]
    *     val (xx, yy) = meshGrid(x, y)
    *     xx ==> [[1, 2, 3],
    *             [1, 2, 3],
    *             [1, 2, 3]]
    *     yy ==> [[4, 5, 6],
    *             [4, 5, 6],
    *             [4, 5, 6]]
    *   }}}
    *
    * @define OpDocBasicStopGradient
    *   The `stopGradient` op stops gradient execution, but otherwise acts as an identity op.
    *
    *   When executed in a graph, this op outputs its input tensor as-is.
    *
    *   When building ops to compute gradients, this op prevents the contribution of its inputs to be taken into
    *   account. Normally, the gradient generator adds ops to a graph to compute the derivatives of a specified 'loss'
    *   by recursively finding out inputs that contributed to its computation. If you insert this op in the graph its
    *   inputs are masked from the gradient generator. They are not taken into account for computing gradients.
    *
    *   This is useful any time you want to compute a value with TensorFlow but need to pretend that the value was a
    *   constant. Some examples include:
    *
    *     - The ''EM'' algorithm where the ''M-step'' should not involve backpropagation through the output of the
    *       ''E-step''.
    *     - Contrastive divergence training of Boltzmann machines where, when differentiating the energy function, the
    *       training must not backpropagate through the graph that generated the samples from the model.
    *     - Adversarial training, where no backprop should happen through the adversarial example generation process.
    *
    * @define OpDocBasicPreventGradient
    *   The `preventGradient` op triggers an error if a gradient is requested.
    *
    *   When executed in a graph, this op outputs its input tensor as-is.
    *
    *   When building ops to compute gradients, the TensorFlow gradient system ill return an error when trying to lookup
    *   the gradient of this op, because no gradient must ever be registered for this function. This op exists to
    *   prevent subtle bugs from silently returning unimplemented gradients in some corner cases.
    */
  private[ops] trait Documentation
}
