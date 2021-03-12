/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops.basic

import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types.{DataType, IsIntOrLong, IsNumeric, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Logging, Op, Output, OutputIndexedSlices, OutputLike, SparseOutput}
import org.platanios.tensorflow.api.ops.NN.CNNDataFormat
import org.platanios.tensorflow.api.ops.math.Math
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.{Tensor, executionContext}
import org.platanios.tensorflow.api.utilities.DefaultsTo.{IntDefault, LongDefault}
import org.platanios.tensorflow.jni.InvalidArgumentException
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import scala.language.postfixOps

/** Contains ops related to tensor shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Manipulation {
  /** $OpDocBasicRank
    *
    * @group BasicOps
    * @param  input    Tensor whose rank to return.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  rank value that `input` has at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def rank[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      optimize: Boolean = true,
      name: String = "Rank"
  ): Output[Int] = {
    input match {
      case o: Output[T] =>
        val inputRank = o.rank
        if (optimize && inputRank != -1) {
          Constructors.constant(Tensor.fill[Int](Shape())(inputRank), name = name)
        } else {
          Op.Builder[Output[T], Output[Int]](
            opType = "Rank",
            name = name,
            input = o
          ).build().output
        }
      case o: OutputIndexedSlices[T] => size(o.denseShape, optimize = optimize, name = name).castTo[Int]
      case o: SparseOutput[T] => size(o.denseShape, optimize = optimize, name = name).castTo[Int]
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
  def size[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      optimize: Boolean = true,
      name: String = "Size"
  ): Output[Long] = {
    input match {
      case o: Output[T] =>
        val inputShape = o.shape
        if (optimize && inputShape.isFullyDefined) {
          Constructors.constant(Tensor.fill[Long](Shape())(inputShape.numElements), name = name)
        } else if (optimize && inputShape.rank > -1 && inputShape.asArray.contains(0)) {
          Constructors.constant(0L, name = name)
        } else {
          Op.Builder[Output[T], Output[Long]](
            opType = "Size",
            name = name,
            input = o
          ).setAttribute("out_type", Long)
              .build().output
        }
      case o: OutputIndexedSlices[T] =>
        Op.nameScope(name) {
          Math.prod(o.denseShape.toLong, Seq(0))
        }
      case o: SparseOutput[T] =>
        Op.nameScope(name) {
          Math.prod(o.denseShape, Seq(0))
        }
    }
  }

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input    Tensor whose shape to return.
    * @param  optimize Boolean flag indicating whether to optimize this op creation by using a constant op with the
    *                  shape of that `input` at graph creation time (instead of execution time), if known.
    * @param  name     Name for the created op.
    * @return Created op output, which is one-dimensional.
    */
  def shape[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      optimize: Boolean = true,
      name: String = "Shape"
  ): Output[Int] = {
    input match {
      case o: Output[T] =>
        val inputShape = o.shape
        if (optimize && inputShape.isFullyDefined) {
          Constructors.constant(inputShape.toTensor, name = name)
        } else {
          Op.Builder[Output[T], Output[Int]](
            opType = "Shape",
            name = name,
            input = o
          ).setAttribute("out_type", Int)
              .build().output
        }
      case o: OutputIndexedSlices[T] => o.denseShape.toInt
      case o: SparseOutput[T] => o.denseShape.toInt
    }
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs Tensors whose shapes to return.
    * @tparam T Data type of the input tensors.
    * @tparam I Data type for the resulting tensors.
    * @return Created op outputs, all of which are one-dimensional.
    */
  def shapeN[T: TF, I: IntDefault : TF : IsIntOrLong](
      inputs: Seq[Output[T]]
  ): Seq[Output[I]] = {
    Op.Builder[Seq[Output[T]], Seq[Output[I]]](
      opType = "ShapeN",
      name = "ShapeN",
      input = inputs
    ).setAttribute("out_type", TF[I].dataType)
        .build().output
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs   Tensors whose shapes to return.
    * @param  dataType Data type of the input tensors.
    * @tparam T Data type of the input tensors.
    * @tparam I Data type for the resulting tensors.
    * @return Created op outputs, all of which are one-dimensional.
    */
  def shapeN[T: TF, I: IsIntOrLong](
      inputs: Seq[Output[T]],
      dataType: DataType[I]
  ): Seq[Output[I]] = {
    implicit val evTF: TF[I] = TF.fromDataType(dataType)
    shapeN[T, I](inputs)
  }

  /** $OpDocBasicIdentity
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def identity[T: TF, OL[A] <: OutputLike[A]](
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
          ).setGradientFn(identityGradient(_, _)(TF[T]))
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

  private[basic] def identityGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
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
  def expandDims[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      axis: Output[I],
      name: String = "ExpandDims"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "ExpandDims",
      name = name,
      input = (input, axis)
    ).setGradientFn(expandDimsGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
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
  def squeeze[T: TF](
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
        .setGradientFn(squeezeGradient(_, _)(TF[T]))
        .build().output
  }

  /** Reshapes the gradient to the shape of the original input. */
  protected def reshapeToInput[T: TF](
      input: Output[T],
      gradient: Output[T]
  ): Output[T] = {
    reshape(gradient, shape(input))
  }

  protected def expandDimsGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reshapeToInput(op.input._1, outputGradient), null)
  }

  protected def squeezeGradient[T: TF](
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
  def stack[T: TF](
      inputs: Seq[Output[T]],
      axis: Int = 0,
      name: String = "Stack"
  ): Output[T] = {
    Op.Builder[Seq[Output[T]], Output[T]](
      opType = "Pack",
      name = name,
      input = inputs
    ).setAttribute("axis", axis)
        .setGradientFn(stackGradient(_, _)(TF[T]))
        .build().output
  }

  protected def stackGradient[T: TF](
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
  def parallelStack[T: TF](
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
  def unstack[T: TF](
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
        .setGradientFn(unstackGradient(_, _)(TF[T]))
        .build().output
  }

  protected def unstackGradient[T: TF](
      op: Op[Output[T], Seq[Output[T]]],
      outputGradient: Seq[Output[T]]
  ): Output[T] = {
    stack(
      inputs = outputGradient.map(_.toOutput),
      axis = op.longAttribute("axis").toInt)
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
  def concatenate[T: TF](
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
      ).setGradientFn[(Seq[OutputLike[T]], Output[Int]), OutputLike[T]](concatenateGradient(_, _)(TF[T]))
          .build().output
    }
  }

  protected def concatenateGradient[T: TF](
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
                concatenationAxis = Constructors.constant(axis)
              }
            case None => ()
          }
          // Using modulus here for convenience since the 'concatenationAxis' value is already verified in the
          // concatenate op implementation to be within the allowed '[-rank, rank)' range.
          val nonNegativeConcatenationAxis = concatenationAxis % rank(inputValues(0))
          // Get the inputs' tensor shapes.
          val shapes = shapeN(inputValues.map(_.toOutput)).map(_.castTo[Int])
          // The magic number of '16' was found through benchmarking a range of sizes on CPUs and a Maxwell Titan X
          // GPU. A speedup was seen in a large majority of cases when switching implementations at N = 16, but it is
          // possible that there will be a small number of performance regressions.
          if (shapes.length > 16) {
            // Extract the size of each input along the concatenation axis.
            val sizes = squeeze(slice(
              stack[Int](shapes, 1),
              stack[Int](Seq(nonNegativeConcatenationAxis, 0)),
              Tensor[Int](1, -1)))
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
          val shapes = inputValues.map(shape(_))
          if (staticConcatenationAxis > 0) {
            // 'nonNegativeConcatenationAxis' > 0. Each input gets OutputIndexedSlices gradients with all the indices,
            // but with the values sliced accordingly. This is like the Output case, except that shape(g.values)(0) is
            // not equal to shape(shapes(i))(0), since only a subset of the axis-0 values are stored.

            // The following creates variables for iteratively slicing a dense gradients tensor.
            // Since shape is 1-D, 'shapeOfShape' is a scalar containing the rank of the inputs.
            val shapeOfShape = shape(shapes(0))
            // Make a vector of length equal to the input rank, with 0's everywhere and 1 in the concatenation axis index.
            val zero = Constructors.zeros[Int](Shape())
            val mask = concatenate(Seq(
              Constructors.fill[Int, Int](expandDims(nonNegativeConcatenationAxis, 0))(zero),
              Constructors.constant[Int](Tensor(1)),
              Constructors.fill[Int, Int](
                shapeOfShape - nonNegativeConcatenationAxis - Constructors.ones[Int](Shape())
              )(zero)
            ), axis = 0)
            var begin = Constructors.fill[Int, Int](shapeOfShape)(zero)
            shapes.map(shape => {
              val newValues = slice(g.values, begin, concatenate[Int](
                Seq(Tensor(-1), slice(shape, 1, -1)), 0))
              begin = Math.add(begin, shape * mask)
              OutputIndexedSlices(g.indices, newValues, shape)
            })
          } else {
            // 'nonNegativeConcatenationAxis' == 0. Each input gets OutputIndexedSlices gradients but only for the
            // relevant indices.
            var start = Constructors.zeros[Int](Shape())
            var end = start
            shapes.map(shape => {
              val shapeConcatenationAxis = gather(shape, nonNegativeConcatenationAxis, axis = 0)
              end = start + shapeConcatenationAxis
              // Compute the 1-D Output of indices relevant for this input.
              val indicesToSelect = squeeze(
                Masking.where(Math.logicalAnd(g.indices >= start, g.indices < end)), axes = Seq(1))
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
  def splitEvenly[T: TF](
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
        .setGradientFn(splitEvenlyGradient(_, _)(TF[T]))
        .build().output
  }

  protected def splitEvenlyGradient[T: TF](
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
  def split[T: TF, I: TF : IsIntOrLong](
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
        .setGradientFn(splitGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def splitGradient[T: TF, I: TF : IsIntOrLong](
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
  def tile[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      multiples: Output[I],
      name: String = "Tile"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Tile",
      name = name,
      input = (input, multiples)
    ).setGradientFn(tileGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def tileGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: OutputLike[T]
  ): (Output[T], Output[I]) = {
    val inputShape = shape(op.input._1).castTo[I]
    // We interleave 'multiples' and 'inputShape' to get 'splitShape', reshape the output gradient to 'splitShape',
    // and reduce along all even dimensions (the tiled dimensions) to get the result with shape 'inputShape'.
    // For example:
    //   inputShape = [20, 30, 40]
    //   multiples = [2, 3, 4]
    //   splitShape = [2, 20, 3, 30, 4, 40]
    //   axes = [0, 2, 4]
    val splitShape = reshape(transpose(stack(Seq(op.input._2, inputShape))), Shape(-1))
    val axes = Math.range(0, size(splitShape).castTo[Int], 2)

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = null

    // Sum reduces grad along the first dimension for indexed slices.
    val (gradient, processedSplitShape) = outputGradient match {
      case g: OutputIndexedSlices[T] =>
        val gradient = Math.unsortedSegmentSum(
          g.values,
          Math.mod(g.indices.castTo[I], inputShape(0)),
          inputShape(0))
        (gradient, concatenate(Seq(Constructors.ones[I](Shape()), splitShape(1 ::)), axis = 0))
      case g => (g, splitShape)
    }
    val inputGradient = Math.sum[T, Int](reshape[T, I](gradient, processedSplitShape), axes)
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
    private[ops] def pad[T: TF, I: TF : IsIntOrLong](
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
    private[api] def pad[T: TF, I: TF : IsIntOrLong](
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
  case class ConstantPadding[V: TF](value: Option[Tensor[V]] = None) extends PaddingMode {
    override private[ops] def pad[T: TF, I: TF : IsIntOrLong](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      val constantValues = value.map(Output.constant[V](_).castTo[T]).getOrElse(Output.zeros[T](Shape()))
      Op.Builder[(Output[T], Output[I], Output[T]), Output[T]](
        opType = "PadV2",
        name = name,
        input = (input, paddings, constantValues)
      ).setGradientFn(padGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
          .build().output
    }

    override private[api] def pad[T: TF, I: TF : IsIntOrLong](
        input: Tensor[T],
        paddings: Tensor[I]
    ): Tensor[T] = {
      Tensor.fromNativeHandle[T](NativeTensorOpsBasic.padV2(
        executionContext.value.nativeHandle, input.nativeHandle, paddings.nativeHandle,
        value.map(_.castTo[T]).getOrElse(Tensor.zeros[T](Shape())).nativeHandle))
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
    override private[ops] def pad[T: TF, I: TF : IsIntOrLong](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "MirrorPad",
        name = name,
        input = (input, paddings)
      ).setAttribute("mode", "REFLECT")
          .setGradientFn(mirrorPadGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
          .build().output
    }

    override private[api] def pad[T: TF, I: TF : IsIntOrLong](
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
    override private[ops] def pad[T: TF, I: TF : IsIntOrLong](
        input: Output[T],
        paddings: Output[I],
        name: String
    ): Output[T] = {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "MirrorPad",
        name = name,
        input = (input, paddings)
      ).setAttribute("mode", "SYMMETRIC")
          .setGradientFn(mirrorPadGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
          .build().output
    }

    override private[api] def pad[T: TF, I: TF : IsIntOrLong](
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
  def pad[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      paddings: Output[I],
      mode: Manipulation.PaddingMode = Manipulation.ConstantPadding(Some(Tensor(0).reshape(Shape()))),
      name: String = "Pad"
  ): Output[T] = {
    mode.pad(input, paddings, name)
  }

  protected def padGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[T]) = {
    // Pad introduces values around the original tensor, and so the gradient function slices the original shape out of
    // the gradient.
    val x = op.input._1
    val a = op.input._2 // == [rank(x), 2]
    // Take a slice of 'a' (the 1st column: [rank(x), 1]).
    val padBefore = slice(a, Tensor[Int](0, 0), stack[Int](Seq[Output[Int]](rank(x), 1)))
    // Make it a one-dimensional tensor and return it.
    val xGradient = slice(outputGradient, reshape(padBefore, Shape(-1)), shape(x).castTo[I])
    (xGradient, null, null)
  }

  protected def mirrorPadGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val gradient = Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "MirrorPadGrad",
      name = "MirrorPadGradient",
      input = (outputGradient, op.input._2)
    ).setAttribute("mode", op.stringAttribute("mode"))
        .setGradientFn(mirrorPadHessian(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
    (gradient, null)
  }

  protected def mirrorPadHessian[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val mode = Manipulation.PaddingMode.fromString(op.stringAttribute("mode"))
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
  def reshape[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      shape: Output[I],
      name: String = "Reshape"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Reshape",
      name = name,
      input = (input, shape)
    ).setGradientFn(reshapeGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def reshapeGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (reshape(outputGradient, shape(op.input._1)), null)
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
  def transpose[T: TF, I: IntDefault : TF : IsIntOrLong](
      input: Output[T],
      permutation: Output[I] = null,
      conjugate: Boolean = false,
      name: String = "Transpose"
  ): Output[T] = {
    val opType = if (conjugate && input.dataType.isComplex) "ConjugateTranspose" else "Transpose"
    if (permutation == null) {
      Op.createWith(nameScope = name) {
        val inputRank = rank(input)
        val zero = Constructors.constant(0)
        val one = Constructors.constant(1)
        val reversePermutation = inputRank - one - Math.range(zero, inputRank, one)
        val transposed = Op.Builder[(Output[T], Output[Int]), Output[T]](
          opType = opType,
          name = name,
          input = (input, reversePermutation)
        ).setGradientFn[(Output[T], Output[Int]), Output[T]]({
          if (opType == "Transpose")
            transposeGradient(_, _)(TF[T], TF[Int], IsIntOrLong[Int])
          else
            conjugateTransposeGradient(_, _)(TF[T], TF[Int], IsIntOrLong[Int])
        }).build()
        // Setting the shape explicitly because transpose is not handled by the shape function.
        val inputShape = transposed.input._1.shape
        if (inputShape != null && inputShape.rank != -1)
          transposed.output.setShape(Shape(inputShape.asArray.reverse))
        transposed.output
      }
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = opType,
        name = name,
        input = (input, permutation)
      ).setGradientFn[(Output[T], Output[I]), Output[T]]({
        if (opType == "Transpose")
          transposeGradient(_, _)(TF[T], TF[I], IsIntOrLong[I])
        else
          conjugateTransposeGradient(_, _)(TF[T], TF[I], IsIntOrLong[I])
      }).build().output
    }
  }

  protected def transposeGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (transpose(outputGradient, invertPermutation(op.input._2)), null)
  }

  protected def conjugateTransposeGradient[T: TF, I: TF : IsIntOrLong](
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
  def matrixTranspose[T: TF](
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
        val inputRankMinus1 = inputRank - Constructors.constant(1)
        val inputRankMinus2 = inputRank - Constructors.constant(2)
        val permutation = concatenate(Seq(
          Math.range(Constructors.constant(0), inputRankMinus2, Constructors.constant(1)),
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
  def invertPermutation[I: TF : IsIntOrLong](
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
  def reverse[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      axes: Output[I],
      name: String = "Reverse"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "ReverseV2",
      name = name,
      input = (input, if (axes.rank < 1) axes else axes(NewAxis))
    ).setGradientFn(reverseGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def reverseGradient[T: TF, I: TF : IsIntOrLong](
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
  def reverseSequence[T: TF, I: TF : IsIntOrLong](
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
        .setGradientFn(reverseSequenceGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def reverseSequenceGradient[T: TF, I: TF : IsIntOrLong](
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
  def spaceToBatch[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      blockSize: Int,
      paddings: Output[I],
      name: String = "SpaceToBatch"
  ): Output[T] = {
    val result = spaceToBatchND(input, Constructors.constant(Tensor(blockSize, blockSize)), paddings, name)
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
  def spaceToBatchND[T: TF, I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
      input: Output[T],
      blockShape: Output[I1],
      paddings: Output[I2],
      name: String = "SpaceToBatchND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "SpaceToBatchND",
      name = name,
      input = (input, blockShape, paddings)
    ).setGradientFn(spaceToBatchNDGradient(_, _)(TF[T], TF[I1], IsIntOrLong[I1], TF[I2], IsIntOrLong[I2]))
        .build().output
  }

  protected def spaceToBatchNDGradient[T: TF, I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
  def batchToSpace[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      blockSize: Int,
      crops: Output[I],
      name: String = "BatchToSpace"
  ): Output[T] = {
    val result = batchToSpaceND(input, Constructors.constant(Tensor(blockSize, blockSize)), crops, name)
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
  def batchToSpaceND[T: TF, I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
      input: Output[T],
      blockShape: Output[I1],
      crops: Output[I2],
      name: String = "BatchToSpaceND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "BatchToSpaceND",
      name = name,
      input = (input, blockShape, crops)
    ).setGradientFn(batchToSpaceNDGradient(_, _)(TF[T], TF[I1], IsIntOrLong[I1], TF[I2], IsIntOrLong[I2]))
        .build().output
  }

  protected def batchToSpaceNDGradient[T: TF, I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
        (Constructors.zeros[Int](Shape(0, 2)), Constructors.zeros[Int](Shape(0, 2)))
      } else {
        inputShape.shape.assertIsCompatibleWith(Shape(numBlockDims))
        val actualBasePaddings = {
          if (basePaddings != null) {
            basePaddings.shape.assertIsCompatibleWith(Shape(numBlockDims, 2))
            basePaddings
          } else {
            Constructors.zeros[Int](Shape(numBlockDims, 2))
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
          val zero = Constructors.zeros[Int](Shape())
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
  def spaceToDepth[T: TF](
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
        .setGradientFn(spaceToDepthGradient(_, _)(TF[T]))
        .build().output
  }

  @throws[InvalidArgumentException]
  protected def spaceToDepthGradient[T: TF](
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
  def depthToSpace[T: TF](
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
        .setGradientFn(depthToSpaceGradient(_, _)(TF[T]))
        .build().output
  }

  @throws[InvalidArgumentException]
  protected def depthToSpaceGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    if (op.stringAttribute("data_format") == "NCHW_VECT_C")
      throw InvalidArgumentException(
        "Cannot compute 'spaceToDepth' gradient with 'NCHW_VECT_C' data format. " +
            "This format requires 'QINT8' data type.")
    spaceToDepth(outputGradient, op.longAttribute("block_size").toInt)
  }

  /** $OpDocBasicGather
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  axis    Tensor containing the axis along which to gather.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def gather[T: TF, I1: TF : IsIntOrLong, I2: IntDefault : TF : IsIntOrLong](
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
        gatherGradient(_, _)(TF[T], TF[I1], IsIntOrLong[I1], TF[I2], IsIntOrLong[I2])
      }).build().output
    } else {
      Op.Builder[(Output[T], Output[I1], Output[Int]), Output[T]](
        opType = "GatherV2",
        name = name,
        input = (input, indices, Tensor.zeros[Int](Shape()))
      ).setGradientFn[(OutputLike[T], Output[I1], Output[Int]), Output[T]]({
        gatherGradient(_, _)(TF[T], TF[I1], IsIntOrLong[I1], TF[Int], IsIntOrLong[Int])
      }).build().output
    }
  }

  protected def gatherGradient[T: TF, I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (OutputLike[T], Output[I1], Output[I2]) = {
    // The input can be large, so we colocate the shape calculation with it.
    // The input can be very large for sparse models and 'shape' raises an exception on the Windows platform whenever
    // any dimension is larger than Int. 'inputShape' is not used in the optimizer 'applySparse' gradients method
    // and so it's fine to convert it back to Int regardless of the truncation.
    val input = op.input._1
    val inputShape = Op.colocateWith(Set(input.op), ignoreExisting = true) {
      shape(input).toLong
    }
    val indices = op.input._2
    val indicesSize = size(indices).toLong.expandDims(0)
    val axis = op.input._3
    val axisStatic = Output.constantValue(axis)
    // For axis 0 gathers, we build appropriately shaped indexed slices.
    if (axisStatic.map(_.scalar).getOrElse(-1) == 0) {
      val valuesShape = concatenate(Seq(indicesSize, inputShape(1 ::)), 0)
      val values = reshape(outputGradient, valuesShape)
      val reshapedIndices = reshape(indices, indicesSize)
      val gradient = OutputIndexedSlices(indices = reshapedIndices.toInt, values = values, denseShape = inputShape.toInt)
      (gradient, null, null)
    } else {
      val zero = Basic.zeros[Long](Shape(1))
      val longAxis = axis.toLong.expandDims(0)
      val inputSize = size(inputShape)
      val outerShape = slice(inputShape, zero, longAxis)
      val outerSize = size(outerShape)
      val valuesShape = ControlFlow.cond(
        longAxis < inputSize - 1,
        () => {
          val innerShape = slice(inputShape, longAxis + 1, inputSize.expandDims(0) - 1)
          val innerSize = size(innerShape)
          concatenate(Seq(outerShape, indicesSize, innerShape), 0)
        },
        () => concatenate(Seq(outerShape, indicesSize), 0)
      )
      val values = reshape(outputGradient, valuesShape)
      val reshapedIndices = reshape(indices, indicesSize)
      // We need to sum up every slice `values(..., i, ...)` corresponding to `input(..., indices(i), ...)`. Since
      // `unsortedSegmentSum` does not support an axis parameter, we transpose the gather dimension to the front, and
      // then use `unsortedSegmentSum` to build a `[gatherAxis, outerAxes, innerAxes]` tensor containing all the
      // gradients affecting each index in `gatherAxis` summed up.
      val outerAxesIndices = Math.range(0L, outerSize)
      val innerAxesIndices = Math.range(outerSize + 1, size(valuesShape))
      val transposeAxes = concatenate(
        Seq(outerSize.expandDims(0), outerAxesIndices, innerAxesIndices), 0)
      val valuesTranspose = transpose(values, transposeAxes)
      val numSegments = inputShape.gather(axis)

      // TODO: [TYPES] !!! Super hacky. Remove in the future.
      implicit val ev: IsNumeric[T] = null

      val inputGradient = Math.unsortedSegmentSum(valuesTranspose, reshapedIndices, numSegments)
      // We now invert the above transpose by moving dimension 0 back to its original position.
      val transposeAxesInverse = concatenate[Long](Seq(outerAxesIndices + 1, zero, innerAxesIndices), 0)
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
  def gatherND[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      indices: Output[I],
      name: String = "GatherND"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "GatherNd",
      name = name,
      input = (input, indices)
    ).setGradientFn[(OutputLike[T], Output[I]), Output[T]]({
      gatherNDGradient(_, _)(TF[T], TF[I], IsIntOrLong[I])
    }).build().output
  }

  protected def gatherNDGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (OutputLike[T], Output[I]) = {
    val indices = op.input._2
    val inputShape = shape(op.input._1)
    if (indices.rank == 2 && indices.shape(-1) == 1) {
      (OutputIndexedSlices(
        indices = squeeze(indices.toInt, axes = Seq(-1)),
        values = outputGradient,
        denseShape = inputShape), null)
    } else {
      (scatterND(indices, outputGradient, inputShape.castTo[I]), null)
    }
  }

  /** Gathers slices from `input` according to `indices` with a leading batch dimension.
    *
    * TODO [OPS]: This needs to be updated at this op now supports multiple leading batch dimensions.
    *
    * This operation assumes that the leading dimensions of `indices` are dense, and computes:
    * {{{
    *   result(i1, ..., in) = input(i1, ..., in-1, indices(i1, ..., in))
    * }}}
    *
    * Therefore, if `input` has shape `[A1, ..., AN, B1, ..., BM]`, and `indices` has shape `[A1, ..., AN-1, C]`, then
    * the resulting tensor will have shape `[A1, ..., AN-1, C, B1, ..., BM]`.
    *
    * In the case in which `indices` is a one-dimensional tensor, this operation is equivalent to `gather`.
    *
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor specifying the indices for the gather. Its values elements must be in the interval
    *                 `[0, input.shape(axis)]`, where `axis` is the last dimension of `indices` itself.
    * @param  name    Namescope for the created ops.
    * @return Tensor containing the gathered elements from `input`.
    * @throws InvalidShapeException If `indices` has unknown rank.
    */
  @throws[InvalidShapeException]
  def batchGather[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      indices: Output[I],
      axis: Int = 1,
      batchDimensionCount: Int = 1,
      name: String = "BatchGather"
  ): Output[T] = {
    val inputRank = input.rank
    val indicesRank = indices.rank

    assert(batchDimensionCount >= 0, "`batchDimensionCount` must be non-negative.")
    assert(inputRank >= 0, "`batchGather` does not allow for inputs with unknown rank.")
    assert(indicesRank >= 0, "`batchGather` does not allow for indices with unknown rank.")
    assert(batchDimensionCount < inputRank, "`batchDimensionCount` must be less than `input.rank`.")
    assert(batchDimensionCount < indicesRank, "`batchDimensionCount` must be less than `indices.rank`.")

    // Adjust the axis to be positive.
    val positiveAxis = if (axis < 0) axis + inputRank else axis

    Op.nameScope(name) {
      val zero = Constructors.zeros[I](Shape())
      val one = Constructors.ones[I](Shape())

      // Handle the axis argument by transposing the axis dimension so that it is the first
      // non-batch dimension, recursively calling `batchGather` with `axis = 0`, and then
      // transposing the result to put the pre-axis dimensions before the indices dimensions.
      if (positiveAxis != batchDimensionCount) {
        val inputRankTensor = Constructors.constant[Int](inputRank).castTo[I]
        val positiveAxisTensor = Constructors.constant[Int](positiveAxis).castTo[I]
        val batchDimensionCountTensor = Constructors.constant[Int](batchDimensionCount).castTo[I]
        // Move `input(axis)` up to `input(batchDimensionCount)`.
        val permutation = concatenate(Seq(
          Math.range(zero, batchDimensionCountTensor, one),
          expandDims(positiveAxisTensor, one),
          Math.range(batchDimensionCountTensor, positiveAxisTensor, one),
          Math.range(one + positiveAxisTensor, inputRankTensor, one)))
        val transposedInput = transpose(input, permutation)
        val result = batchGather(
          transposedInput,
          indices,
          axis = batchDimensionCount,
          batchDimensionCount = batchDimensionCount)
        // Move the result dimensions that correspond to `input(batchDimensionCount, ..., axis - 1)` to just before the
        // dimensions that correspond to `indices(batchDimensionCount, ...)`.
        val indicesRankTensor = Constructors.constant[Int](indicesRank).castTo[I]
        val resultRankTensor = Constructors.constant[Int](result.rank).castTo[I]
        val startTensor = Constructors.constant[Int](indicesRank + positiveAxis - batchDimensionCount).castTo[I]
        val resultPermutation = concatenate(Seq(
          Math.range(zero, batchDimensionCountTensor, one),
          Math.range(indicesRankTensor, startTensor, one),
          Math.range(batchDimensionCountTensor, indicesRankTensor, one),
          Math.range(startTensor, resultRankTensor, one)))
        transpose(result, resultPermutation)
      } else {
        val inputShape = shape(input)
        val castedInputShape = inputShape.castTo[I]
        val indicesShape = shape(indices)
        var batchIndices = indices
        var accumulatedDimValue = Constructors.ones[I](Shape())
        for (i <- batchDimensionCount to 1 by -1) {
          accumulatedDimValue *= castedInputShape(i)
          val dimValue = castedInputShape(i - 1)
          val dimIndices = accumulatedDimValue * Math.range(zero, dimValue, one)
          val dimShape = stack((Seq.fill(i - 1)(one) :+ dimValue) ++ Seq.fill(indicesRank - i)(one))
          batchIndices += reshape(dimIndices, dimShape)
        }
        val batchIndicesShape = shape(batchIndices)
        val flatIndices = reshape(batchIndices, Shape(-1))
        val outerShape = inputShape((batchDimensionCount + 1) ::)
        val flatInnerShape = Math.prod(inputShape(0 :: (batchDimensionCount + 1)), axes = 0, keepDims = false)
        val flatInput = reshape(input, concatenate(Seq(flatInnerShape(NewAxis), outerShape), axis = 0))
        val flatResult = gather(flatInput, flatIndices)
        val result = reshape(flatResult, concatenate(Seq(batchIndicesShape, outerShape), axis = 0))
        var finalShape = indices.shape(0 :: batchDimensionCount).mergeWith(input.shape(0 :: batchDimensionCount))
        finalShape ++= indices.shape(batchDimensionCount ::)
        if (input.rank > batchDimensionCount + 1)
          finalShape ++= input.shape((batchDimensionCount + 1) ::)
        result.setShape(finalShape)
        result
      }
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
  def scatterND[T: TF, I: TF : IsIntOrLong](
      indices: Output[I],
      updates: Output[T],
      shape: Output[I],
      name: String = "ScatterND"
  ): Output[T] = {
    Op.Builder[(Output[I], Output[T], Output[I]), Output[T]](
      opType = "ScatterNd",
      name = name,
      input = (indices, updates, shape)
    ).setGradientFn(scatterNDGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def scatterNDGradient[T: TF, I: TF : IsIntOrLong](
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
  def slice[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      begin: Output[I],
      size: Output[I],
      name: String = "Slice"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I], Output[I]), Output[T]](
      opType = "Slice",
      name = name,
      input = (input, begin, size)
    ).setGradientFn(sliceGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def sliceGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[I]) = {
    // Create an N x 2 padding where the first column represents how many zeros are to be prepended for each
    // dimension, and the second column indicates how many zeros are to be appended. The number of zeros to append
    // corresponds to the shape of the input elementwise-subtracted by both the begin vector and sizes vector. Some
    // more reshaping is needed to assemble this tensor with the right dimensions.
    val inputVector = op.input._1
    val beginVector = op.input._2
    val inputRank = rank(inputVector)
    val padShape = stack[Int](Seq(inputRank, 1))
    val beforePad = reshape(beginVector, padShape)
    val afterPad = reshape(shape(inputVector).castTo[I] - shape(op.output).castTo[I] - beginVector, padShape)
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
  def stridedSlice[T: TF, I: TF : IsIntOrLong](
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
    val stridesWithDefault = if (strides == null) Constructors.onesLike(begin) else strides
    Op.Builder[(Output[T], Output[I], Output[I], Output[I]), Output[T]](
      opType = "StridedSlice",
      name = name,
      input = (input, begin, end, stridesWithDefault)
    ).setAttribute("begin_mask", beginMask)
        .setAttribute("end_mask", endMask)
        .setAttribute("ellipsis_mask", ellipsisMask)
        .setAttribute("new_axis_mask", newAxisMask)
        .setAttribute("shrink_axis_mask", shrinkAxisMask)
        .setGradientFn(stridedSliceGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  protected def stridedSliceGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I], Output[I], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[I], Output[I]) = {
    val inputShape = shape(op.input._1).castTo[I]
    val gradient = Op.Builder[(Output[I], Output[I], Output[I], Output[I], Output[T]), Output[T]](
      opType = "StridedSliceGrad",
      name = "StridedSliceGradient",
      input = (inputShape, op.input._2, op.input._3, op.input._4, outputGradient)
    ).setAttribute("begin_mask", op.longAttribute("begin_mask"))
        .setAttribute("end_mask", op.longAttribute("end_mask"))
        .setAttribute("ellipsis_mask", op.longAttribute("ellipsis_mask"))
        .setAttribute("new_axis_mask", op.longAttribute("new_axis_mask"))
        .setAttribute("shrink_axis_mask", op.longAttribute("shrink_axis_mask"))
        .setGradientFn(stridedSliceHessian(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
    (gradient, null, null, null)
  }

  protected def stridedSliceHessian[T: TF, I: TF : IsIntOrLong](
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
  private[api] def stridedSliceAssign[T: TF, I: TF : IsIntOrLong](
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
    val stridesWithDefault = if (strides != null) Constructors.onesLike(begin) else strides
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
}

object Manipulation extends Manipulation
