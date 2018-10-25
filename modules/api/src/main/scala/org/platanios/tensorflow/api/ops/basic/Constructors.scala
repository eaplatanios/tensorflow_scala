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

package org.platanios.tensorflow.api.ops.basic

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{DataType, IsIntOrLong, IsNumeric, TF}
import org.platanios.tensorflow.api.ops.{Math, Op, Output, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor

import org.tensorflow.framework.AttrValue

/** Contains ops related to constructing tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Constructors {
  /** $OpDocBasicConstant
    *
    * @group BasicOps
    * @param  tensor Constant value.
    * @param  shape  Shape of the resulting tensor.
    * @param  name   Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
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
    * @tparam T Tensor data type.
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
    ).setGradientFn(Manipulation.identityGradient(_, _)(TF.fromDataType(input.dataType)))
        .build().output
  }

  /** $OpDocBasicImmutableConstant
    *
    * @group BasicOps
    * @param  shape            Shape of the resulting tensor.
    * @param  memoryRegionName Name of the read-only memory region used by the tensor. Please refer to the C++
    *                          `NewReadOnlyMemoryRegionFromFile` function in `tensorflow::Env`.
    * @param  name             Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def immutableConstant[T: TF](
      shape: Shape,
      memoryRegionName: String,
      name: String = "ImmutableConstant"
  ): Output[T] = {
    Op.Builder[Unit, Output[T]](
      opType = "ImmutableConst",
      name = name,
      input = ()
    ).setAttribute("dtype", TF[T].dataType)
        .setAttribute("shape", shape)
        .setAttribute("memory_region_name", memoryRegionName)
        .build().output
  }

  /** Creates an empty tensor with the specified shape and data type.
    *
    * @group BasicOps
    * @param  shape      Tensor shape.
    * @param  initialize Boolean indicating whether or not to initialize the new tensor with the default value of data
    *                    type `T`.
    * @param  name       Name for the created op.
    * @tparam T Tensor data type.
    * @return Create empty tensor that may contain arbitrary data.
    */
  def empty[T: TF](
      shape: Output[Int],
      initialize: Boolean = false,
      name: String = "Empty"
  ): Output[T] = {
    Op.Builder[Output[Int], Output[T]](
      opType = "Empty",
      name = name,
      input = shape
    ).setAttribute("dtype", TF[T].dataType)
        .setAttribute("init", initialize)
        .build().output
  }

  /** Creates an empty tensor with the same shape and data type as `input`.
    *
    * @group BasicOps
    * @param  input    Input tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @tparam T Tensor data type.
    * @return Create empty tensor that may contain arbitrary data.
    */
  def emptyLike[T: TF](
      input: Output[T],
      initialize: Boolean = false,
      optimize: Boolean = true,
      name: String = "EmptyLike"
  ): Output[T] = {
    Op.nameScope(name) {
      if (optimize && input.shape.isFullyDefined) {
        // We can produce a zeros tensor independent of the value of 'tensor' since the shape is known statically.
        empty[T](input.shape.toOutput.toInt, initialize)
      } else {
        empty[T](Manipulation.shape(input).toInt, initialize)
      }
    }
  }

  /** $OpDocBasicZeros
    *
    * @group BasicOps
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def zeros[T: TF](shape: Output[Int]): Output[T] = {
    Op.nameScope("Zeros") {
      fill[T, Int](shape)(Tensor.zeros[T](Shape()))
    }
  }

  /** $OpDocBasicZeros
    *
    * @group BasicOps
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def zeros[T: TF, I: TF : IsIntOrLong](shape: Output[I]): Output[T] = {
    Op.nameScope("Zeros") {
      fill[T, I](shape)(Tensor.zeros[T](Shape()))
    }
  }

  /** $OpDocBasicZeros
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def zeros[T](dataType: DataType[T], shape: Output[Int]): Output[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    Op.nameScope("Zeros") {
      fill[T, Int](shape)(Tensor.zeros[T](Shape()))
    }
  }

  /** $OpDocBasicZeros
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def zeros[T, I: TF : IsIntOrLong](
      dataType: DataType[T],
      shape: Output[I]
  ): Output[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    Op.nameScope("Zeros") {
      fill[T, I](shape)(Tensor.zeros[T](Shape()))
    }
  }

  /** $OpDocBasicZerosLike
    *
    * @group BasicOps
    * @param  input    Input tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def zerosLike[T](
      input: Output[T],
      optimize: Boolean = true,
      name: String = "ZerosLike"
  ): Output[T] = {
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a zeros tensor independent of the value of 'tensor' since the shape is known statically.
      Op.nameScope(name) {
        zeros(input.dataType, input.shape)
      }
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
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def ones[T: TF](shape: Output[Int]): Output[T] = {
    Op.nameScope("Ones") {
      fill[T, Int](shape)(Tensor.ones[T](Shape()))
    }
  }

  /** $OpDocBasicOnes
    *
    * @group BasicOps
    * @param  shape Tensor shape.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def ones[T: TF, I: TF : IsIntOrLong](shape: Output[I]): Output[T] = {
    Op.nameScope("Ones") {
      fill[T, I](shape)(Tensor.ones[T](Shape()))
    }
  }

  /** $OpDocBasicOnes
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def ones[T](dataType: DataType[T], shape: Output[Int]): Output[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    Op.nameScope("Ones") {
      fill[T, Int](shape)(Tensor.ones[T](Shape()))
    }
  }

  /** $OpDocBasicOnes
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def ones[T, I: TF : IsIntOrLong](
      dataType: DataType[T],
      shape: Output[I]
  ): Output[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    Op.nameScope("Ones") {
      fill[T, I](shape)(Tensor.ones[T](Shape()))
    }
  }

  /** $OpDocBasicOnesLike
    *
    * @group BasicOps
    * @param  input    Input tensor.
    * @param  optimize Boolean flag indicating whether to optimize this op if the shape of `input` is known at graph
    *                  creation time.
    * @param  name     Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def onesLike[T](
      input: Output[T],
      optimize: Boolean = true,
      name: String = "OnesLike"
  ): Output[T] = {
    if (optimize && input.shape.isFullyDefined) {
      // We can produce a ones tensor independent of the value of 'tensor' since the shape is known statically.
      Op.nameScope(name) {
        ones(input.dataType, input.shape)
      }
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
    * @param  shape Tensor shape.
    * @param  value Value to fill the output tensor.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def fill[T: TF, I: TF : IsIntOrLong](shape: Output[I])(
      value: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[I], Output[T]), Output[T]](
      opType = "Fill",
      name = "Fill",
      input = (shape, value)
    ).setGradientFn(fillGradient(_, _)(TF[T], TF[I], IsIntOrLong[I]))
        .build().output
  }

  /** $OpDocBasicFill
    *
    * @group BasicOps
    * @param  dataType Tensor data type.
    * @param  shape    Tensor shape.
    * @param  value    Value to fill the output tensor.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def fill[T, I: TF : IsIntOrLong](
      dataType: DataType[T],
      shape: Output[I]
  )(
      value: Output[T]
  ): Output[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(dataType)
    fill[T, I](shape)(value)
  }

  protected def fillGradient[T: TF, I: TF : IsIntOrLong](
      op: Op[(Output[I], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[I], Output[T]) = {

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNumeric[T] = null

    (null, Math.sum(outputGradient))
  }

  /** $OpDocBasicPlaceholder
    *
    * @group BasicOps
    * @param  shape Shape of the tensor that will be fed. The shape can be any partially-specified,
    *               or even completely unknown.
    * @param  name  Name for the created op.
    * @tparam T Data type of the elements in the tensor that will be fed.
    * @return Created op output.
    */
  def placeholder[T: TF](
      shape: Shape = null,
      name: String = "Placeholder"
  ): Output[T] = {
    val opBuilder = Op.Builder[Unit, Output[T]](
      opType = "Placeholder",
      name = name,
      input = ()
    ).setAttribute("dtype", TF[T].dataType)
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
    * @tparam T Data type of the elements in the tensor that will be fed.
    * @return Created op output.
    */
  def placeholderWithDefault[T: TF](
      default: Output[T],
      shape: Shape,
      name: String = "PlaceholderWithDefault"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "PlaceholderWithDefault",
      name = name,
      input = default
    ).setAttribute("shape", shape)
        .setGradientFn(Manipulation.identityGradient(_, _)(TF[T]))
        .build().output
  }

  /** $OpDocBasicSparsePlaceholder
    *
    * @group BasicOps
    * @param  shape Shape of the tensor that will be fed. The shape can be any partially-specified, or even
    *               completely unknown. This represents the shape of the dense tensor that corresponds to the sparse
    *               tensor that this placeholder refers to.
    * @param  name  Name for the created op.
    * @tparam T Data type of the elements in the tensor that will be fed.
    * @return Created op output.
    */
  def sparsePlaceholder[T: TF](
      shape: Shape = null,
      name: String = "SparsePlaceholder"
  ): SparseOutput[T] = {
    val shapeWithDefault = {
      if (shape == null)
        placeholder[Long](Shape(-1), s"$name/Shape")
      else
        constant(shape.toTensor.toLong)
    }
    SparseOutput[T](
      indices = placeholder[Long](Shape(-1, -1), s"$name/Indices"),
      values = placeholder[T](Shape(-1), s"$name/Values"),
      denseShape = shapeWithDefault)
  }
}

object Constructors extends Constructors
