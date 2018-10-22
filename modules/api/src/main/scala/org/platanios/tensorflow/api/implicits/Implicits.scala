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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.{core, learn, ops, tensors}
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.DataType
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import scala.reflect._

/** Groups together all the implicits of the API and takes care of their priorities.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with core.Implicits {
  /** Convenient implicit conversion function used to convert devices specified as strings for use with the
    * [[Op.createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceImplicitConversion(device: String): OpSpecification => String = {
    _ => device
  }

  //region Cached Implicits

  implicit val classTagByte       : ClassTag[Byte]          = classTag[Byte]
  implicit val classTagShort      : ClassTag[Short]         = classTag[Short]
  implicit val classTagInt        : ClassTag[Int]           = classTag[Int]
  implicit val classTagLong       : ClassTag[Long]          = classTag[Long]
  implicit val classTagShape      : ClassTag[Shape]         = classTag[Shape]
  implicit val classTagDataTypeAny: ClassTag[DataType[Any]] = classTag[DataType[Any]]

  implicit def booleanToTensor(value: Boolean): Tensor[Boolean] = {
    tensorFromSupportedType[Boolean](value)
  }

  implicit def intToTensor(value: Int): Tensor[Int] = {
    tensorFromSupportedType[Int](value)
  }

  implicit def longToTensor(value: Long): Tensor[Long] = {
    tensorFromSupportedType[Long](value)
  }

  implicit def floatToTensor(value: Float): Tensor[Float] = {
    tensorFromSupportedType[Float](value)
  }

  implicit def doubleToTensor(value: Double): Tensor[Double] = {
    tensorFromSupportedType[Double](value)
  }

  implicit def booleanToOutput(value: Boolean): Output[Boolean] = {
    outputFromSupportedType[Boolean](value)
  }

  implicit def intToOutput(value: Int): Output[Int] = {
    outputFromSupportedType[Int](value)
  }

  implicit def longToOutput(value: Long): Output[Long] = {
    outputFromSupportedType[Long](value)
  }

  implicit def floatToOutput(value: Float): Output[Float] = {
    outputFromSupportedType[Float](value)
  }

  implicit def doubleToOutput(value: Double): Output[Double] = {
    outputFromSupportedType[Double](value)
  }

  implicit def shapeToTensor(shape: Shape): Tensor[Long] = {
    shape.toTensor
  }

  implicit def shapeToOutput(shape: Shape): Output[Long] = {
    shape.toOutput
  }

  implicit def booleanOutputBasicOps(output: Output[Boolean]): BasicOps[Boolean] = {
    new BasicOps[Boolean](output)
  }

  implicit def intOutputBasicOps(output: Output[Int]): BasicOps[Int] = {
    new BasicOps[Int](output)
  }

  implicit def longOutputBasicOps(output: Output[Long]): BasicOps[Long] = {
    new BasicOps[Long](output)
  }

  implicit def floatOutputBasicOps(output: Output[Float]): BasicOps[Float] = {
    new BasicOps[Float](output)
  }

  implicit def doubleOutputBasicOps(output: Output[Double]): BasicOps[Double] = {
    new BasicOps[Double](output)
  }

  implicit def outputBasicOps[T](output: Output[T]): BasicOps[T] = {
    new BasicOps[T](output)
  }

  implicit def booleanOutputMathOps(output: Output[Boolean]): MathOps[Boolean] = {
    new MathOps[Boolean](output)
  }

  implicit def intOutputMathOps(output: Output[Int]): MathOps[Int] = {
    new MathOps[Int](output)
  }

  implicit def longOutputMathOps(output: Output[Long]): MathOps[Long] = {
    new MathOps[Long](output)
  }

  implicit def floatOutputMathOps(output: Output[Float]): MathOps[Float] = {
    new MathOps[Float](output)
  }

  implicit def doubleOutputMathOps(output: Output[Double]): MathOps[Double] = {
    new MathOps[Double](output)
  }

  implicit def outputMathOps[T](output: Output[T]): MathOps[T] = {
    new MathOps[T](output)
  }

  //endregion Cached Implicits
}

private[api] trait LowPriorityImplicits
    extends ops.Implicits
        with tensors.Implicits
        with learn.Implicits {
  implicit def tensorAsUntyped[T](tensor: Tensor[T]): Tensor[Any] = {
    tensor.asInstanceOf[Tensor[Any]]
  }

  implicit def opAsUntyped[I, O](op: Op[I, O]): UntypedOp = {
    op.asInstanceOf[UntypedOp]
  }

  implicit def opUntypedOutputAsOutputLike[I, O](
      op: Op[Seq[Output[Any]], Seq[Output[Any]]]
  ): Op[Seq[OutputLike[Any]], Seq[OutputLike[Any]]] = {
    op.asInstanceOf[Op[Seq[OutputLike[Any]], Seq[OutputLike[Any]]]]
  }

  implicit def outputAsUntyped[T](output: Output[T]): Output[Any] = {
    output.asInstanceOf[Output[Any]]
  }

  implicit def outputLikeAsUntyped[T](outputLike: OutputLike[T]): OutputLike[Any] = {
    outputLike.asInstanceOf[OutputLike[Any]]
  }

  implicit def tensorArrayAsUntyped[T](tensorArray: TensorArray[T]): TensorArray[Any] = {
    tensorArray.asInstanceOf[TensorArray[Any]]
  }

  implicit def variableAsUntyped[T](variable: Variable[T]): Variable[Any] = {
    variable.asInstanceOf[Variable[Any]]
  }

  implicit def opSetAsUntyped[I, O](ops: Set[Op[I, O]]): Set[UntypedOp] = {
    ops.asInstanceOf[Set[UntypedOp]]
  }

  implicit def outputSeqAsUntyped[T](outputs: Seq[Output[T]]): Seq[Output[Any]] = {
    outputs.asInstanceOf[Seq[Output[Any]]]
  }
}

private[api] object Implicits extends Implicits
