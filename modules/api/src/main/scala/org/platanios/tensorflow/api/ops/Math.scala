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
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

import scala.language.postfixOps

/** Contains functions for constructing general math-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Math {
  /** $OpDocMathSelect
    *
    * @group MathOps
    * @param  condition Condition tensor.
    * @param  x         Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may
    *                   have a higher rank, but its first dimension must match the size of `condition`.
    * @param  y         Tensor with the same data type and shape as `t`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def select[T: TF](
      condition: Output[Boolean],
      x: Output[T],
      y: Output[T],
      name: String = "Select"
  ): Output[T] = {
    Op.Builder[(Output[Boolean], Output[T], Output[T]), Output[T]](
      opType = "Select",
      name = name,
      input = (condition, x, y)
    ).setGradientFn(selectGradient(_, _)(TF[T]))
        .build().output
  }

  protected def selectGradient[T: TF](
      op: Op[(Output[Boolean], Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[Boolean], Output[T], Output[T]) = {
    val c = op.input._1
    val x = op.input._2
    val zeros = Basic.zerosLike(x)
    (null, select(c, outputGradient, zeros), select(c, zeros, outputGradient))
  }

  // TODO: [OPS] Incorrect type T in `range`.

  /** $OpDocMathRange
    *
    * @group MathOps
    * @param  start Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  limit Rank 0 (i.e., scalar) tensor that contains the ending value (exclusive) of the number sequence.
    * @param  delta Rank 0 (i.e., scalar) tensor that contains the difference between consecutive numbers in the
    *               sequence.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def range[T: TF : IsNumeric](
      start: Output[T],
      limit: Output[T],
      delta: Output[T] = null,
      name: String = "Range"
  ): Output[T] = {
    val deltaWithDefault = if (delta == null) Basic.ones[T](Shape()) else delta
    Op.Builder[(Output[T], Output[T], Output[T]), Output[T]](
      opType = "Range",
      name = name,
      input = (start, limit, deltaWithDefault)
    ).build().output
  }

  /** $OpDocMathLinspace
    *
    * @group MathOps
    * @param  start          Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  stop           Rank 0 (i.e., scalar) tensor that contains the ending value (inclusive) of the number
    *                        sequence.
    * @param  numberOfValues Rank 0 (i.e., scalar) tensor that contains the number of values in the number sequence.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def linspace[T: TF : IsBFloat16OrFloat32OrFloat64, I: TF : IsInt32OrInt64](
      start: Output[T],
      stop: Output[T],
      numberOfValues: Output[I],
      name: String = "LinSpace"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T], Output[I]), Output[T]](
      opType = "LinSpace",
      name = name,
      input = (start, stop, numberOfValues)
    ).build().output
  }

  /** $OpDocMathAddN
    *
    * @group MathOps
    * @param  inputs Input tensors.
    * @param  name   Created op name.
    * @return Created op output.
    */
  def addN[T: TF : IsNumeric](
      inputs: Seq[Output[T]],
      name: String = "AddN"
  ): Output[T] = {
    if (inputs.length == 1) {
      Basic.identity(inputs(0), name)
    } else {
      Op.Builder[Seq[Output[T]], Output[T]](
        opType = "AddN",
        name = name,
        input = inputs
      ).setGradientFn(addNGradient(_, _)(TF[T], IsNumeric[T]))
          .build().output
    }
  }

  protected def addNGradient[T: TF : IsNumeric](
      op: Op[Seq[Output[T]], Output[T]],
      outputGradient: Output[T]
  ): Seq[Output[T]] = {
    Seq.fill(op.numInputs)(outputGradient)
  }

  /** $OpDocMathAccumulateN
    *
    * @param  inputs Input tensors.
    * @param  shape  Shape of the elements of `inputs` (in case it's not known statically and needs to be retained).
    * @param  name   Created op name.
    * @return Created op output.
    * @throws InvalidArgumentException If any of the inputs has a different data type and/or shape than the rest.
    */
  @throws[InvalidArgumentException]
  def accumulateN[T: TF : IsNumeric](
      inputs: Seq[Output[T]],
      shape: Shape = null,
      name: String = "AccumulateN"
  ): Output[T] = {
    val dataType = inputs.head.dataType
    if (inputs.exists(_.dataType != dataType))
      throw InvalidArgumentException("All input tensors must have the same data type.")
    val inferredShape = if (shape == null) Shape.unknown() else shape
    if (inputs.exists(!_.shape.isCompatibleWith(inferredShape)))
      throw InvalidArgumentException("All input tensors must have the same shape.")
    if (inputs.length == 1 && name == null) {
      inputs.head
    } else if (inputs.length == 1) {
      Basic.identity(inputs.head, name = name)
    } else {
      Op.Builder[Seq[Output[T]], Output[T]](
        opType = "AccumulateNV2",
        name = name,
        input = inputs
      ).setAttribute("shape", shape)
          .setGradientFn(accumulateNGradient(_, _)(TF[T], IsNumeric[T]))
          .build().output
    }
  }

  protected def accumulateNGradient[T: TF : IsNumeric](
      op: Op[Seq[Output[T]], Output[T]],
      outputGradient: Output[T]
  ): Seq[Output[T]] = {
    Seq.fill(op.numInputs)(outputGradient)
  }

  //region Unary Ops

  // TODO: [OPS] Fix documentation of `abs`, `magnituteFloat`, and `magnituteDouble`.

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def abs[T: TF : IsReal, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Abs"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Abs",
        name = name,
        input = o
      ).setGradientFn(absGradient(_, _)(TF[T], IsReal[T]))
          .build().output)
  }

  protected def absGradient[T: TF : IsReal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    multiply(outputGradient, sign(op.input))
  }

  /** $OpDocMathNegate
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def negate[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Negate"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Neg",
        name = name,
        input = o
      ).setGradientFn(negateGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def negateGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    negate(outputGradient)
  }

  /** $OpDocMathReciprocal
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def reciprocal[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Reciprocal"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[OutputLike[T], Output[T]](
        opType = "Reciprocal",
        name = name,
        input = o
      ).setGradientFn(reciprocalGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def reciprocalGradient[T: TF : IsNotQuantized](
      op: Op[OutputLike[T], Output[T]],
      outputGradient: Output[T]
  ): OutputLike[T] = {
    Gradients.unaryHelper[T, OutputLike, OutputLike](
      op.output,
      outputGradient,
      opType = "ReciprocalGrad",
      name = "ReciprocalGradient",
      gradientFn = Some(reciprocalHessian(_, _)(TF[T], IsNotQuantized[T])))
  }

  protected def reciprocalHessian[T: TF : IsNotQuantized](
      op: Op[(Output[T], OutputLike[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], OutputLike[T]) = {
    val a = op.input._1
    val b = op.input._2
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val ca = conjugate(a)
      val cg = conjugate(outputGradient)
      val rg = Gradients.unaryHelper[T, OutputLike, OutputLike](
        ca,
        outputGradient,
        opType = "ReciprocalGrad",
        name = "ReciprocalGradient",
        gradientFn = Some(reciprocalHessian(_, _)(TF[T], IsNotQuantized[T])))
      (Basic.constant(-2).castTo[T] * cg * b * ca, rg)
    }
  }

  /** $OpDocMathSquare
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def square[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Reciprocal"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Square",
        name = name,
        input = o
      ).setGradientFn(squareGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def squareGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    // Using control dependencies to prevent 2*x from being computed too early.
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * (Basic.constant(2).castTo[T] * conjugate(x))
    }
  }

  /** $OpDocMathSqrt
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sqrt[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Sqrt"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[OutputLike[T], Output[T]](
        opType = "Sqrt",
        name = name,
        input = o
      ).setGradientFn(sqrtGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def sqrtGradient[T: TF : IsNotQuantized](
      op: Op[OutputLike[T], Output[T]],
      outputGradient: Output[T]
  ): OutputLike[T] = {
    Gradients.unaryHelper[T, OutputLike, OutputLike](
      op.output,
      outputGradient,
      opType = "SqrtGrad",
      name = "SqrtGradient",
      gradientFn = Some(sqrtHessian(_, _)(TF[T], IsNotQuantized[T])))
  }

  protected def sqrtHessian[T: TF : IsNotQuantized](
      op: Op[(Output[T], OutputLike[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], OutputLike[T]) = {
    val a = op.input._1
    val y = op.output
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val ga = divide(outputGradient, a)
      (negate(conjugate(ga)) * y, Basic.constant(0.5).castTo[T] * ga)
    }
  }

  /** $OpDocMathRsqrt
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def rsqrt[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Rqsrt"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[OutputLike[T], Output[T]](
        opType = "Rsqrt",
        name = name,
        input = o
      ).setGradientFn(rsqrtGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def rsqrtGradient[T: TF : IsNotQuantized](
      op: Op[OutputLike[T], Output[T]],
      outputGradient: Output[T]
  ): OutputLike[T] = {
    Gradients.unaryHelper[T, OutputLike, OutputLike](
      op.output,
      outputGradient,
      opType = "RsqrtGrad",
      name = "RSqrtGradient",
      gradientFn = Some(rsqrtHessian(_, _)(TF[T], IsNotQuantized[T])))
  }

  protected def rsqrtHessian[T: TF : IsNotQuantized](
      op: Op[(Output[T], OutputLike[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], OutputLike[T]) = {
    val a = op.input._1
    val b = op.input._2
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val ca = conjugate(a)
      val cg = conjugate(outputGradient)

      val rg = Gradients.unaryHelper[T, OutputLike, OutputLike](
        ca,
        outputGradient,
        opType = "RsqrtGrad",
        name = "RSqrtGradient",
        gradientFn = Some(rsqrtHessian(_, _)(TF[T], IsNotQuantized[T])))
      (Basic.constant(-1.5).castTo[T] * cg * b * square(ca), rg)
    }
  }

  /** $OpDocMathExp
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def exp[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Exp"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Exp",
        name = name,
        input = o
      ).setGradientFn(expGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def expGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * conjugate(op.output)
    }
  }

  /** $OpDocMathExpm1
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def expm1[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Expm1"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Expm1",
        name = name,
        input = o
      ).setGradientFn(expm1Gradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def expm1Gradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * exp(conjugate(op.input))
    }
  }

  /** $OpDocMathLog
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Log"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Log",
        name = name,
        input = o
      ).setGradientFn(logGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def logGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * reciprocal(conjugate(op.input))
    }
  }

  /** $OpDocMathLog1p
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log1p[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Log1p"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Log1p",
        name = name,
        input = o
      ).setGradientFn(log1pGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def log1pGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      outputGradient * reciprocal(one + conjugate(x))
    }
  }

  /** $OpDocMathSin
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sin[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Sin"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Sin",
        name = name,
        input = o
      ).setGradientFn(sinGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def sinGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * cos(conjugate(op.input))
    }
  }

  /** $OpDocMathCos
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cos[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Cos"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Cos",
        name = name,
        input = o
      ).setGradientFn(cosGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def cosGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      negate(outputGradient) * sin(conjugate(op.input))
    }
  }

  /** $OpDocMathTan
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tan[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Tan"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Tan",
        name = name,
        input = o
      ).setGradientFn(tanGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def tanGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * square(reciprocal(cos(conjugate(op.input))))
    }
  }

  /** $OpDocMathAsin
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asin[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Asin"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Asin",
        name = name,
        input = o
      ).setGradientFn(asinGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def asinGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      outputGradient * reciprocal(sqrt(one - square(conjugate(x))))
    }
  }

  /** $OpDocMathAcos
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acos[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Acos"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Acos",
        name = name,
        input = o
      ).setGradientFn(acosGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def acosGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      negate(outputGradient) * reciprocal(sqrt(one - square(conjugate(x))))
    }
  }

  /** $OpDocMathAtan
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Atan"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Atan",
        name = name,
        input = o
      ).setGradientFn(atanGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def atanGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      outputGradient * reciprocal(one+ square(conjugate(x)))
    }
  }

  /** $OpDocMathSinh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sinh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Sinh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Sinh",
        name = name,
        input = o
      ).setGradientFn(sinhGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def sinhGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * cosh(conjugate(x))
    }
  }

  /** $OpDocMathCosh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cosh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Cosh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Cosh",
        name = name,
        input = o
      ).setGradientFn(coshGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def coshGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * sinh(conjugate(x))
    }
  }

  /** $OpDocMathTanh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tanh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Tanh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[OutputLike[T], Output[T]](
        opType = "Tanh",
        name = name,
        input = o
      ).setGradientFn(tanhGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def tanhGradient[T: TF : IsNotQuantized](
      op: Op[OutputLike[T], Output[T]],
      outputGradient: Output[T]
  ): OutputLike[T] = {
    var y = op.output
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      y = conjugate(y)
      Gradients.unaryHelper[T, OutputLike, OutputLike](
        y,
        outputGradient,
        opType = "TanhGrad",
        name = "TanhGradient",
        gradientFn = Some(tanhHessian(_, _)(TF[T], IsNotQuantized[T])))
    }
  }

  protected def tanhHessian[T: TF : IsNotQuantized](
      op: Op[(Output[T], OutputLike[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], OutputLike[T]) = {
    val a = op.input._1
    val b = op.input._2
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val ca = conjugate(a)
      val cb = conjugate(b)
      val rg = Gradients.unaryHelper[T, OutputLike, OutputLike](
        ca,
        outputGradient,
        opType = "TanhGrad",
        name = "TanhGradient",
        gradientFn = Some(tanhHessian(_, _)(TF[T], IsNotQuantized[T])))
      (Basic.constant(-2.0).castTo[T] * outputGradient * cb * ca, rg)
    }
  }

  /** $OpDocMathAsinh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asinh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "ASinh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Asinh",
        name = name,
        input = o
      ).setGradientFn(asinhGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def asinhGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val y = op.output
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient / cosh(conjugate(y))
    }
  }

  /** $OpDocMathAcosh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acosh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "ACosh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Acosh",
        name = name,
        input = o
      ).setGradientFn(acoshGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def acoshGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val y = op.output
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient / sinh(conjugate(y))
    }
  }

  /** $OpDocMathAtanh
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atanh[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "ATanh"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Atanh",
        name = name,
        input = o
      ).setGradientFn(atanhGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def atanhGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      outputGradient * reciprocal(one - square(conjugate(x)))
    }
  }

  /** $OpDocMathLogGamma
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logGamma[T: TF : IsFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "LogGamma"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Lgamma",
        name = name,
        input = o
      ).setGradientFn(logGammaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
          .build().output)
  }

  protected def logGammaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * digamma(conjugate(x))
    }
  }

  /** $OpDocMathDigamma
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def digamma[T: TF : IsFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Digamma"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Digamma",
        name = name,
        input = o
      ).setGradientFn(digammaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
          .build().output)
  }

  def digammaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val one = Basic.ones[T](Shape())
      outputGradient * polygamma(one, conjugate(x))
    }
  }

  /** $OpDocMathErf
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erf[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Erf"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Erf",
        name = name,
        input = o
      ).setGradientFn(erfGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def erfGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    val twoOverRootPi = Basic.constant(2.0 / math.sqrt(math.Pi)).castTo[T]
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * twoOverRootPi * exp(negate(square(conjugate(x))))
    }
  }

  /** $OpDocMathErfc
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erfc[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Erfc"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Erfc",
        name = name,
        input = o
      ).setGradientFn(erfcGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def erfcGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val x = op.input
    val minusTwoOverRootPi = Basic.constant(-2.0 / math.sqrt(math.Pi)).castTo[T]
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      outputGradient * minusTwoOverRootPi * exp(negate(square(conjugate(x))))
    }
  }

  /** $OpDocMathSigmoid
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sigmoid[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Sigmoid"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[OutputLike[T], Output[T]](
        opType = "Sigmoid",
        name = name,
        input = o
      ).setGradientFn(sigmoidGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def sigmoidGradient[T: TF : IsNotQuantized](
      op: Op[OutputLike[T], Output[T]],
      outputGradient: Output[T]
  ): OutputLike[T] = {
    var y = op.output
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      y = conjugate(y)
      Gradients.unaryHelper[T, OutputLike, OutputLike](
        y,
        outputGradient,
        opType = "SigmoidGrad",
        name = "SigmoidGradient",
        gradientFn = Some(sigmoidHessian(_, _)(TF[T], IsNotQuantized[T])))
    }
  }

  protected def sigmoidHessian[T: TF : IsNotQuantized](
      op: Op[(Output[T], OutputLike[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], OutputLike[T]) = {
    val a = op.input._1
    val b = op.input._2
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val ca = conjugate(a)
      val cb = conjugate(b)
      val gb = outputGradient * cb
      val rg = Gradients.unaryHelper[T, OutputLike, OutputLike](
        ca,
        outputGradient,
        opType = "SigmoidGrad",
        name = "SigmoidGradient",
        gradientFn = Some(sigmoidHessian(_, _)(TF[T], IsNotQuantized[T])))
      (subtract(gb, Basic.constant(-2.0).castTo[T] * gb * ca), rg)
    }
  }

  /** $OpDocMathLogSigmoid
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logSigmoid[T: TF : IsDecimal, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "LogSigmoid"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    Op.nameScope(name) {
      negate(NN.softplus(negate(x)))
    }
  }

  /** $OpDocMathSign
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sign[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Sign"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Sign",
        name = name,
        input = o
      ).setGradientFn(signGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output)
  }

  protected def signGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Basic.zerosLike(op.input)
  }

  /** $OpDocMathRound
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def round[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Round"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Round",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathRoundInt
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def roundInt[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "RoundInt"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Rint",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathFloor
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floor[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Floor"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Floor",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathCeil
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def ceil[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "Ceil"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[T]](
        opType = "Ceil",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathIsNaN
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isNaN[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "IsNaN"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[Boolean] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[Boolean]](
        opType = "IsNan",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathIsInf
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isInf[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "IsInf"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[Boolean] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[Boolean]](
        opType = "IsInf",
        name = name,
        input = o
      ).build().output)
  }

  /** $OpDocMathIsFinite
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isFinite[T: TF : IsFloat16OrFloat32OrFloat64, OL[A] <: OutputLike[A]](
      x: OL[T],
      name: String = "IsFinite"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[Boolean] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[T], Output[Boolean]](
        opType = "IsFinite",
        name = name,
        input = o
      ).build().output)
  }

  //endregion Unary Ops

  //region Binary Ops

  /** Returns `true` if the shapes of `x`, `y`, and `gradient` are all fully specified (i.e., statically known)
    * and equal. */
  protected def shapeFullySpecifiedAndEqual[T: TF](
      x: Output[T],
      y: Output[T],
      gradient: Output[T]
  ): Boolean = {
    x.shape.isFullyDefined &&
        y.shape.isFullyDefined &&
        gradient.shape.isFullyDefined &&
        x.shape == y.shape &&
        x.shape == gradient.shape
  }

  /** $OpDocMathAdd
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def add[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Add"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Add",
      name = name,
      input = (x, y)
    ).setGradientFn(addGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def addGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    if (shapeFullySpecifiedAndEqual(x, y, outputGradient)) {
      (outputGradient, outputGradient)
    } else {
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      (Basic.reshape(sum(outputGradient, rx), xShape),
          Basic.reshape(sum(outputGradient, ry), yShape))
    }
  }

  /** $OpDocMathSubtract
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def subtract[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Subtract"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Sub",
      name = name,
      input = (x, y)
    ).setGradientFn(subtractGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def subtractGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    if (shapeFullySpecifiedAndEqual(x, y, outputGradient)) {
      (outputGradient, -outputGradient)
    } else {
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      (Basic.reshape(sum(outputGradient, rx), xShape),
          Basic.reshape(-sum(outputGradient, ry), yShape))
    }
  }

  /** $OpDocMathMultiply
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def multiply[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Multiply"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Mul",
      name = name,
      input = (x, y)
    ).setGradientFn(multiplyGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def multiplyGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    if (shapeFullySpecifiedAndEqual(x, y, outputGradient) &&
        (outputGradient.dataType == INT32 || outputGradient.dataType == FLOAT32)) {
      (outputGradient * y, outputGradient * x)
    } else {
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      (Basic.reshape(sum(multiply(outputGradient, y), rx), xShape),
          Basic.reshape(sum(multiply(x, outputGradient), ry), yShape))
    }
  }

  /** $OpDocMathDivide
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def divide[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Divide"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Div",
      name = name,
      input = (x, y)
    ).setGradientFn(divideGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def divideGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    (Basic.reshape(sum(divide(outputGradient, y), rx), xShape),
        Basic.reshape(sum(multiply(outputGradient, divide(divide(negate(x), y), y)), ry), yShape))
  }

  /** $OpDocMathFloorDivide
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  @deprecated("Use `truncateDivide` instead.", "0.1")
  def floorDivide[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "FloorDivide"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "FloorDiv",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathTruncateDivide
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateDivide[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "TruncateDivide"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "TruncateDiv",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathRealDivide
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def realDivide[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "RealDivide"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "RealDiv",
      name = name,
      input = (x, y)
    ).setGradientFn(realDivideGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def realDivideGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = conjugate(op.input._1)
    val y = conjugate(op.input._2)
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    (Basic.reshape(sum(realDivide(outputGradient, y), rx), xShape),
        Basic.reshape(sum(multiply(outputGradient, realDivide(realDivide(negate(x), y), y)), ry), yShape))
  }

  /** $OpDocMathSquaredDifference
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def squaredDifference[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "SquaredDifference"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "SquaredDifference",
      name = name,
      input = (x, y)
    ).setGradientFn(squaredDifferenceGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def squaredDifferenceGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    val xGradient = Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val two = Basic.constant(2).castTo[T]
      multiply(scalarMul(two, outputGradient), subtract(x, y))
    }
    (Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(xGradient, ry), yShape))
  }

  /** $OpDocMathMod
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def mod[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Mod"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Mod",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathFloorMod
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floorMod[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "FloorMod"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "FloorMod",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathTruncateMod
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateMod[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "TruncateMod"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "TruncateMod",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathPow
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def pow[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Pow"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Pow",
      name = name,
      input = (x, y)
    ).setGradientFn(powGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def powGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val z = conjugate(op.output)
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    // Avoid false singularity at x = 0.
    val logX = {
      if (x.dataType.isComplex) {
        // real(x) < 0 is fine for the complex case.
        select(notEqual(x, Basic.constant(0).castTo[T]), log(x), Basic.zerosLike(x))
      } else {
        // There's no sensible real value to return if x < 0, so we return 0.
        select(greater(x, Basic.constant(0).castTo[T]), log(x), Basic.zerosLike(x))
      }
    }
    (Basic.reshape(sum(outputGradient * y * pow(x, subtract(y, Basic.constant(1).castTo[T])), rx), xShape),
        Basic.reshape(sum(outputGradient * z * logX, ry), yShape))
  }

  // TODO: !!! [OPS] Fix this.

  /** $OpDocMathIgammac
    *
    * @group MathOps
    * @param  a    First input tensor.
    * @param  x    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igammac[T: TF : IsFloat32OrFloat64](
      a: Output[T],
      x: Output[T],
      name: String = "Igammac"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Igammac",
      name = name,
      input = (a, x)
    ).setGradientFn(igammacGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def igammacGradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val igammaGradients = igammaGradient(op, outputGradient)
    (negate(igammaGradients._1), negate(igammaGradients._2))
  }

  /** $OpDocMathIgamma
    *
    * @group MathOps
    * @param  a    First input tensor.
    * @param  x    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igamma[T: TF : IsFloat32OrFloat64](
      a: Output[T],
      x: Output[T],
      name: String = "Igamma"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Igamma",
      name = name,
      input = (a, x)
    ).setGradientFn(igammaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def igammaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val a = op.input._1
    val x = op.input._2
    val aShape = Basic.shape(a)
    val xShape = Basic.shape(x)
    val (ra, rx) = Basic.broadcastGradientArguments(aShape, xShape)
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val partialA = Op.Builder[(Output[T], Output[T]), Output[T]](
        opType = "IgammaGradA",
        name = "IGammaGradA",
        input = (a, x)
      ).build().output
      // Perform operations in log space before summing, because Gamma(a) and Gamma'(a) can grow large.
      val partialX = exp(negate(x) + multiply(subtract(a, Basic.ones[T](Shape())), log(x)) - logGamma(a))
      (Basic.reshape(sum(multiply(partialA, outputGradient), ra), aShape),
          Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
    }
  }

  /** $OpDocMathZeta
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  q    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def zeta[T: TF : IsFloat32OrFloat64](
      x: Output[T],
      q: Output[T],
      name: String = "Zeta"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Zeta",
      name = name,
      input = (x, q)
    ).setGradientFn(zetaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def zetaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val x = conjugate(op.input._1)
      val q = conjugate(op.input._2)
      val xShape = Basic.shape(x)
      val qShape = Basic.shape(q)
      val (_, rq) = Basic.broadcastGradientArguments(xShape, qShape)
      val partialQ = negate(x) * zeta(add(x, Basic.ones[T](Shape())), q)
      (null, Basic.reshape(sum(multiply(partialQ, outputGradient), rq), qShape))
    }
  }

  /** $OpDocMathPolygamma
    *
    * @group MathOps
    * @param  n    First input tensor.
    * @param  x    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def polygamma[T: TF : IsFloat32OrFloat64](
      n: Output[T],
      x: Output[T],
      name: String = "Polygamma"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Polygamma",
      name = name,
      input = (n, x)
    ).setGradientFn(polygammaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def polygammaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val n = conjugate(op.input._1)
      val x = conjugate(op.input._2)
      val nShape = Basic.shape(n)
      val xShape = Basic.shape(x)
      val (_, rx) = Basic.broadcastGradientArguments(nShape, xShape)
      val partialX = polygamma(add(n, Basic.ones[T](Shape())), x)
      (null, Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
    }
  }

  /** $OpDocMathAtan2
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan2[T: TF : IsFloat32OrFloat64](
      x: Output[T],
      y: Output[T],
      name: String = "ATan2"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Atan2",
      name = name,
      input = (x, y)
    ).setGradientFn(atan2Gradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def atan2Gradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    Op.createWith(controlDependencies = Set(outputGradient.op)) {
      val gradientInverse = divide(outputGradient, add(square(x), square(y)))
      (multiply(x, gradientInverse),
          multiply(negate(y), gradientInverse))
    }
  }

  /** $OpDocMathMinimum
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def minimum[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Minimum"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Minimum",
      name = name,
      input = (x, y)
    ).setGradientFn(minimumGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def minimumGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val zeros = Basic.zerosLike(outputGradient)
    val xMask = lessEqual(x, y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    val xGradient = select(xMask, outputGradient, zeros)
    val yGradient = select(xMask, zeros, outputGradient)
    (Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(yGradient, ry), yShape))
  }

  /** $OpDocMathMaximum
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def maximum[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Maximum"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Maximum",
      name = name,
      input = (x, y)
    ).setGradientFn(maximumGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def maximumGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val zeros = Basic.zerosLike(outputGradient)
    val xMask = greaterEqual(x, y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    val xGradient = select(xMask, outputGradient, zeros)
    val yGradient = select(xMask, outputGradient, zeros)
    (Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(yGradient, ry), yShape))
  }

  //endregion Binary Ops

  /** $OpDocMathIncompleteBeta
    *
    * @group MathOps
    * @param  a    First input tensor.
    * @param  b    Second input tensor.
    * @param  x    Third input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def incompleteBeta[T: TF : IsFloat32OrFloat64](
      a: Output[T],
      b: Output[T],
      x: Output[T],
      name: String = "IncompleteBeta"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T], Output[T]), Output[T]](
      opType = "Betainc",
      name = name,
      input = (a, b, x)
    ).setGradientFn(incompleteBetaGradient(_, _)(TF[T], IsFloat32OrFloat64[T]))
        .build().output
  }

  protected def incompleteBetaGradient[T: TF : IsFloat32OrFloat64](
      op: Op[(Output[T], Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T], Output[T]) = {
    // TODO: [GRADIENTS] Mark the derivative w.r.t. a and b as not implemented somehow, or implement it.
    val a = conjugate(op.input._1)
    val b = conjugate(op.input._2)
    val x = conjugate(op.input._3)
    val aShape = Basic.shape(a)
    val xShape = Basic.shape(x)
    val (_, rx) = Basic.broadcastGradientArguments(aShape, xShape)
    // Perform operations in log space before summing, because terms can grow large.
    val logBeta = logGamma(a) + logGamma(b) - logGamma(a + b)
    val one = Basic.ones[T](Shape())
    val partialX = exp(((b - one) * log(one - x)) + ((a - one) * log(x)) - logBeta)
    (null, null, Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
  }

  //region Logical Ops

  /** $OpDocMathLogicalNot
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalNot(
      x: Output[Boolean],
      name: String = "LogicalNot"
  ): Output[Boolean] = {
    Op.Builder[Output[Boolean], Output[Boolean]](
      opType = "LogicalNot",
      name = name,
      input = x
    ).build().output
  }

  /** $OpDocMathLogicalAnd
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalAnd(
      x: Output[Boolean],
      y: Output[Boolean],
      name: String = "LogicalAnd"
  ): Output[Boolean] = {
    Op.Builder[(Output[Boolean], Output[Boolean]), Output[Boolean]](
      opType = "LogicalAnd",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathLogicalOr
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalOr(
      x: Output[Boolean],
      y: Output[Boolean],
      name: String = "LogicalOr"
  ): Output[Boolean] = {
    Op.Builder[(Output[Boolean], Output[Boolean]), Output[Boolean]](
      opType = "LogicalOr",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathLogicalXOr
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalXOr(
      x: Output[Boolean],
      y: Output[Boolean],
      name: String = "LogicalXOr"
  ): Output[Boolean] = {
    logicalAnd(logicalOr(x, y), logicalNot(logicalAnd(x, y)), name = name)
  }

  //endregion Logical Ops

  //region Comparison Ops

  /** $OpDocMathEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def equal[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "Equal"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "Equal",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathNotEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def notEqual[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "NotEqual"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "NotEqual",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocMathApproximatelyEqual
    *
    * @group MathOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def approximatelyEqual[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      tolerance: Float = 0.00001f,
      name: String = "ApproximatelyEqual"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "ApproximateEqual",
      name = name,
      input = (x, y)
    ).setAttribute("tolerance", tolerance)
        .build().output
  }

  /** OpDocMathLess
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def less[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "Less"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "Less",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** OpDocMathLessEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def lessEqual[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "LessEqual"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "LessEqual",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** OpDocMathGreater
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greater[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "Greater"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "Greater",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** OpDocMathGreaterEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greaterEqual[T: TF : IsNumeric](
      x: Output[T],
      y: Output[T],
      name: String = "GreaterEqual"
  ): Output[Boolean] = {
    Op.Builder[(Output[T], Output[T]), Output[Boolean]](
      opType = "GreaterEqual",
      name = name,
      input = (x, y)
    ).build().output
  }

  //endregion Comparison Ops

  //region Reduction Ops

  protected def reductionAxes[T: TF, I: TF : IsInt32OrInt64, OL[A] <: OutputLike[A]](
      tensor: OL[T],
      axes: Output[I]
  ): Output[I] = {
    if (axes != null) {
      axes
    } else {
      // Fast path: Avoid creating range and rank ops if the rank is known statically.
      val reductionAxes = tensor match {
        case o: Output[T] if o.rank == 0 =>
          Basic.constant(Tensor.zeros[Int](Shape(0)))
        case o: Output[T] if o.rank > -1 =>
          Basic.constant(0 until o.rank)
        case o: OutputIndexedSlices[T] if o.denseShape.shape.isFullyDefined =>
          Basic.constant(0 until o.denseShape.shape(0))
        case o: SparseOutput[T] if o.denseShape.shape.isFullyDefined =>
          Basic.constant(0 until o.denseShape.shape(0))
        case _ => // Otherwise, we rely on range and rank to do the right thing at run-time.
          range(0, Basic.rank(tensor))
      }
      // TODO: [TYPES] !!! This is wrong...
      reductionAxes.asInstanceOf[Output[I]]
    }
  }

  protected def safeShapeDiv[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T]
  ): Output[T] = {
    truncateDivide(x, maximum(y, Basic.ones(y.dataType, Shape())))
  }

  /** $OpDocMathSum
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sum[T: TF : IsNumeric, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Sum"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "Sum",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .setGradientFn(sumGradient(_, _)(TF[T], IsNumeric[T], TF[I], IsInt32OrInt64[I]))
          .build().output
    }
  }

  protected def sumGradient[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val input = op.input._1
    val axes = op.input._2
    val rank = input.rank
    // Fast path for when reducing to a scalar and rank is known, which
    // adds only reshape and tile ops (and possibly a shape op too).
    if (rank == 0) {
      (outputGradient, null)
    } else if (rank != -1
        && axes.op.opType == "Const"
        && Output.constantValue(axes).exists(a => {
      a.castTo[Int].entriesIterator.toArray[Int].sameElements((0 until rank).toArray[Int])
    })) {
      // In this case the reduction was over all dimensions.
      val reshapedOutputGradient = Basic.reshape(outputGradient, Shape(Array.fill(rank)(1)))
      val inputShape = {
        // If the shape is not fully defined but the rank is, we use the shape op.
        if (input.shape.isFullyDefined)
          input.shape.toOutput
        else
          Basic.shape(input)
      }
      (Basic.tile(reshapedOutputGradient, inputShape), null)
    } else {
      val inputShape = Basic.shape(input).castTo[Int]
      val outputShapeKeptDimensions = Math.reducedShape(inputShape, axes)
      val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDimensions)
      val reshapedOutputGradient = Basic.reshape(outputGradient, outputShapeKeptDimensions)
      (Basic.tile(reshapedOutputGradient, tileScaling), null)
    }
  }

  /** $OpDocMathMean
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def mean[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Mean"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "Mean",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .setGradientFn(meanGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
          .build().output
    }
  }

  protected def meanGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val sumGrad = sumGradient(op, outputGradient)._1
    val factor = {
      val inputSize = op.input._1.size
      val outputSize = op.output.size
      if (inputSize != -1 && outputSize != -1) {
        Basic.constant(inputSize / scala.math.max(outputSize, 1)).castTo[T]
      } else {
        val inputShape = Basic.shape(op.input._1)
        val outputShape = Basic.shape(op.output)
        safeShapeDiv(prod(inputShape), prod(outputShape)).castTo[T]
      }
    }
    (divide(sumGrad, factor), null)
  }

  /** $OpDocMathProd
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def prod[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Prod"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "Prod",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .setGradientFn(prodGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
          .build().output
    }
  }

  def prodGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    // The gradient can be expressed by dividing the product by each entry of the input tensor, but this approach
    // can't deal with zeros in the input. Here, we avoid this problem by composing the output as a product of two
    // cumulative product operations.
    val inputShape = Basic.shape(op.input._1).castTo[Int]
    // Expand the gradient to the full input shape
    val outputShapeKeptDims = Math.reducedShape(inputShape.castTo[Int], op.input._2)
    val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
    var gradient = outputGradient
    gradient = Basic.reshape(gradient, outputShapeKeptDims)
    gradient = Basic.tile(gradient, tileScaling)

    // Pack all reduced dimensions into a single one, so we can perform the cumulative product ops. If the reduction
    // dimensions list is empty, it defaults to FLOAT32 data type, so we need to cast here. We place all the
    // shape-related ops on the CPU to avoid copying back and forth, and since "listdiff" is a CPU-only op.
    val (permutation, reducedNum, otherNum) = Op.createWith(device = "/cpu:0") {
      val rank = Basic.rank(op.input._1)
      // Reshape the reduction indices for the case where the parameters is a scalar.
      val reductionIndices = floorMod(add(Basic.reshape(op.input._2.castTo[Int], Shape(-1)), rank), rank)
      val reduced = reductionIndices.castTo[Int]
      val indices = range(Basic.constant(0), rank)
      val (other, _) = Basic.listDiff(indices, reduced, indicesDataType = Int)
      (Basic.concatenate(Seq(reduced, other), 0),
          prod(Basic.gather(inputShape, reduced, axis = 0)),
          prod(Basic.gather(inputShape, other, axis = 0)))
    }

    val permuted = Basic.transpose(op.input._1, permutation)
    val permutedShape = Basic.shape(permuted)
    val reshaped = Basic.reshape(permuted, Basic.concatenate(Seq(reducedNum, otherNum)))

    // Calculate the product, leaving out the current entry.
    val left = cumprod(reshaped, axis = 0, exclusive = true)
    val right = cumprod(reshaped, axis = 0, exclusive = true, reverse = true)
    // For complex inputs, the gradient is in the conjugate direction.
    val y = Basic.reshape(multiply(Math.conjugate(left), Math.conjugate(right)), permutedShape)

    // Invert the transpose and reshape operations.
    val output = multiply(gradient, Basic.transpose(y, Basic.invertPermutation(permutation)))
    // Make sure to set the statically known shape information through a reshape.
    (Basic.reshape(output, inputShape), null)
  }

  /** $OpDocMathMin
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def min[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Min"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "Min",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .setGradientFn(minOrMaxGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
          .build().output
    }
  }

  /** $OpDocMathMax
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def max[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Max"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[T], Output[I]), Output[T]](
        opType = "Max",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .setGradientFn(minOrMaxGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
          .build().output
    }
  }

  protected def minOrMaxGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val inputShape = Basic.shape(op.input._1).castTo[Int]
    val outputShapeKeptDims = Math.reducedShape(inputShape, op.input._2)
    val y = Basic.reshape(op.output, outputShapeKeptDims)
    var gradient = outputGradient
    gradient = Basic.reshape(gradient, outputShapeKeptDims)

    // Compute the number of selected (maximum or minimum) elements in each reduction dimension. If there are multiple
    // minimum or maximum elements then the gradient will be divided among them.
    val indicators = equal(y, op.input._1).castTo[T]
    val numberOfSelected = Basic.reshape(sum(indicators, op.input._2), outputShapeKeptDims)

    (multiply(divide(indicators, numberOfSelected), gradient), null)
  }

  /** $OpDocMathAll
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def all[I: IntDefault : TF : IsInt32OrInt64](
      input: Output[Boolean],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "All"
  ): Output[Boolean] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[Boolean], Output[I]), Output[Boolean]](
        opType = "All",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .build().output
    }
  }

  /** $OpDocMathAny
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def any[I: IntDefault : TF : IsInt32OrInt64](
      input: Output[Boolean],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "Any"
  ): Output[Boolean] = {
    if (input.rank == 0) {
      input
    } else {
      Op.Builder[(Output[Boolean], Output[I]), Output[Boolean]](
        opType = "Any",
        name = name,
        input = (input, reductionAxes(input, axes))
      ).setAttribute("keep_dims", keepDims)
          .build().output
    }
  }

  /** $OpDocMathLogSumExp
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def logSumExp[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "LogSumExp"
  ): Output[T] = {
    if (input.rank == 0) {
      input
    } else {
      Op.nameScope(name) {
        val maxValue = Basic.stopGradient(max(input, axes, keepDims = true))
        var result = log(sum(exp(subtract(input, maxValue)), axes, keepDims = keepDims))
        if (!keepDims)
          result += Basic.reshape(maxValue, Basic.shape(result))
        else
          result += maxValue
        result
      }
    }
  }

  /** $OpDocMathCountNonZero
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def countNonZero[T: TF : IsNumeric, I: IntDefault : TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I] = null,
      keepDims: Boolean = false,
      name: String = "CountNonZero"
  ): Output[Long] = {
    Op.nameScope(name) {
      sum(notEqual(input, Basic.zeros[T](Shape())).castTo[Long], axes, keepDims)
    }
  }

  /** $OpDocMathCountNonZero
    *
    * @group MathOps
    * @param  input Input tensor for which to count the number of non-zero entries.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def countNonZeroSparse[T: TF : IsNumeric, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "CountNonZero"
  ): Output[Long] = {
    Op.nameScope(name) {
      val zero = Basic.zeros[T](Shape())
      input match {
        case o: Output[T] => sum(notEqual(o, zero).castTo[Long])
        case o: OutputIndexedSlices[T] => sum(notEqual(o.values, zero).castTo[Long])
        case o: SparseOutput[T] => sum(notEqual(o.values, zero).castTo[Long])
      }
    }
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def argmin[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64, R : TF](
      input: Output[T],
      axes: Output[I],
      outputDataType: DataType[R],
      name: String = "ArgMin"
  ): Output[R] = {
    Op.Builder[(Output[T], Output[I]), Output[R]](
      opType = "ArgMin",
      name = name,
      input = (input, axes)
    ).setAttribute("output_type", outputDataType)
        .build().output
  }

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def argmax[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64, R : TF](
      input: Output[T],
      axes: Output[I],
      outputDataType: DataType[R],
      name: String = "ArgMax"
  ): Output[R] = {
    Op.Builder[(Output[T], Output[I]), Output[R]](
      opType = "ArgMax",
      name = name,
      input = (input, axes)
    ).setAttribute("output_type", outputDataType)
        .build().output
  }

  /** $OpDocMathCumsum
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      Tensor containing the axis along which to perform the cumulative sum.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def cumsum[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Output[T],
      axis: Output[I],
      exclusive: Boolean = false,
      reverse: Boolean = false,
      name: String = "Cumsum"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Cumsum",
      name = name,
      input = (input, axis)
    ).setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .setGradientFn(cumsumGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def cumsumGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val axis = op.input._2
    val exclusive = op.booleanAttribute("exclusive")
    val reverse = op.booleanAttribute("reverse")
    (cumsum(outputGradient, axis, exclusive = exclusive, reverse = !reverse), null)
  }

  /** $OpDocMathCumprod
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      Tensor containing the axis along which to perform the cumulative product.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def cumprod[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Output[T],
      axis: Output[I],
      exclusive: Boolean = false,
      reverse: Boolean = false,
      name: String = "Cumprod"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "Cumprod",
      name = name,
      input = (input, axis)
    ).setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .setGradientFn(cumprodGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def cumprodGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val x = op.input._1
    val axis = op.input._2
    val exclusive = op.booleanAttribute("exclusive")
    val reverse = op.booleanAttribute("reverse")
    // TODO: [GRADIENTS] !!! This fails when x contains 0 and should be fixed.
    val product = cumprod(x, axis, exclusive = exclusive, reverse = reverse)
    val result = cumsum(product * outputGradient, axis, exclusive = exclusive, reverse = !reverse)
    (divide(result, x), null)
  }

  //endregion Reduction Ops

  /** $OpDocMathBinCount
    *
    * @group MathOps
    * @param  input     Tensor containing non-negative values.
    * @param  dataType  If `weights` is `null`, this determines the data type used for the output tensor (i.e., the
    *                   tensor containing the bin counts).
    * @param  weights   If not `null`, this tensor must have the same shape as `input`. For each value in `input`, the
    *                   corresponding bin count will be incremented by the corresponding weight instead of `1`.
    * @param  minLength If not `null`, this ensures the output has length at least `minLength`, padding with zeros at
    *                   the end, if necessary.
    * @param  maxLength If not `null`, this skips values in `input` that are equal or greater than `maxLength`, ensuring
    *                   that the output has length at most `maxLength`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def binCount[T: TF : IsInt32OrInt64OrFloat32OrFloat64](
      input: Output[Int],
      dataType: DataType[T],
      weights: Output[T] = null,
      minLength: Output[Int] = null,
      maxLength: Output[Int] = null,
      name: String = "BinCount"
  ): Output[T] = {
    val inputNonEmpty = greater(prod(Basic.shape(input).castTo[Int]), 0)
    var outputSize = inputNonEmpty.castTo[Int] * (max(input) + 1)
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else {
        Basic.zeros[T](Shape.scalar())
      }
    }
    Op.Builder[(Output[Int], Output[Int], Output[T]), Output[T]](
      opType = "Bincount",
      name = name,
      input = (input, outputSize, effectiveWeights)
    ).build().output
  }

  //region Segment Ops

  /** $OpDocMathSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentSum[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I],
      name: String = "SegmentSum"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "SegmentSum",
      name = name,
      input = (data, segmentIndices)
    ).setGradientFn(segmentSumGradient(_, _)(TF[T], IsNumeric[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def segmentSumGradient[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    (Basic.gather(outputGradient, op.input._2, axis = 0), null)
  }

  /** $OpDocMathSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMean[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I],
      name: String = "SegmentMean"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "SegmentMean",
      name = name,
      input = (data, segmentIndices)
    ).setGradientFn(segmentMeanGradient(_, _)(TF[T], IsNotQuantized[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def segmentMeanGradient[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    val inputRank = Basic.rank(op.input._1)
    val onesShape = Basic.concatenate(Seq(
      Basic.shape(op.input._2).castTo[Int],
      Basic.fill[Int, Int](Basic.expandDims(subtract(inputRank, 1), 0))(
        Basic.ones[Int](Shape()))))
    val ones = Basic.fill[T, Int](onesShape)(Basic.ones[T](Shape()))
    val scaledGradient = divide(outputGradient, segmentSum(ones, op.input._2))
    (Basic.gather(scaledGradient, op.input._2, axis = 0), null)
  }

  /** $OpDocMathSegmentProd
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentProd[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I],
      name: String = "SegmentProd"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "SegmentProd",
      name = name,
      input = (data, segmentIndices)
    ).build().output
  }

  // TODO: [OPS] Missing gradient for 'segmentProd'.

  /** $OpDocMathSegmentMin
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMin[T: TF : IsReal, I: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I],
      name: String = "SegmentMin"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "SegmentMin",
      name = name,
      input = (data, segmentIndices)
    ).setGradientFn(segmentMinOrMaxGradient(_, _)(TF[T], IsReal[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  /** $OpDocMathSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMax[T: TF : IsReal, I: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I],
      name: String = "SegmentMax"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I]), Output[T]](
      opType = "SegmentMax",
      name = name,
      input = (data, segmentIndices)
    ).setGradientFn(segmentMinOrMaxGradient(_, _)(TF[T], IsReal[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def segmentMinOrMaxGradient[T: TF : IsReal, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I]) = {
    // Get the number of selected (minimum or maximum) elements in each segment.
    val gatheredOutputs = Basic.gather(op.output, op.input._2, axis = 0)
    val isSelected = equal(op.input._1, gatheredOutputs)
    val numSelected = segmentSum(isSelected.castTo[T], op.input._2)

    // Compute the gradient for each segment. The gradient for the ith segment is divided evenly among the selected
    // elements in that segment.
    val weightedGradients = divide(outputGradient, numSelected)
    val gatheredGradients = Basic.gather(weightedGradients, op.input._2, axis = 0)
    val zeros = Basic.zerosLike(gatheredGradients)

    (select(isSelected, gatheredGradients, zeros), null)
  }

  protected def gatherDropNegatives[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      parameters: Output[T],
      indices: Output[I],
      zeroClippedIndices: Output[I] = null,
      isPositive: Output[Boolean] = null
  ): (Output[T], Output[I], Output[Boolean]) = {
    val computedZeroClippedIndices = {
      if (zeroClippedIndices != null)
        zeroClippedIndices
      else
        Math.maximum(indices, Basic.zerosLike(indices))
    }
    val gathered = Basic.gather(parameters, zeroClippedIndices, axis = 0)
    val computedIsPositive = {
      if (isPositive != null) {
        isPositive
      } else {
        val zero = Basic.zeros[I](Shape())
        var isPositive = Math.greaterEqual(indices, zero)
        // `select` requires that the condition has the same shape as the other two arguments.
        val minusOne = Basic.constant(-1)
        (0 until (gathered.rank - isPositive.rank)).foreach(_ => {
          isPositive = Basic.expandDims(isPositive, minusOne)
        })
        Math.logicalAnd(isPositive, Basic.onesLike(gathered).castTo[Boolean])
      }
    }
    (Math.select(computedIsPositive, gathered, Basic.zerosLike(gathered)),
        computedZeroClippedIndices,
        computedIsPositive)
  }

  /** $OpDocMathUnsortedSegmentSum
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentSum[T: TF : IsNumeric, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentSum"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "UnsortedSegmentSum",
      name = name,
      input = (data, segmentIndices, segmentsNumber)
    ).setGradientFn(unsortedSegmentSumGradient(_, _)(TF[T], IsNumeric[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
        .build().output
  }

  protected def unsortedSegmentSumGradient[T: TF : IsNumeric, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[I2]) = {
    (gatherDropNegatives(outputGradient, op.input._2)._1, null, null)
  }

  /** Helper function for `unsortedSegmentMean` and `unsortedSegmentSqrtN` that computes the number of segment entries
    * with zero entries set to `1`, in order to allow for division by `N`.
    *
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @return Created op output.
    */
  protected def unsortedSegmentN[T: TF : IsNotQuantized, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentN"
  ): Output[T] = {
    Op.nameScope(name) {
      // `binCount` does not support negative indices and so we use `unsortedSegmentSum`.
      val ones = Basic.ones[T](Basic.shape(segmentIndices))
      val N = unsortedSegmentSum(ones, segmentIndices, segmentsNumber)
      val outputRank = Basic.rank(data) - Basic.rank(segmentIndices)
      val outputRankTiled = Basic.tile(Basic.ones[I2](Shape(1)), outputRank.expandDims(0))
      val broadcastShape = Basic.concatenate(Seq(segmentsNumber.expandDims(0), outputRankTiled))
      val one = Basic.ones[T](Shape())
      maximum(one, Basic.reshape(N, broadcastShape))
    }
  }

  /** $OpDocMathUnsortedSegmentMean
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMean[T: TF : IsNotQuantized, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentMean"
  ): Output[T] = {
    Op.nameScope(name) {
      val n = unsortedSegmentN(data, segmentIndices, segmentsNumber, name = "N")
      unsortedSegmentSum(data, segmentIndices, segmentsNumber, name = "Sum") / n
    }
  }

  // TODO: [OPS] Missing gradient for 'unsortedSegmentMean'.

  /** $OpDocMathUnsortedSegmentProd
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentProd[T: TF : IsNotQuantized, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentProd"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "UnsortedSegmentProd",
      name = name,
      input = (data, segmentIndices, segmentsNumber)
    ).setGradientFn(unsortedSegmentProdGradient(_, _)(TF[T], IsNotQuantized[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
        .build().output
  }

  protected def unsortedSegmentProdGradient[T: TF : IsNotQuantized, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[I2]) = {
    // This gradient can be expressed for each segment by dividing the segment's product by each element of the
    // segment input tensor, but this approach cannot deal with zeros in the input. Unlike `prod` we cannot use the
    // cumulative sum op here, as individual segments may have a different number of elements. Therefore, we consider
    // three cases:
    //
    //   1) A segment input contains no zeros and can safely be divided by the input tensor.
    //   2) A segment contains exactly one zero. In this case, the gradient of each input of the segment is zero,
    //      except for the 0-input. There the gradient is the product of the remaining segment entries.
    //   3) A segment contains at least two zeros. In this case, the gradient is zero for all segment inputs.

    // Note that `unsortedSegmentSum` will filter out the negative indices, and so we do not need to do a `logicalAnd`
    // with `isPositive` here.
    val zero = Basic.zeros[T](Shape())
    val isZero = Math.equal(op.input._1, zero)
    val numZeros = Math.unsortedSegmentSum(isZero.castTo[Int], op.input._2, op.input._3)
    // Handle case 3 and set the gradient to 0 for segments with more than one 0 as input.
    val gradient = Math.select(
      Math.greater(numZeros, 1),
      Basic.zerosLike(outputGradient),
      outputGradient)
    // Replace all zeros with ones and compute the `unsortedSegmentProd`.
    val nonZeroData = Math.select(isZero, Basic.onesLike(op.input._1), op.input._1)
    val nonZeroProd = Math.unsortedSegmentProd(nonZeroData, op.input._2, op.input._3)
    // Clip the indices for the gather to be positive.
    val zeroClippedIndices = Math.maximum(op.input._2, Basic.zerosLike(op.input._2))
    val gatheredProd = Basic.gather(op.output, zeroClippedIndices, axis = 0)
    val gatheredNonZeroProd = Basic.gather(nonZeroProd, zeroClippedIndices, axis = 0)
    // The following may contain NaN/Inf.
    val gatheredProdDivided = gatheredProd / op.input._1
    // Now fetch the individual results for segments containing zero and those that do not. `isZero` will also fetch
    // results for entries with negative indices, but the following `gatherDropNegatives` sets the corresponding entry
    // in the gradient to zero for these.
    val partialDerivative = Math.select(isZero, gatheredNonZeroProd, gatheredProdDivided)
    val gatheredGradient = gatherDropNegatives(gradient, op.input._2, zeroClippedIndices)._1
    (gatheredGradient * partialDerivative, null, null)
  }

  /** $OpDocMathUnsortedSegmentMin
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMin[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentMin"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "UnsortedSegmentMin",
      name = name,
      input = (data, segmentIndices, segmentsNumber)
    ).setGradientFn(unsortedSegmentMinOrMaxGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
        .build().output
  }

  /** $OpDocMathUnsortedSegmentMax
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMax[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentMax"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I1], Output[I2]), Output[T]](
      opType = "UnsortedSegmentMax",
      name = name,
      input = (data, segmentIndices, segmentsNumber)
    ).setGradientFn(unsortedSegmentMinOrMaxGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
        .build().output
  }

  protected def unsortedSegmentMinOrMaxGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[I2]) = {
    // Get the number of selected (minimum or maximum) elements in each segment.
    val (gatheredOutputs, zeroClippedIndices, isPositive) = gatherDropNegatives(op.output, op.input._2)
    val isSelected = Math.logicalAnd(Math.equal(op.input._1, gatheredOutputs), isPositive)
    val numSelected = unsortedSegmentSum(isSelected.castTo[T], op.input._2, op.input._3)
    // Compute the gradient for each segment. The gradient for the ith segment is divided evenly among the selected
    // elements in that segment.
    val weightedGradients = divide(outputGradient, numSelected)
    val (gatheredGradients, _, _) = gatherDropNegatives(weightedGradients, null, zeroClippedIndices, isPositive)
    val zeros = Basic.zerosLike(gatheredGradients)

    (select(isSelected, gatheredGradients, zeros), null, null)
  }

  /** $OpDocMathUnsortedSegmentSqrtN
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentSqrtN[T: TF : IsNotQuantized, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Output[T],
      segmentIndices: Output[I1],
      segmentsNumber: Output[I2],
      name: String = "UnsortedSegmentSqrtN"
  ): Output[T] = {
    Op.nameScope(name) {
      val N = unsortedSegmentN(data, segmentIndices, segmentsNumber, name = "N")
      unsortedSegmentSum(data, segmentIndices, segmentsNumber, name = "Sum") / sqrt(N)
    }
  }

  /** $OpDocMathSparseSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentSum[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Output[T],
      indices: Output[I1],
      segmentIndices: Output[Int],
      numSegments: Output[I2] = null,
      name: String = "SparseSegmentSum"
  ): Output[T] = {
    if (numSegments == null) {
      Op.Builder[(Output[T], Output[I1], Output[Int]), Output[T]](
        opType = "SparseSegmentSum",
        name = name,
        input = (data, indices, segmentIndices)
      ).setGradientFn(sparseSegmentSumGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1]))
          .build().output
    } else {
      Op.Builder[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]](
        opType = "SparseSegmentSumWithNumSegments",
        name = name,
        input = (data, indices, segmentIndices, numSegments)
      ).setGradientFn(sparseSegmentSumWithNumSegmentsGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
          .build().output
    }
  }

  protected def sparseSegmentSumGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = unsortedSegmentSum(
      Basic.gather(outputGradient, op.input._3, axis = 0),
      op.input._2,
      inputRows)
    (gradient, null, null)
  }

  protected def sparseSegmentSumWithNumSegmentsGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int], Output[I2]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = unsortedSegmentSum(
      Basic.gather(outputGradient, op.input._3, axis = 0),
      op.input._2,
      inputRows)
    (gradient, null, null, null)
  }

  /** $OpDocMathSparseSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentMean[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Output[T],
      indices: Output[I1],
      segmentIndices: Output[Int],
      numSegments: Output[I2] = null,
      name: String = "SparseSegmentMean"
  ): Output[T] = {
    if (numSegments == null) {
      Op.Builder[(Output[T], Output[I1], Output[Int]), Output[T]](
        opType = "SparseSegmentMean",
        name = name,
        input = (data, indices, segmentIndices)
      ).setGradientFn(sparseSegmentMeanGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1]))
          .build().output
    } else {
      Op.Builder[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]](
        opType = "SparseSegmentMeanWithNumSegments",
        name = name,
        input = (data, indices, segmentIndices, numSegments)
      ).setGradientFn(sparseSegmentMeanWithNumSegmentsGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
          .build().output
    }
  }

  protected def sparseSegmentMeanGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = Op.Builder[(Output[T], Output[I1], Output[Int], Output[Long]), Output[T]](
      opType = "SparseSegmentMeanGrad",
      name = "SparseSegmentMeanGrad",
      input = (outputGradient, op.input._2, op.input._3, inputRows)
    ).build().output
    (gradient, null, null)
  }

  protected def sparseSegmentMeanWithNumSegmentsGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int], Output[I2]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = Op.Builder[(Output[T], Output[I1], Output[Int], Output[Long]), Output[T]](
      opType = "SparseSegmentMeanGrad",
      name = "SparseSegmentMeanGrad",
      input = (outputGradient, op.input._2, op.input._3, inputRows)
    ).build().output
    (gradient, null, null, null)
  }

  /** $OpDocMathSparseSegmentSumSqrtN
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentSumSqrtN[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Output[T],
      indices: Output[I1],
      segmentIndices: Output[Int],
      numSegments: Output[I2] = null,
      name: String = "SparseSegmentSumSqrtN"
  ): Output[T] = {
    if (numSegments == null) {
      Op.Builder[(Output[T], Output[I1], Output[Int]), Output[T]](
        opType = "SparseSegmentSqrtN",
        name = name,
        input = (data, indices, segmentIndices)
      ).setGradientFn(sparseSegmentSumSqrtNGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1]))
          .build().output
    } else {
      Op.Builder[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]](
        opType = "SparseSegmentSqrtNWithNumSegments",
        name = name,
        input = (data, indices, segmentIndices, numSegments)
      ).setGradientFn(sparseSegmentSumSqrtNWithNumSegmentsGradient(_, _)(TF[T], IsReal[T], TF[I1], IsInt32OrInt64[I1], TF[I2], IsInt32OrInt64[I2]))
          .build().output
    }
  }

  protected def sparseSegmentSumSqrtNGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = Op.Builder[(Output[T], Output[I1], Output[Int], Output[Long]), Output[T]](
      opType = "SparseSegmentSqrtNGrad",
      name = "SparseSegmentSumSqrtNGrad",
      input = (outputGradient, op.input._2, op.input._3, inputRows)
    ).build().output
    (gradient, null, null)
  }

  protected def sparseSegmentSumSqrtNWithNumSegmentsGradient[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I1], Output[Int], Output[I2]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I1], Output[Int], Output[I2]) = {
    val inputRows = Basic.shape(op.input._1).slice(0)
    val gradient = Op.Builder[(Output[T], Output[I1], Output[Int], Output[Long]), Output[T]](
      opType = "SparseSegmentSqrtNGrad",
      name = "SparseSegmentSumSqrtNGrad",
      input = (outputGradient, op.input._2, op.input._3, inputRows)
    ).build().output
    (gradient, null, null, null)
  }

  //endregion Segment Ops

  //region Matrix Ops

  /** $OpDocMathDiag
    *
    * @group MathOps
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def diag[T: TF : IsNotQuantized](
      diagonal: Output[T],
      name: String = "Diag"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "Diag",
      name = name,
      input = diagonal
    ).setGradientFn(diagGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def diagGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    diagPart(outputGradient)
  }

  /** $OpDocMathDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def diagPart[T: TF : IsNotQuantized](
      input: Output[T],
      name: String = "DiagPart"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "DiagPart",
      name = name,
      input = input
    ).setGradientFn(diagPartGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def diagPartGradient[T: TF : IsNotQuantized](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    diag(outputGradient)
  }

  /** $OpDocMathMatrixDiag
    *
    * @group MathOps
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its last
    *         dimension duplicated.
    */
  def matrixDiag[T : TF](
      diagonal: Output[T],
      name: String = "MatrixDiag"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "MatrixDiag",
      name = name,
      input = diagonal
    ).setGradientFn(matrixDiagGradient(_, _)(TF[T]))
        .build().output
  }

  protected def matrixDiagGradient[T : TF](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    matrixDiagPart(outputGradient)
  }

  /** $OpDocMathMatrixSetDiag
    *
    * @group MathOps
    * @param  input    Rank-`K+1` tensor, where `K >= 2`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `input`.
    */
  def matrixSetDiag[T : TF](
      input: Output[T],
      diagonal: Output[T],
      name: String = "MatrixSetDiag"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "MatrixSetDiag",
      name = name,
      input = (input, diagonal)
    ).setGradientFn(matrixSetDiagGradient(_, _)(TF[T]))
        .build().output
  }

  protected def matrixSetDiagGradient[T : TF](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val inputShape = op.input._1.shape.mergeWith(outputGradient.shape)
    val batchShape = inputShape(0 :: -2).mergeWith(op.input._1.shape(0 :: -1))
    val matrixShape = inputShape(-2 ::)
    val diagShape = {
      if (batchShape.isFullyDefined && matrixShape.isFullyDefined) {
        Basic.constant(tensors.ops.Basic.stack((batchShape.asArray :+ matrixShape.asArray.min).map(Tensor(_))))
      } else {
        Op.colocateWith(Set(outputGradient.op), ignoreExisting = true) {
          val gradShape = Basic.shape(outputGradient).castTo[Int]
          val gradRank = Basic.rank(outputGradient)
          val batchShape = Basic.slice(gradShape, 0, gradRank - 2)
          val matrixShape = Basic.slice(gradShape, gradRank - 2, 2)
          val minDim = min(matrixShape)
          Basic.concatenate(Seq(batchShape, minDim), 0)
        }
      }
    }
    val gradInput = matrixSetDiag(
      outputGradient, Basic.fill[T, Int](diagShape)(Tensor.zeros[T](Shape())))
    val gradDiag = matrixDiagPart(outputGradient)
    (gradInput, gradDiag)
  }

  /** $OpDocMathMatrixDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` tensor, where `K >= 2`.
    * @param  name  Name for the created op.
    * @return Created op output containing the diagonal(s) and having shape equal to
    *         `input.shape[::-2] + [min(input.shape[-2::])]`.
    */
  def matrixDiagPart[T: TF](
      input: Output[T],
      name: String = "MatrixDiagPart"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "MatrixDiagPart",
      name = name,
      input = input
    ).setGradientFn(matrixDiagPartGradient(_, _)(TF[T]))
        .build().output
  }

  protected def matrixDiagPartGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val matrixShape = op.input.shape(-2 ::)
    if (matrixShape.isFullyDefined && matrixShape(0) == matrixShape(1))
      matrixDiag(outputGradient)
    else
      matrixSetDiag(Basic.zerosLike(op.input), outputGradient)
  }

  /** $OpDocMathMatrixBandPart
    *
    * @group MathOps
    * @param  input             Input tensor.
    * @param  numSubDiagonals   Scalar tensor that contains the number of sub-diagonals to keep. If negative,
    *                           the entire lower triangle is kept.
    * @param  numSuperDiagonals Scalar tensor that contains the number of super-diagonals to keep. If negative,
    *                           the entire upper triangle is kept.
    * @param  name              Name for the created op.
    * @return Created op output.
    */
  def matrixBandPart[T: TF, I: TF : IsInt32OrInt64](
      input: Output[T],
      numSubDiagonals: Output[I],
      numSuperDiagonals: Output[I],
      name: String = "MatrixBandPart"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[I], Output[I]), Output[T]](
      opType = "MatrixBandPart",
      name = name,
      input = (input, numSubDiagonals, numSuperDiagonals)
    ).setGradientFn(matrixBandPartGradient(_, _)(TF[T], TF[I], IsInt32OrInt64[I]))
        .build().output
  }

  protected def matrixBandPartGradient[T: TF, I: TF : IsInt32OrInt64](
      op: Op[(Output[T], Output[I], Output[I]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[I], Output[I]) = {
    (matrixBandPart(outputGradient, op.input._2, op.input._3), null, null)
  }

  /** $OpDocMathTrace
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def trace[T: TF : IsNumeric](
      input: Output[T],
      name: String = "Trace"
  ): Output[T] = {
    Op.nameScope(name) {
      sum(matrixDiagPart(input), axes = -1)
    }
  }

  /** $OpDocMathScalarMul
    *
    * @group MathOps
    * @param  scalar Scalar tensor.
    * @param  tensor Tensor to multiply the scalar tensor with.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def scalarMul[T: TF : IsNotQuantized, OL[A] <: OutputLike[A]](
      scalar: Output[T],
      tensor: OL[T],
      name: String = "ScalarMul"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    Op.nameScope(name) {
      ev.applyUnary(tensor, o => multiply(scalar, o))
    }
  }

  // TODO: [OPS] The following type constraints are wrong.

  /** $OpDocMathMatmul
    *
    * @group MathOps
    * @param  a          First input tensor.
    * @param  b          Second input tensor.
    * @param  transposeA If `true`, `a` is transposed before the multiplication.
    * @param  transposeB If `true`, `b` is transposed before the multiplication.
    * @param  conjugateA If `true`, `a` is conjugated before the multiplication.
    * @param  conjugateB If `true`, `b` is conjugated before the multiplication.
    * @param  aIsSparse  If `true`, `a` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @param  bIsSparse  If `true`, `b` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @param  name       Name for the created op.
    * @return Created op output that has the same data type as `a` and `b` and where each inner-most matrix is the
    *         product of the corresponding matrices in `a` and `b`.
    */
  def matmul[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      transposeA: Boolean = false,
      transposeB: Boolean = false,
      conjugateA: Boolean = false,
      conjugateB: Boolean = false,
      aIsSparse: Boolean = false,
      bIsSparse: Boolean = false,
      name: String = "MatMul"
  ): Output[T] = {
    val sparseMatMulDataTypes = Set[DataType[Any]](BFLOAT16, FLOAT32)
    if (!aIsSparse && !bIsSparse && (a.rank == -1 || a.rank > 2) && (b.rank == -1 || b.rank > 2)) {
      // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
      // The "conj" op is a no-op for real matrices.
      val (x, adjointX) = transposeConjugateToAdjoint(a, transposeA, conjugateA)
      val (y, adjointY) = transposeConjugateToAdjoint(b, transposeB, conjugateB)
      Op.Builder[(Output[T], Output[T]), Output[T]](
        opType = "BatchMatMul",
        name = name,
        input = (x, y)
      ).setAttribute("adj_x", adjointX)
          .setAttribute("adj_y", adjointY)
          .setGradientFn(batchMatmulGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output
    } else if ((aIsSparse || bIsSparse) &&
        sparseMatMulDataTypes.contains(a.dataType) &&
        sparseMatMulDataTypes.contains(b.dataType)) {
      val (x, transposeX) = transposeConjugateToTranspose(a, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(b, transposeB, conjugateB)
      Op.Builder[(Output[T], Output[T]), Output[T]](
        opType = "SparseMatMul",
        name = name,
        input = (x, y)
      ).setAttribute("transpose_a", transposeX)
          .setAttribute("transpose_b", transposeY)
          .setAttribute("a_is_sparse", aIsSparse)
          .setAttribute("b_is_sparse", bIsSparse)
          .setGradientFn(sparseMatmulGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output
    } else {
      val (x, transposeX) = transposeConjugateToTranspose(a, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(b, transposeB, conjugateB)
      Op.Builder[(Output[T], Output[T]), Output[T]](
        opType = "MatMul",
        name = name,
        input = (x, y)
      ).setAttribute("transpose_a", transposeX)
          .setAttribute("transpose_b", transposeY)
          .setGradientFn(matmulGradient(_, _)(TF[T], IsNotQuantized[T]))
          .build().output
    }
  }

  protected def transposeConjugateToAdjoint[T: TF : IsNotQuantized](
      tensor: Output[T],
      transpose: Boolean,
      conj: Boolean
  ): (Output[T], Boolean) = {
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) => (conjugate(tensor), false)
      case (true, false) => (conjugate(tensor), true)
      case (true, true) => (tensor, true)
    }
  }

  protected def transposeConjugateToTranspose[T: TF : IsNotQuantized](
      tensor: Output[T],
      transpose: Boolean,
      conj: Boolean
  ): (Output[T], Boolean) = {
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) => (conjugate(tensor), false)
      case (true, false) => (tensor, true)
      case (true, true) => (conjugate(tensor), true)
    }
  }

  protected def batchMatmulGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val x = op.input._1
    val y = op.input._2
    val adjointX = op.booleanAttribute("adj_x")
    val adjointY = op.booleanAttribute("adj_y")
    (adjointX, adjointY) match {
      case (false, false) =>
        (matmul(outputGradient, y, transposeA = false, transposeB = true, conjugateA = false, conjugateB = true),
            matmul(x, outputGradient, transposeA = true, transposeB = false, conjugateA = true, conjugateB = false))
      case (false, true) =>
        (matmul(outputGradient, y, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false),
            matmul(outputGradient, x, transposeA = true, transposeB = false, conjugateA = true, conjugateB = false))
      case (true, false) =>
        (matmul(y, outputGradient, transposeA = false, transposeB = true, conjugateA = false, conjugateB = true),
            matmul(x, outputGradient, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false))
      case (true, true) =>
        (matmul(y, outputGradient, transposeA = true, transposeB = true, conjugateA = true, conjugateB = true),
            matmul(outputGradient, x, transposeA = true, transposeB = true, conjugateA = true, conjugateB = true))
    }
  }

  protected def matmulGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val a = op.input._1
    val b = op.input._2
    val transposeA = op.booleanAttribute("transpose_a")
    val transposeB = op.booleanAttribute("transpose_b")
    (transposeA, transposeB) match {
      case (false, false) =>
        (matmul(outputGradient, b, transposeA = false, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(a, outputGradient, transposeA = true, transposeB = false, conjugateA = false, conjugateB = false))
      case (false, true) =>
        (matmul(outputGradient, b, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false),
            matmul(outputGradient, a, transposeA = true, transposeB = false, conjugateA = false, conjugateB = false))
      case (true, false) =>
        (matmul(b, outputGradient, transposeA = false, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(a, outputGradient, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false))
      case (true, true) =>
        (matmul(b, outputGradient, transposeA = true, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(outputGradient, a, transposeA = true, transposeB = true, conjugateA = false, conjugateB = false))
    }
  }

  protected def sparseMatmulGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val a = op.input._1
    val b = op.input._2
    val transposeA = op.booleanAttribute("transpose_a")
    val transposeB = op.booleanAttribute("transpose_b")
    val aIsSparse = op.booleanAttribute("a_is_sparse")
    val bIsSparse = op.booleanAttribute("b_is_sparse")
    // Use heuristic to figure out if the gradient may be sparse.
    val gradIsSparse = outputGradient.op.opType == "ReluGrad"

    def helper(
        a: Output[T],
        b: Output[T],
        tA: Boolean = false,
        tB: Boolean = false,
        sA: Boolean = false,
        sB: Boolean = false
    ): Output[T] = {
      matmul(
        a = a,
        b = if (tB) Basic.transpose(b) else b,
        transposeA = tA,
        transposeB = false,
        conjugateA = false,
        conjugateB = false,
        aIsSparse = sA,
        bIsSparse = sB)
    }

    (transposeA, transposeB) match {
      case (false, false) =>
        (helper(outputGradient, b, tA = false, tB = true, sA = gradIsSparse, sB = bIsSparse),
            helper(a, outputGradient, tA = true, tB = false, sA = aIsSparse, sB = gradIsSparse))
      case (false, true) =>
        (helper(outputGradient, b, tA = false, tB = false, sA = gradIsSparse, sB = bIsSparse),
            helper(outputGradient, a, tA = true, tB = false, sA = gradIsSparse, sB = aIsSparse))
      case (true, false) =>
        (helper(b, outputGradient, tA = false, tB = true, sA = bIsSparse, sB = gradIsSparse),
            helper(a, outputGradient, tA = false, tB = false, sA = aIsSparse, sB = gradIsSparse))
      case (true, true) =>
        (helper(b, outputGradient, tA = true, tB = true, sA = bIsSparse, sB = gradIsSparse),
            helper(outputGradient, a, tA = true, tB = true, sA = gradIsSparse, sB = aIsSparse))
    }
  }

  /** $OpDocMathCross
    *
    * @group MathOps
    * @param  a    First input tensor.
    * @param  b    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cross[T: TF : IsReal](
      a: Output[T],
      b: Output[T],
      name: String = "Cross"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Cross",
      name = name,
      input = (a, b)
    ).setGradientFn(crossGradient(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def crossGradient[T: TF : IsReal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val u = op.input._1
    val v = op.input._2
    (cross(v, outputGradient), cross(outputGradient, u))
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @return Created op output.
    * @throws InvalidArgumentException If the `numAxes < 1` or the rank of `a` is unknown.
    */
  @throws[InvalidArgumentException]
  def tensorDot[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      numAxes: Int
  ): Output[T] = {
    tensorDot(a, b, numAxes, "TensorDot")
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @param  name    Name for the created ops.
    * @return Created op output.
    * @throws InvalidArgumentException If the `numAxes < 1` or the rank of `a` is unknown.
    */
  @throws[InvalidArgumentException]
  def tensorDot[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      numAxes: Int,
      name: String
  ): Output[T] = {
    if (numAxes < 1)
      throw InvalidArgumentException("'numAxes' must be at least 1.")
    if (a.rank == -1)
      throw InvalidArgumentException(
        "Cannot use 'tensorDot' with an unknown input tensor shape. Use 'tensorDotDynamic' instead.")
    tensorDot(a, b, a.rank - numAxes until a.rank, 0 until numAxes, name)
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @return Created op output.
    * @throws InvalidArgumentException If the size of `axesA` does not match the size of `axesB`.
    */
  @throws[InvalidArgumentException]
  def tensorDot[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      axesA: Seq[Int],
      axesB: Seq[Int]
  ): Output[T] = {
    tensorDot(a, b, axesA, axesB, "TensorDot")
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @param  name  Name for the created ops.
    * @return Created op output.
    * @throws InvalidArgumentException If the size of `axesA` does not match the size of `axesB`.
    */
  @throws[InvalidArgumentException]
  def tensorDot[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      axesA: Seq[Int],
      axesB: Seq[Int],
      name: String
  ): Output[T] = {
    if (axesA.lengthCompare(axesB.size) != 0)
      throw InvalidArgumentException(
        s"Different number of contraction axes for 'a' and 'b', ${axesA.size} != ${axesB.size}.")

    /** Helper method to perform transpose and reshape for the tensor contraction op. This method is helpful in reducing
      * `tensorDot` to `matmul` using the `transpose` and the `reshape` ops. The method takes a tensor and performs the
      * correct transpose and reshape operations for the provided indices. It returns the reshaped tensor as well as a
      * list of indices necessary to reshape the tensor back to its proper shape after the matrix multiplication.
      *
      * @param  a       Tensor being reshaped.
      * @param  axes    Sequence of unique indices of axes of `a`.
      * @param  flipped If `true`, the method assumes that `a` is the second argument in the contraction operation.
      * @return Tuple that contains: (i) the reshaped tensor `a` that allows contraction via `matmul`, (ii) a
      *         tensor that contains the shape of the free axes, and (iii) a sequence of integers representing the
      *         inferred static shape of the free axes.
      */
    def tensorDotReshape(
        a: Output[T],
        axes: Seq[Int],
        flipped: Boolean = false
    ): (Output[T], Output[Int], Seq[Int]) = {
      if (a.shape.isFullyDefined) {
        val mappedAxes = axes.map(i => if (i >= 0) i else i + a.rank)
        val prodAxes = mappedAxes.map(a.shape(_)).product
        val free = (0 until a.rank).filter(!mappedAxes.contains(_))
        val freeAxes = free.map(a.shape(_))
        val prodFree = freeAxes.product
        val permutation = if (flipped) mappedAxes ++ free else free ++ mappedAxes
        val newShape = if (flipped) Shape(prodAxes, prodFree) else Shape(prodFree, prodAxes)
        val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
        val freeAxesOutput = if (freeAxes.isEmpty) Basic.constant(Tensor.empty[Int]) else Basic.constant(freeAxes)
        (reshapedA, freeAxesOutput, freeAxes)
      } else {
        val (mappedAxes, freeAxesStatic) = {
          if (a.rank != -1) {
            val mappedAxes = axes.map(i => if (i >= 0) i else i + a.rank)
            val free = (0 until a.rank).filter(!mappedAxes.contains(_))
            val freeAxes = free.map(a.shape(_))
            (mappedAxes, freeAxes)
          } else {
            (axes, null)
          }
        }
        val shapeA = Basic.shape(a).castTo[Int]
        val rankA = Basic.rank(a)
        var axesO = Basic.constant(mappedAxes, name = "Axes")
        axesO = ((axesO >= 0).castTo[Int] * axesO) + ((axesO < 0).castTo[Int] * (axesO + rankA))
        val (free, _) = Basic.listDiff(Math.range(0, rankA), axesO, indicesDataType = Int)
        val freeAxes = Basic.gather(shapeA, free, axis = 0)
        val axesAxes = Basic.gather(shapeA, axesO, axis = 0)
        val prodFree = freeAxes.prod()
        val prodAxes = axesAxes.prod()
        val (permutation, newShape) = {
          if (flipped) {
            val permutation = Basic.concatenate(Seq(axesO, free), 0)
            val newShape = Basic.stack(Seq(prodAxes, prodFree))
            (permutation, newShape)
          } else {
            val permutation = Basic.concatenate(Seq(free, axesO), 0)
            val newShape = Basic.stack(Seq(prodFree, prodAxes))
            (permutation, newShape)
          }
        }
        val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
        (reshapedA, freeAxes, freeAxesStatic)
      }
    }

    Op.nameScope(name) {
      val (reshapedA, freeA, freeAStatic) = tensorDotReshape(a, axesA)
      val (reshapedB, freeB, freeBStatic) = tensorDotReshape(b, axesB, flipped = true)
      val abMatmul = matmul(reshapedA, reshapedB)
      val reshaped = Basic.reshape(abMatmul, Basic.concatenate(Seq(freeA, freeB), 0))
      if (freeAStatic != null && freeBStatic != null)
        reshaped.setShape(Shape.fromSeq(freeAStatic ++ freeBStatic))
      reshaped
    }
  }

  /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @return Created op output.
    * @throws InvalidArgumentException If `numAxes` is not a scalar.
    */
  @throws[InvalidArgumentException]
  def tensorDotDynamic[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      numAxes: Output[Int]
  ): Output[T] = {
    tensorDotDynamic(a, b, numAxes, "TensorDot")
  }

  /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @param  name    Name for the created ops.
    * @return Created op output.
    * @throws InvalidArgumentException If `numAxes` is not a scalar.
    */
  @throws[InvalidArgumentException]
  def tensorDotDynamic[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      numAxes: Output[Int],
      name: String
  ): Output[T] = {
    if (numAxes.rank != 0)
      throw InvalidArgumentException("'numAxes' must be a scalar.")
    val rankA = Basic.rank(a)
    tensorDotDynamic(a, b, range(rankA - numAxes, rankA), range(0, numAxes), name)
  }

  /** Dynamic version (i.e., where `axesA` and `axesB` may be symbolic tensors) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @return Created op output.
    * @throws InvalidArgumentException If the rank `axesA` or `axesB` if larger than `1`.
    */
  @throws[InvalidArgumentException]
  def tensorDotDynamic[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      axesA: Output[Int],
      axesB: Output[Int]
  ): Output[T] = {
    tensorDotDynamic(a, b, axesA, axesB, "TensorDot")
  }

  /** Dynamic version (i.e., where `axesA` and `axesB` may be symbolic tensors) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @param  name  Name for the created ops.
    * @return Created op output.
    * @throws InvalidArgumentException If the rank `axesA` or `axesB` if larger than `1`.
    */
  @throws[InvalidArgumentException]
  def tensorDotDynamic[T: TF : IsNotQuantized](
      a: Output[T],
      b: Output[T],
      axesA: Output[Int],
      axesB: Output[Int],
      name: String = "TensorDot"
  ): Output[T] = {
    if (axesA.rank != 1)
      throw InvalidArgumentException("'axesA' must be a vector.")
    if (axesB.rank != 1)
      throw InvalidArgumentException("'axesB' must be a vector.")

    /** Helper method to perform transpose and reshape for the tensor contraction op. This method is helpful in reducing
      * `tensorDot` to `matmul` using the `transpose` and the `reshape` ops. The method takes a tensor and performs the
      * correct transpose and reshape operations for the provided indices. It returns the reshaped tensor as well as a
      * list of indices necessary to reshape the tensor back to its proper shape after the matrix multiplication.
      *
      * @param  a       Tensor being reshaped.
      * @param  axes    Sequence of unique indices of axes of `a`.
      * @param  flipped If `true`, the method assumes that `a` is the second argument in the contraction operation.
      * @return Tuple that contains: (i) the reshaped tensor `a` that allows contraction via `matmul`, and (ii) a
      *         tensor that contains the shape of the free axes.
      */
    def tensorDotReshape(
        a: Output[T],
        axes: Output[Int],
        flipped: Boolean = false
    ): (Output[T], Output[Int]) = {
      val shapeA = Basic.shape(a).castTo[Int]
      val rankA = Basic.rank(a)
      val mappedAxes = ((axes >= 0).castTo[Int] * axes) + ((axes < 0).castTo[Int] * (axes + rankA))
      val (free, _) = Basic.listDiff(Math.range(0, rankA), mappedAxes, indicesDataType = Int)
      val freeAxes = Basic.gather(shapeA, free, axis = 0)
      val axesAxes = Basic.gather(shapeA, mappedAxes, axis = 0)
      val prodFree = freeAxes.prod()
      val prodAxes = axesAxes.prod()
      val (permutation, newShape) = {
        if (flipped) {
          val permutation = Basic.concatenate(Seq(mappedAxes, free), 0)
          val newShape = Basic.stack(Seq(prodAxes, prodFree))
          (permutation, newShape)
        } else {
          val permutation = Basic.concatenate(Seq(free, mappedAxes), 0)
          val newShape = Basic.stack(Seq(prodFree, prodAxes))
          (permutation, newShape)
        }
      }
      val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
      (reshapedA, freeAxes)
    }

    Op.nameScope(name) {
      val (reshapedA, freeA) = tensorDotReshape(a, axesA)
      val (reshapedB, freeB) = tensorDotReshape(b, axesB, flipped = true)
      val abMatmul = matmul(reshapedA, reshapedB)
      Basic.reshape(abMatmul, Basic.concatenate(Seq(freeA, freeB), 0))
    }
  }

  //endregion Matrix Ops

  //region Complex Ops

  /** $OpDocMathComplex
    *
    * @group MathOps
    * @param  real Tensor containing the real component.
    * @param  imag Tensor containing the imaginary component.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def complexFloat(
      real: Output[Float],
      imag: Output[Float],
      name: String = "Complex"
  ): Output[ComplexFloat] = {
    Op.Builder[(Output[Float], Output[Float]), Output[ComplexFloat]](
      opType = "Complex",
      name = name,
      input = (real, imag)
    ).setAttribute("Tout", COMPLEX64)
        .setGradientFn(complexFloatGradient)
        .build().output
  }

  def complexFloatGradient(
      op: Op[(Output[Float], Output[Float]), Output[ComplexFloat]],
      outputGradient: Output[ComplexFloat]
  ): (Output[Float], Output[Float]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    (Basic.reshape(sum(realFloat(outputGradient), rx), xShape),
        Basic.reshape(sum(imagFloat(outputGradient), ry), yShape))
  }

  /** $OpDocMathComplex
    *
    * @group MathOps
    * @param  real Tensor containing the real component.
    * @param  imag Tensor containing the imaginary component.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def complexDouble(
      real: Output[Double],
      imag: Output[Double],
      name: String = "Complex"
  ): Output[ComplexDouble] = {
    Op.Builder[(Output[Double], Output[Double]), Output[ComplexDouble]](
      opType = "Complex",
      name = name,
      input = (real, imag)
    ).setAttribute("Tout", COMPLEX128)
        .setGradientFn(complexDoubleGradient)
        .build().output
  }

  def complexDoubleGradient(
      op: Op[(Output[Double], Output[Double]), Output[ComplexDouble]],
      outputGradient: Output[ComplexDouble]
  ): (Output[Double], Output[Double]) = {
    val x = op.input._1
    val y = op.input._2
    val xShape = Basic.shape(x)
    val yShape = Basic.shape(y)
    val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
    (Basic.reshape(sum(realDouble(outputGradient), rx), xShape),
        Basic.reshape(sum(imagDouble(outputGradient), ry), yShape))
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def realFloat[OL[A] <: OutputLike[A]](
      input: OL[ComplexFloat],
      name: String = "Real"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexFloat]
  ): OL[Float] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexFloat], Output[Float]](
          opType = "Real",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT32)
            .setGradientFn(realFloatGradient)
            .build().output
      })
    }
  }

  protected def realFloatGradient(
      op: Op[Output[ComplexFloat], Output[Float]],
      outputGradient: Output[Float]
  ): Output[ComplexFloat] = {
    complexFloat(outputGradient, Basic.zeros[Float](Shape()))
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def realDouble[OL[A] <: OutputLike[A]](
      input: OL[ComplexDouble],
      name: String = "Real"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexDouble]
  ): OL[Double] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexDouble], Output[Double]](
          opType = "Real",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT64)
            .setGradientFn(realDoubleGradient)
            .build().output
      })
    }
  }

  protected def realDoubleGradient(
      op: Op[Output[ComplexDouble], Output[Double]],
      outputGradient: Output[Double]
  ): Output[ComplexDouble] = {
    complexDouble(outputGradient, Basic.zeros[Double](Shape()))
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def imagFloat[OL[A] <: OutputLike[A]](
      input: OL[ComplexFloat],
      name: String = "Imag"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexFloat]
  ): OL[Float] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexFloat], Output[Float]](
          opType = "Imag",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT32)
            .setGradientFn(imagFloatGradient)
            .build().output
      })
    }
  }

  protected def imagFloatGradient(
      op: Op[Output[ComplexFloat], Output[Float]],
      outputGradient: Output[Float]
  ): Output[ComplexFloat] = {
    complexFloat(Basic.zeros[Float](Shape()), outputGradient)
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def imagDouble[OL[A] <: OutputLike[A]](
      input: OL[ComplexDouble],
      name: String = "Imag"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexDouble]
  ): OL[Double] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexDouble], Output[Double]](
          opType = "Imag",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT64)
            .setGradientFn(imagDoubleGradient)
            .build().output
      })
    }
  }

  protected def imagDoubleGradient(
      op: Op[Output[ComplexDouble], Output[Double]],
      outputGradient: Output[Double]
  ): Output[ComplexDouble] = {
    complexDouble(Basic.zeros[Double](Shape()), outputGradient)
  }

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def magnitudeFloat[OL[A] <: OutputLike[A]](
      x: OL[ComplexFloat],
      name: String = "Magnitude"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexFloat]
  ): OL[Float] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[ComplexFloat], Output[Float]](
        opType = "ComplexAbs",
        name = name,
        input = o
      ).setAttribute("Tout", FLOAT32)
          .setGradientFn(magnitudeFloatGradient)
          .build().output)
  }

  protected def magnitudeFloatGradient(
      op: Op[Output[ComplexFloat], Output[Float]],
      outputGradient: Output[Float]
  ): Output[ComplexFloat] = {
    multiply(complexFloat(outputGradient, Basic.zerosLike(outputGradient)), sign(op.input))
  }

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def magnitudeDouble[OL[A] <: OutputLike[A]](
      x: OL[ComplexDouble],
      name: String = "Magnitude"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexDouble]
  ): OL[Double] = {
    ev.applyUnary(x, o =>
      Op.Builder[Output[ComplexDouble], Output[Double]](
        opType = "ComplexAbs",
        name = name,
        input = o
      ).setAttribute("Tout", FLOAT64)
          .setGradientFn(magnitudeDoubleGradient)
          .build().output)
  }

  protected def magnitudeDoubleGradient(
      op: Op[Output[ComplexDouble], Output[Double]],
      outputGradient: Output[Double]
  ): Output[ComplexDouble] = {
    multiply(complexDouble(outputGradient, Basic.zerosLike(outputGradient)), sign(op.input))
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def angleFloat[OL[A] <: OutputLike[A]](
      input: OL[ComplexFloat],
      name: String = "Angle"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexFloat]
  ): OL[Float] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexFloat], Output[Float]](
          opType = "Angle",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT32)
            .build().output
      })
    }
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def angleDouble[OL[A] <: OutputLike[A]](
      input: OL[ComplexDouble],
      name: String = "Angle"
  )(implicit
      ev: OutputOps.Aux[OL, ComplexDouble]
  ): OL[Double] = {
    Op.nameScope(s"${input.name}/") {
      ev.applyUnary(input, o => {
        Op.Builder[Output[ComplexDouble], Output[Double]](
          opType = "Angle",
          name = name,
          input = o
        ).setAttribute("Tout", FLOAT64)
            .build().output
      })
    }
  }

  /** $OpDocMathConjugate
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def conjugate[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "Conjugate"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      if (o.dataType.isComplex) {
        Op.Builder[Output[T], Output[T]](
          opType = "Conj",
          name = name,
          input = o
        ).setGradientFn(conjugateGradient(_, _)(TF[T]))
            .build().output
      } else {
        o
      }
    })
  }

  protected def conjugateGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    conjugate(outputGradient)
  }

  //endregion Complex Ops

  //region Quantization Ops

  // TODO: [OPS] quantization

  //endregion Quantization Ops

  //region Bucketization Ops

  /** $OpDocMathBucketize
    *
    * @group MathOps
    * @param  input      Numeric tensor to bucketize.
    * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def bucketize[T: TF : IsInt32OrInt64OrFloat32OrFloat64](
      input: Output[T],
      boundaries: Seq[Float],
      name: String = "Bucketize"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "Bucketize",
      name = name,
      input = input
    ).setAttribute("boundaries", boundaries.toArray)
        .build().output
  }

  //endregion Bucketization Ops

  //region Other Ops

  /** $OpDocMathZerosFraction
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def zerosFraction[T: TF : IsNumeric](
      input: Output[T],
      name: String = "ZerosFraction"
  ): Output[Float] = {
    Op.nameScope(name) {
      val zero = Basic.zeros[T](Shape())
      mean(equal(input, zero).castTo[Float])
    }
  }

  // TODO: bessel_i0, bessel_i1, bessel_i0e, bessel_i1e.

  //endregion Other Ops
}

object Math extends Math {
  private[ops] trait Implicits {
    implicit def outputConvertibleToMathOps[T: TF, OC](
        value: OC
    )(implicit f: OC => Output[T]): MathOps[T] = {
      new MathOps(f(value))
    }

    implicit def outputConvertibleToFloatMathOps[OC](
        value: OC
    )(implicit f: OC => Output[Float]): FloatMathOps = {
      new FloatMathOps(f(value))
    }

    implicit def outputConvertibleToDoubleMathOps[OC](
        value: OC
    )(implicit f: OC => Output[Double]): DoubleMathOps = {
      new DoubleMathOps(f(value))
    }

    implicit def outputConvertibleToComplexFloatMathOps[OC](
        value: OC
    )(implicit f: OC => Output[ComplexFloat]): ComplexFloatMathOps = {
      new ComplexFloatMathOps(f(value))
    }

    implicit def outputConvertibleToComplexDoubleMathOps[OC](
        value: OC
    )(implicit f: OC => Output[ComplexDouble]): ComplexDoubleMathOps = {
      new ComplexDoubleMathOps(f(value))
    }

    implicit class MathOps[T](val output: Output[T]) {
      protected implicit val evTTF: TF[T] = {
        TF.fromDataType(output.dataType)
      }

      /** $OpDocMathSelect
        *
        * @group MathOps
        * @param  x Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may have
        *           a higher rank, but its first dimension must match the size of `condition`.
        * @param  y Tensor with the same data type and shape as `t`.
        * @return Created op output.
        */
      def select[R: TF](
          x: Output[R],
          y: Output[R]
      )(implicit ev: T =:= Boolean): Output[R] = {
        Math.select(output.asInstanceOf[Output[Boolean]], x, y)
      }

      //region Operators

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_!(implicit ev: T =:= Boolean): Output[Boolean] = {
        logicalNot
      }

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def &&(other: Output[Boolean])(implicit ev: T =:= Boolean): Output[Boolean] = {
        logicalAnd(other)
      }

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ||(other: Output[Boolean])(implicit ev: T =:= Boolean): Output[Boolean] = {
        logicalOr(other)
      }

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ===(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        equal(other)
      }

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def =!=(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        notEqual(other)
      }

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        less(other)
      }

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <=(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        lessEqual(other)
      }

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        greater(other)
      }

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >=(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        greaterEqual(other)
      }
      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_-(implicit ev: IsNotQuantized[T]): Output[T] = {
        negate
      }

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def +(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        add(other)
      }

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def -(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        subtract(other)
      }

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def *(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        multiply(other)
      }

      private[this] def divHelper(
          x: Output[T],
          y: Output[T]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        if (x.dataType.isFloatingPoint || x.dataType.isComplex)
          Math.divide(x, y)
        else
          Math.truncateDivide(x, y)
      }

      /** $OpDocMathDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def /(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        divHelper(output, other)
      }

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def %(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        mod(other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def **(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        pow(other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ^(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        pow(other)
      }

      //endregion Operators

      //region Unary Ops

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def abs(implicit ev: IsReal[T]): Output[T] = {
        Math.abs(output)
      }

      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def negate(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.negate(output)
      }

      /** $OpDocMathReciprocal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def reciprocal(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.reciprocal(output)
      }

      /** $OpDocMathSquare
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def square(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.square(output)
      }

      /** $OpDocMathSqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sqrt(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.sqrt(output)
      }

      /** $OpDocMathRsqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def rsqrt(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.rsqrt(output)
      }

      /** $OpDocMathExp
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def exp(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.exp(output)
      }

      /** $OpDocMathExpm1
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def expm1(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.expm1(output)
      }

      /** $OpDocMathLog
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.log(output)
      }

      /** $OpDocMathLog1p
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log1p(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.log1p(output)
      }

      /** $OpDocMathSin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sin(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.sin(output)
      }

      /** $OpDocMathCos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cos(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.cos(output)
      }

      /** $OpDocMathTan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tan(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tan(output)
      }

      /** $OpDocMathAsin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asin(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.asin(output)
      }

      /** $OpDocMathAcos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acos(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.acos(output)
      }

      /** $OpDocMathAtan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.atan(output)
      }

      /** $OpDocMathSinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sinh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.sinh(output)
      }

      /** $OpDocMathCosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cosh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.cosh(output)
      }

      /** $OpDocMathTanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tanh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tanh(output)
      }

      /** $OpDocMathAsinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asinh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.asinh(output)
      }

      /** $OpDocMathAcosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acosh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.acosh(output)
      }

      /** $OpDocMathAtanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atanh(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.atanh(output)
      }

      /** $OpDocMathLogGamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logGamma(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.logGamma(output)
      }

      /** $OpDocMathDigamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def digamma(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.digamma(output)
      }

      /** $OpDocMathErf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erf(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.erf(output)
      }

      /** $OpDocMathErfc
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erfc(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.erfc(output)
      }

      /** $OpDocMathSigmoid
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sigmoid(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.sigmoid(output)
      }

      /** $OpDocMathLogSigmoid
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logSigmoid(implicit ev: IsDecimal[T]): Output[T] = {
        Math.logSigmoid(output)
      }

      /** $OpDocMathSign
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sign(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.sign(output)
      }

      /** $OpDocMathRound
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def round(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.round(output)
      }

      /** $OpDocMathRoundInt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def roundInt(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[T] = {
        Math.roundInt(output)
      }

      /** $OpDocMathFloor
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floor(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[T] = {
        Math.floor(output)
      }

      /** $OpDocMathCeil
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ceil(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[T] = {
        Math.ceil(output)
      }

      /** $OpDocMathIsNaN
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isNaN(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[Boolean] = {
        Math.isNaN(output)
      }

      /** $OpDocMathIsInf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isInf(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[Boolean] = {
        Math.isInf(output)
      }

      /** $OpDocMathIsFinite
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isFinite(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Output[Boolean] = {
        Math.isFinite(output)
      }

      //endregion Unary Ops

      //region Binary Ops

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def add(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.add(output, other)
      }

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def subtract(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.subtract(output, other)
      }

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def multiply(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.multiply(output, other)
      }

      /** $OpDocMathDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def divide(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.divide(output, other)
      }

      /** $OpDocMathFloorDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      @deprecated("Use `truncateDivide` instead.", "0.1")
      def floorDivide(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.floorDivide(output, other)
      }

      /** $OpDocMathTruncateDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateDivide(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.truncateDivide(output, other)
      }

      /** $OpDocMathRealDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def realDivide(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.realDivide(output, other)
      }

      /** $OpDocMathSquaredDifference
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def squaredDifference(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.squaredDifference(output, other)
      }

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def mod(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.mod(output, other)
      }

      /** $OpDocMathFloorMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floorMod(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.floorMod(output, other)
      }

      /** $OpDocMathTruncateMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateMod(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.truncateMod(output, other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def pow(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.pow(output, other)
      }

      /** $OpDocMathIgammac
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igammac(other: Output[T])(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.igammac(output, other)
      }

      /** $OpDocMathIgamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igamma(other: Output[T])(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.igamma(output, other)
      }

      /** $OpDocMathZeta
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zeta(other: Output[T])(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.zeta(output, other)
      }

      /** $OpDocMathPolygamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def polygamma(other: Output[T])(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.polygamma(output, other)
      }

      /** $OpDocMathAtan2
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan2(other: Output[T])(implicit ev: IsFloat32OrFloat64[T]): Output[T] = {
        Math.atan2(output, other)
      }

      /** $OpDocMathMinimum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def minimum(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.minimum(output, other)
      }

      /** $OpDocMathMaximum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def maximum(other: Output[T])(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.maximum(output, other)
      }

      //endregion Binary Ops

      //region Logical Ops

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalNot(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.logicalNot(output.asInstanceOf[Output[Boolean]])
      }

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalAnd(other: Output[Boolean])(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.logicalAnd(output.asInstanceOf[Output[Boolean]], other)
      }

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalOr(other: Output[Boolean])(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.logicalOr(output.asInstanceOf[Output[Boolean]], other)
      }

      /** $OpDocMathLogicalXOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalXOr(other: Output[Boolean])(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.logicalXOr(output.asInstanceOf[Output[Boolean]], other)
      }

      //endregion Logical Ops

      //region Comparison Ops

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def equal(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.equal(output, other)
      }

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def notEqual(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.notEqual(output, other)
      }

      /** $OpDocMathApproximatelyEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def approximatelyEqual(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.approximatelyEqual(output, other)
      }

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def less(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.less(output, other)
      }

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def lessEqual(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.lessEqual(output, other)
      }

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greater(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.greater(output, other)
      }

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greaterEqual(other: Output[T])(implicit ev: IsNumeric[T]): Output[Boolean] = {
        Math.greaterEqual(output, other)
      }

      //endregion Comparison Ops

      //region Reduction Ops

      /** $OpDocMathSum
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def sum[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNumeric[T]): Output[T] = {
        Math.sum(output, axes, keepDims)
      }

      /** $OpDocMathMean
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def mean[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.mean(output, axes, keepDims)
      }

      /** $OpDocMathProd
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def prod[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.prod(output, axes, keepDims)
      }

      /** $OpDocMathMin
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def min[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.min(output, axes, keepDims)
      }

      /** $OpDocMathMax
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def max[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.max(output, axes, keepDims)
      }

      /** $OpDocMathAll
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def all[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.all(output.asInstanceOf[Output[Boolean]], axes, keepDims)
      }

      /** $OpDocMathAny
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def any[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: T =:= Boolean): Output[Boolean] = {
        Math.any(output.asInstanceOf[Output[Boolean]], axes, keepDims)
      }

      /** $OpDocMathLogSumExp
        *
        * @group MathOps
        * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def logSumExp[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.logSumExp(output, axes, keepDims)
      }

      /** $OpDocMathCountNonZero
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def countNonZero[I: IntDefault : TF : IsInt32OrInt64](
          axes: Output[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNumeric[T]): Output[Long] = {
        Math.countNonZero(output, axes, keepDims)
      }

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmin[I: TF : IsInt32OrInt64](
          axes: Output[I]
      )(implicit ev: IsNotQuantized[T]): Output[Long] = {
        Math.argmin(output, axes, outputDataType = INT64)
      }

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmin[I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
          axes: Output[I],
          outputDataType: DataType[IR]
      )(implicit ev: IsNotQuantized[T]): Output[IR] = {
        Math.argmin(output, axes, outputDataType)
      }

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmax[I: TF : IsInt32OrInt64](
          axes: Output[I]
      )(implicit ev: IsNotQuantized[T]): Output[Long] = {
        Math.argmax(output, axes, outputDataType = INT64)
      }

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmax[I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
          axes: Output[I],
          outputDataType: DataType[IR]
      )(implicit ev: IsNotQuantized[T]): Output[IR] = {
        Math.argmax(output, axes, outputDataType)
      }

      /** $OpDocMathCumsum
        *
        * @group MathOps
        * @param  axis      Tensor containing the axis along which to perform the cumulative sum.
        * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
        * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
        * @return Result as a new tensor.
        */
      def cumsum[I: TF : IsInt32OrInt64](
          axis: Output[I],
          exclusive: Boolean = false,
          reverse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.cumsum(output, axis, exclusive, reverse)
      }

      /** $OpDocMathCumprod
        *
        * @group MathOps
        * @param  axis      Tensor containing the axis along which to perform the cumulative product.
        * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
        * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
        * @return Result as a new tensor.
        */
      def cumprod[I: TF : IsInt32OrInt64](
          axis: Output[I],
          exclusive: Boolean = false,
          reverse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.cumprod(output, axis, exclusive, reverse)
      }

      //endregion Reduction Ops

      /** $OpDocMathBinCount
        *
        * @group MathOps
        * @param  dataType  If `weights` is `null`, this determines the data type used for the output tensor (i.e., the
        *                   tensor containing the bin counts).
        * @param  weights   If not `null`, this tensor must have the same shape as `input`. For each value in `input`,
        *                   the corresponding bin count will be incremented by the corresponding weight instead of `1`.
        * @param  minLength If not `null`, this ensures the output has length at least `minLength`, padding with zeros
        *                   at the end, if necessary.
        * @param  maxLength If not `null`, this skips values in `input` that are equal or greater than `maxLength`,
        *                   ensuring that the output has length at most `maxLength`.
        * @return Created op output.
        */
      def binCount[R: TF : IsInt32OrInt64OrFloat32OrFloat64](
          dataType: DataType[R],
          weights: Output[R] = null,
          minLength: Output[Int] = null,
          maxLength: Output[Int] = null
      )(implicit ev: T =:= Int): Output[R] = {
        Math.binCount(output.asInstanceOf[Output[Int]], dataType, weights, minLength, maxLength)
      }

      //region Segment Ops

      /** $OpDocMathSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentSum[I: TF : IsInt32OrInt64](
          segmentIndices: Output[I]
      )(implicit ev: IsNumeric[T]): Output[T] = {
        Math.segmentSum(output, segmentIndices)
      }

      /** $OpDocMathSegmentMean
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMean[I: TF : IsInt32OrInt64](
          segmentIndices: Output[I]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.segmentMean(output, segmentIndices)
      }

      /** $OpDocMathSegmentProd
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentProd[I: TF : IsInt32OrInt64](
          segmentIndices: Output[I]
      )(implicit ev: IsNumeric[T]): Output[T] = {
        Math.segmentProd(output, segmentIndices)
      }

      /** $OpDocMathSegmentMin
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMin[I: TF : IsInt32OrInt64](
          segmentIndices: Output[I]
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.segmentMin(output, segmentIndices)
      }

      /** $OpDocMathSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMax[I: TF : IsInt32OrInt64](
          segmentIndices: Output[I]
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.segmentMax(output, segmentIndices)
      }

      /** $OpDocMathUnsortedSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentSum[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Output[I1],
          segmentsNumber: Output[I2]
      )(implicit ev: IsNumeric[T]): Output[T] = {
        Math.unsortedSegmentSum(output, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathUnsortedSegmentMean
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Created op output.
        */
      def unsortedSegmentMean[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Output[I1],
          segmentsNumber: Output[I2]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.unsortedSegmentMean(output, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathUnsortedSegmentProd
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Created op output.
        */
      def unsortedSegmentProd[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Output[I1],
          segmentsNumber: Output[I2]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.unsortedSegmentProd(output, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathUnsortedSegmentMin
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentMin[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Output[I1],
          segmentsNumber: Output[I2]
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.unsortedSegmentMin(output, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathUnsortedSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentMax[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Output[I1],
          segmentsNumber: Output[I2]
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.unsortedSegmentMax(output, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathSparseSegmentSum
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentSum[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Output[I1],
          segmentIndices: Output[Int],
          numSegments: Output[I2] = null
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.sparseSegmentSum(output, indices, segmentIndices, numSegments)
      }

      /** $OpDocMathSparseSegmentMean
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentMean[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Output[I1],
          segmentIndices: Output[Int],
          numSegments: Output[I2] = null
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.sparseSegmentMean(output, indices, segmentIndices, numSegments)
      }

      /** $OpDocMathSparseSegmentSumSqrtN
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentSumSqrtN[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Output[I1],
          segmentIndices: Output[Int],
          numSegments: Output[I2] = null
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.sparseSegmentSumSqrtN(output, indices, segmentIndices, numSegments)
      }

      //endregion Segment Ops

      //region Matrix Ops

      /** $OpDocMathDiag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diag(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.diag(output)
      }

      /** $OpDocMathDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diagPart(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.diagPart(output)
      }

      /** $OpDocMathMatrixDiag
        *
        * @group MathOps
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its
        *         last dimension duplicated.
        */
      def matrixDiag: Output[T] = {
        Math.matrixDiag(output)
      }

      /** $OpDocMathMatrixSetDiag
        *
        * @group MathOps
        * @param  diagonal Rank-`K` tensor, where `K >= 1`.
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
        */
      def matrixSetDiag(diagonal: Output[T]): Output[T] = {
        Math.matrixSetDiag(output, diagonal)
      }

      /** $OpDocMathMatrixDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor containing the diagonal(s) and having shape equal to
        *         `input.shape[:-2] + [min(input.shape[-2:])]`.
        */
      def matrixDiagPart: Output[T] = {
        Math.matrixDiagPart(output)
      }

      /** $OpDocMathMatrixBandPart
        *
        * @group MathOps
        * @param  numSubDiagonals   Scalar tensor that contains the number of sub-diagonals to keep. If negative,
        *                           the entire lower triangle is kept.
        * @param  numSuperDiagonals Scalar tensor that contains the number of super-diagonals to keep. If negative,
        *                           the entire upper triangle is kept.
        * @return Tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
        */
      def matrixBandPart[I: TF : IsInt32OrInt64](
          numSubDiagonals: Output[I],
          numSuperDiagonals: Output[I]
      ): Output[T] = {
        Math.matrixBandPart(output, numSubDiagonals, numSuperDiagonals)
      }

      /** $OpDocMathTrace
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def trace(implicit ev: IsNumeric[T]): Output[T] = {
        Math.trace(output)
      }

      /** $OpDocMathMatmul
        *
        * @group MathOps
        * @param  other      Tensor to multiply with.
        * @param  transposeA If `true`, this tensor is transposed before the multiplication.
        * @param  transposeB If `true`, `other` is transposed before the multiplication.
        * @param  conjugateA If `true`, this tensor is conjugated before the multiplication.
        * @param  conjugateB If `true`, `other` is conjugated before the multiplication.
        * @param  aIsSparse  If `true`, this tensor is treated as a sparse matrix (i.e., it is assumed it contains many
        *                    zeros).
        * @param  bIsSparse  If `true`, `other` is treated as a sparse matrix (i.e., it is assumed it contains many
        *                    zeros).
        * @return Result as a new tensor.
        */
      def matmul(
          other: Output[T],
          transposeA: Boolean = false,
          transposeB: Boolean = false,
          conjugateA: Boolean = false,
          conjugateB: Boolean = false,
          aIsSparse: Boolean = false,
          bIsSparse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.matmul(output, other, transposeA, transposeB, conjugateA, conjugateB, aIsSparse, bIsSparse)
      }

      /** $OpDocMathCross
        *
        * @group MathOps
        * @param  other Tensor to multiply with.
        * @return Result as a new tensor.
        */
      def cross(
          other: Output[T]
      )(implicit ev: IsReal[T]): Output[T] = {
        Math.cross(output, other)
      }

      /** $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other   Tensor to contract with.
        * @param  numAxes Number of axes to contract.
        * @return Created op output.
        */
      def tensorDot(
          other: Output[T],
          numAxes: Int
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tensorDot(output, other, numAxes)
      }

      /** $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other Tensor to contract with.
        * @param  axesA Axes to contract in `a`.
        * @param  axesB Axes to contract in `b`.
        * @return Created op output.
        */
      def tensorDot(
          other: Output[T],
          axesA: Seq[Int],
          axesB: Seq[Int]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tensorDot(output, other, axesA, axesB)
      }

      /** Dynamic version (i.e., where `numAxes` may be a tensor) of the `tensorDot` op.
        *
        * $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other   Tensor to contract with.
        * @param  numAxes Number of axes to contract.
        * @return Created op output.
        */
      def tensorDotDynamic(
          other: Output[T],
          numAxes: Output[Int]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tensorDotDynamic(output, other, numAxes)
      }

      /** Dynamic version (i.e., where `axesA` and `axesB` may be tensors) of the `tensorDot` op.
        *
        * $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other Tensor to contract with.
        * @param  axesA Axes to contract in `a`.
        * @param  axesB Axes to contract in `b`.
        * @return Created op output.
        */
      def tensorDotDynamic(
          other: Output[T],
          axesA: Output[Int],
          axesB: Output[Int]
      )(implicit ev: IsNotQuantized[T]): Output[T] = {
        Math.tensorDotDynamic(output, other, axesA, axesB)
      }

      //endregion Matrix Ops

      //region Complex Ops

      /** $OpDocMathConjugate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def conjugate(implicit ev: IsComplex[T]): Output[T] = {
        Math.conjugate(output)
      }

      //endregion Complex Ops

      //region Bucketization Ops

      /** $OpDocMathBucketize
        *
        * @group MathOps
        * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
        * @return Result as a new tensor.
        */
      def bucketize(
          boundaries: Seq[Float]
      )(implicit ev: IsInt32OrInt64OrFloat32OrFloat64[T]): Output[T] = {
        Math.bucketize(output, boundaries)
      }

      //endregion Bucketization Ops

      //region Other Ops

      /** $OpDocMathZerosFraction
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zerosFraction(implicit ev: IsNumeric[T]): Output[Float] = {
        Math.zerosFraction(output)
      }

      //endregion Other Ops
    }

    implicit class FloatMathOps(val output: Output[Float]) {
      /** Creates a new complex number with the provided imaginary part.
        *
        * @param  imag Imaginary part.
        * @return Resulting complex number.
        */
      def toComplex(imag: Output[Float] = 0.0f): Output[ComplexFloat] = {
        Math.complexFloat(output.asInstanceOf[Output[Float]], imag)
      }
    }

    implicit class DoubleMathOps(val output: Output[Double]) {
      /** Creates a new complex number with the provided imaginary part.
        *
        * @param  imag Imaginary part.
        * @return Resulting complex number.
        */
      def toComplex(imag: Output[Double] = 0.0): Output[ComplexDouble] = {
        Math.complexDouble(output.asInstanceOf[Output[Double]], imag)
      }
    }

    implicit class ComplexFloatMathOps(val output: Output[ComplexFloat]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Output[Float] = {
        Math.realFloat(output.asInstanceOf[Output[ComplexFloat]])
      }

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Output[Float] = {
        Math.imagFloat(output.asInstanceOf[Output[ComplexFloat]])
      }

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def magnitude: Output[Float] = {
        Math.magnitudeFloat(output.asInstanceOf[Output[ComplexFloat]])
      }

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Output[Float] = {
        Math.angleFloat(output.asInstanceOf[Output[ComplexFloat]])
      }
    }

    implicit class ComplexDoubleMathOps(val output: Output[ComplexDouble]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Output[Double] = {
        Math.realDouble(output.asInstanceOf[Output[ComplexDouble]])
      }

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Output[Double] = {
        Math.imagDouble(output.asInstanceOf[Output[ComplexDouble]])
      }

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def magnitude: Output[Double] = {
        Math.magnitudeDouble(output.asInstanceOf[Output[ComplexDouble]])
      }

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Output[Double] = {
        Math.angleDouble(output.asInstanceOf[Output[ComplexDouble]])
      }
    }
  }

  /** Helper function for reduction ops that computes the reduction output shape, assuming `keepDims` is `true`.
    *
    * For example:
    * {{{
    *   // inputShape == [2, 3, 5, 7]
    *   // axes = [1, 2]
    *   reducedShape(inputShape, axes) ==> [2, 1, 1, 7]
    * }}}
    *
    * @param  inputShape Shape of the tensor being reduced.
    * @param  axes       Reduction axes.
    * @return One-dimensional tensor representing the reduction output shape, assuming `keepDims` is `true`.
    */
  private[api] def reducedShape[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      inputShape: Output[I1],
      axes: Output[I2]
  ): Output[I1] = {
    // Cast needed for SparseOutput reductions.
    val inputRank = Basic.size(inputShape).castTo[Int]
    val reshapedAxes = {
      if (axes.rank == 0)
        Basic.reshape(axes, Seq(1))
      else
        axes
    }
    val intAxes = floorMod(add(reshapedAxes.castTo[Int], inputRank), inputRank)
    val axesShape = Basic.shape(intAxes).castTo[Int]
    DataFlow.dynamicStitch(
      Seq(range(Basic.constant(0), inputRank), intAxes),
      Seq(inputShape, Basic.ones[I1, Int](axesShape)))
  }

  /** @define OpDocMathSelect
    *   The `select` op selects elements from `x` or `y`, depending on `condition`.
    *
    *   The `x`, and `y` tensors must have the same shape. The output tensor will also have the same shape.
    *
    *   The `condition` tensor must be a scalar if `x` and `y` are scalars. If `x` and `y` are vectors or higher rank,
    *   then `condition` must be either a scalar, or a vector with size matching the first dimension of `x`, or it must
    *   have the same shape as `x`.
    *
    *   The `condition` tensor acts as a mask that chooses, based on the value at each element, whether the
    *   corresponding element / row in the output should be taken from `x` (if true) or `y` (if false).
    *
    *   If `condition` is a vector and `x` and `y` are higher rank matrices, then it chooses which row (outer dimension)
    *   to copy from `x` and `y`. If `condition` has the same shape as `x` and `y`, then it chooses which element to
    *   copy from `x` and `y`.
    *
    *   For example:
    *   {{{
    *     // 'condition' tensor is [[true,  false], [false, true]]
    *     // 'x' is [[1, 2], [3, 4]]
    *     // 'y' is [[5, 6], [7, 8]]
    *     select(condition, x, y) ==> [[1, 6], [7, 4]]
    *
    *     // 'condition' tensor is [true, false]
    *     // 'x' is [[1, 2], [3, 4]]
    *     // 'y' is [[5, 6], [7, 8]]
    *     select(condition, x, y) ==> [[1, 2], [7, 8]]
    *   }}}
    *
    * @define OpDocMathRange
    *   The `range` op constructs a sequence of numbers.
    *
    *   The op creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not
    *   including `limit`. The data type of the resulting tensor is inferred from the inputs unless it is provided
    *   explicitly.
    *
    *   For example:
    *   {{{
    *     // 'start' is 3
    *     // 'limit' is 18
    *     // 'delta' is 3
    *     range(start, limit, delta) ==> [3, 6, 9, 12, 15]
    *
    *     // 'start' is 3
    *     // 'limit' is 1
    *     // 'delta' is -0.5
    *     range(start, limit, delta) ==> [3.0, 2.5, 2.0, 1.5]
    *   }}}
    *
    * @define OpDocMathLinspace
    *   The `linspace` op generates values in an interval.
    *
    *   The op generates a sequence of `numberOfValues` evenly-spaced values beginning at `start`. If
    *   `numberOfValues > 1`, the values in the sequence increase by `(stop - start) / (numberOfValues - 1)`, so that the
    *   last value is exactly equal to `stop`.
    *
    *   For example:
    *   {{{
    *     linspace(10.0, 12.0, 3) ==> [10.0  11.0  12.0]
    *   }}
    *
    * @define OpDocMathAddN
    *   The `addN` op adds all input tensors element-wise.
    *
    * @define OpDocMathAccumulateN
    *   The `accumulateN` op adds all input tensors element-wise.
    *
    *   This op performs the same operation as the `addN` op, but it does not wait for all of its inputs to be ready
    *   before beginning to sum. This can save memory if the inputs become available at different times, since the
    *   minimum temporary storage is proportional to the output size, rather than the inputs size.
    *
    * @define OpDocMathAbs
    *   The `abs` op computes the absolute value of a tensor.
    *
    *   Given a tensor `x` of real numbers, the op returns a tensor containing the absolute value of each element in
    *   `x`. For example, if `x` is an input element and `y` is an output element, the op computes `y = |x|`.
    *
    *   Given a tensor `x` of complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that is the
    *   magnitude value of each element in `x`. All elements in `x` must be complex numbers of the form `a + bj`. The
    *   magnitude is computed as `\sqrt{a^2 + b^2}`. For example:
    *   {{{
    *     // Tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
    *     abs(x) ==> [5.25594902, 6.60492229]
    *   }}}
    *
    * @define OpDocMathNegate
    *   The `negate` op computes the numerical negative value of a tensor element-wise. I.e., `y = -x`.
    *
    * @define OpDocMathReciprocal
    *   The `reciprocal` op computes the reciprocal value of a tensor element-wise. I.e., `y = 1 / x`.
    *
    * @define OpDocMathSquare
    *   The `square` op computes the square of a tensor element-wise. I.e., `y = x * x = x^2`.
    *
    * @define OpDocMathSqrt
    *   The `sqrt` op computes the square root of a tensor element-wise. I.e., `y = \sqrt{x} = x^{1/2}`.
    *
    * @define OpDocMathRsqrt
    *   The `rsqrt` op computes the reciprocal of the square root of a tensor element-wise. I.e.,
    *   `y = 1 / \sqrt{x} = 1 / x^{1/2}`.
    *
    * @define OpDocMathExp
    *   The `exp` op computes the exponential of a tensor element-wise. I.e., `y = \exp{x} = e^x`.
    *
    * @define OpDocMathExpm1
    *   The `expm1` op computes the exponential of a tensor minus `1` element-wise. I.e., `y = \exp{x} - 1`.
    *
    * @define OpDocMathLog
    *   The `log` op computes the logarithm of a tensor element-wise. I.e., `y = \log{x}`.
    *
    * @define OpDocMathLog1p
    *   The `log1p` op computes the logarithm of a tensor plus `1` element-wise. I.e., `y = \log{1 + x}`.
    *
    * @define OpDocMathSin
    *   The `sin` op computes the sine of a tensor element-wise. I.e., `y = \sin{x}`.
    *
    * @define OpDocMathCos
    *   The `cos` op computes the cosine of a tensor element-wise. I.e., `y = \cos{x}`.
    *
    * @define OpDocMathTan
    *   The `tan` op computes the tangent of a tensor element-wise. I.e., `y = \tan{x}`.
    *
    * @define OpDocMathAsin
    *   The `asin` op computes the inverse sine of a tensor element-wise. I.e., `y = \asin{x}`.
    *
    * @define OpDocMathAcos
    *   The `acos` op computes the inverse cosine of a tensor element-wise. I.e., `y = \acos{x}`.
    *
    * @define OpDocMathAtan
    *   The `atan` op computes the inverse tangent of a tensor element-wise. I.e., `y = \atan{x}`.
    *
    * @define OpDocMathSinh
    *   The `sinh` op computes the hyperbolic sine of a tensor element-wise. I.e., `y = \sinh{x}`.
    *
    * @define OpDocMathCosh
    *   The `cosh` op computes the hyperbolic cosine of a tensor element-wise. I.e., `y = \cosh{x}`.
    *
    * @define OpDocMathTanh
    *   The `tanh` op computes the hyperbolic tangent of a tensor element-wise. I.e., `y = \tanh{x}`.
    *
    * @define OpDocMathAsinh
    *   The `asinh` op computes the inverse hyperbolic sine of a tensor element-wise. I.e., `y = \asinh{x}`.
    *
    * @define OpDocMathAcosh
    *   The `acosh` op computes the inverse hyperbolic cosine of a tensor element-wise. I.e., `y = \acosh{x}`.
    *
    * @define OpDocMathAtanh
    *   The `atanh` op computes the inverse hyperbolic tangent of a tensor element-wise. I.e., `y = \atanh{x}`.
    *
    * @define OpDocMathLogGamma
    *   The `logGamma` op computes the logarithm of the absolute value of the Gamma function applied element-wise on a
    * 	tensor. I.e., `y = \log{|\Gamma{x}|}`.
    *
    * @define OpDocMathDigamma
    *   The `digamma` op computes the derivative of the logarithm of the absolute value of the Gamma function applied
    * 	element-wise on a tensor (i.e., the digamma or Psi function). I.e., `y = \partial\log{|\Gamma{x}|}`.
    *
    * @define OpDocMathErf
    *   The `erf` op computes the Gaussian error function element-wise on a tensor.
    *
    * @define OpDocMathErfc
    *   The `erfc` op computes the complementary Gaussian error function element-wise on a tensor.
    *
    * @define OpDocMathSigmoid
    *   The `sigmoid` op computes the sigmoid function element-wise on a tensor.
    *
    *   Specifically, `y = 1 / (1 + exp(-x))`.
    *
    * @define OpDocMathLogSigmoid
    *   The `logSigmoid` op computes the log-sigmoid function element-wise on a tensor.
    *
    *   Specifically, `y = log(1 / (1 + exp(-x)))`.  For numerical stability, we use `y = -tf.nn.softplus(-x)`.
    *
    * @define OpDocMathSign
    *   The `sign` op computes an element-wise indication of the sign of a tensor.
    *
    * 	I.e., `y = sign(x) = -1` if `x < 0`; `0` if `x == 0`; `1` if `x > 0`.
    *
    * 	Zero is returned for `NaN` inputs.
    *
    * 	For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
    *
    * @define OpDocMathRound
    *   The `round` op computes the round value of a tensor element-wise.
    *
    * 	Rounds half to even. Also known as bankers rounding. If you want to round according to the current system
    *		rounding mode use the [[roundInt]] op instead.
    *
    * 	For example:
    * 	{{{
    *   	// 'a' is [0.9, 2.5, 2.3, 1.5, -4.5]
    *   	round(a) ==> [1.0, 2.0, 2.0, 2.0, -4.0]
    * 	}}}
    *
    * @define OpDocMathRoundInt
    *   The `roundInt` op computes the round value of a tensor element-wise.
    *
    * 	If the result is midway between two representable values, the even representable is chosen.
    *
    * 	For example:
    * 	{{{
    *   	roundInt(-1.5) ==> -2.0
    *   	roundInt(0.5000001) ==> 1.0
    *   	roundInt([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
    * 	}}}
    *
    * @define OpDocMathFloor
    *   The `floor` op computes the largest integer not greater than the current value of a tensor, element-wise.
    *
    * @define OpDocMathCeil
    *   The `ceil` op computes the smallest integer not greater than the current value of a tensor, element-wise.
    *
    * @define OpDocMathIsNaN
    *   The `isNaN` op returns a boolean tensor indicating which elements of a tensor are NaN-valued.
    *
    * @define OpDocMathIsInf
    *   The `isInf` op returns a boolean tensor indicating which elements of a tensor are Inf-valued.
    *
    * @define OpDocMathIsFinite
    *   The `isFinite` op returns a boolean tensor indicating which elements of a tensor are finite-valued.
		*
		* @define OpDocMathAdd
		*   The `add` op adds two tensors element-wise. I.e., `z = x + y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathSubtract
		*   The `subtract` op subtracts two tensors element-wise. I.e., `z = x - y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathMultiply
		*   The `multiply` op multiplies two tensors element-wise. I.e., `z = x * y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathDivide
		*   The `divide` op divides two tensors element-wise. I.e., `z = x / y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathFloorDivide
		*   The `floorDivide` op floor-divides two tensors element-wise. I.e., `z = x // y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathTruncateDivide
		*   The `truncateDivide` op truncate-divides two tensors element-wise.
		*
		*   Truncation designates that negative numbers will round fractional quantities toward zero. I.e. `-7 / 5 = 1`.
		*   This matches C semantics but it is different than Python semantics. See `floorDivide` for a division function
		*   that matches Python semantics.
    *
    *   I.e., `z = x / y`, for `x` and `y` being integer tensors.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathRealDivide
		*   The `realDivide` op divides two real tensors element-wise.
		*
		*   If `x` and `y` are real-valued tensors, the op will return the floating-point division.
    *
    *   I.e., `z = x / y`, for `x` and `y` being real tensors.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathSquaredDifference
		*   The `squaredDifference` op computes the squared difference between two tensors element-wise.
		*   I.e., `z = (x - y) * (x - y)`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathMod
		*   The `mod` op computes the remainder of the division between two tensors element-wise.
    *
    *   The op emulates C semantics in that the result is consistent with a truncating divide.
    *   E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathFloorMod
		*   The `floorMod` op computes the remainder of the division between two tensors element-wise.
    *
    *   When `x < 0` xor `y < 0` is true, the op follows Python semantics in that the result here is
    *   consistent with a flooring divide. E.g., `floor(x / y) * y + mod(x, y) = x`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathTruncateMod
		*   The `truncateMod` op computes the remainder of the division between two tensors element-wise.
    *
    *   The op emulates C semantics in that the result here is consistent with a truncating divide.
    *   E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathPow
		*   The `pow` op computes the power of one tensor raised to another, element-wise.
    *
    *   Given a tensor `x` and a tensor `y`, the op computes `x^y` for the corresponding elements in `x`
    *   and `y`.
    *
    *   For example:
    *   {{{
    *     // Tensor 'x' is [[2, 2], [3, 3]]
    *     // Tensor 'y' is [[8, 16], [2, 3]]
    *     pow(x, y) ==> [[256, 65536], [9, 27]]
    *   }}}
		*
		* @define OpDocMathIgammac
		*   The `igammac` op computes the upper regularized incomplete Gamma function `Q(a, x)`.
    *
    *   The upper regularized incomplete Gamma function is defined as:
    *
    *   `Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)`, where:
    *
    *   `Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt`
    *
    *   is the upper incomplete Gama function.
    *
    *   Note that, above, `P(a, x)` (`Igamma`) is the lower regularized complete Gamma function.
		*
		* @define OpDocMathIgamma
		*   The `igamma` op computes the lower regularized incomplete Gamma function `Q(a, x)`.
    *
    *   The lower regularized incomplete Gamma function is defined as:
    *
    *   `P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)`, where:
    *
    *   `Gamma(a, x) = \int_{0}^{x} t^{a-1} exp(-t) dt`
    *
    *   is the lower incomplete Gamma function.
    *
    *   Note that, above, `Q(a, x)` (`Igammac`) is the upper regularized complete Gamma function.
		*
		* @define OpDocMathZeta
		*   The `zeta` op computes the Hurwitz zeta function `\zeta(x, q)`.
    *
    *   The Hurwitz zeta function is defined as:
    *
    *   `\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}`.
		*
		* @define OpDocMathPolygamma
		*   The `polygamma` op computes the polygamma function `\psi^{(n)}(x)`.
    *
    *   The polygamma function is defined as:
    *
    *   `\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)`, where `\psi(x)` is the digamma function.
		*
		* @define OpDocMathAtan2
		*   The `atan2` op computes the inverse tangent of `x / y` element-wise, respecting signs of the arguments.
    *
    *   The op computes the angle `\theta \in [-\pi, \pi]` such that `y = r \cos(\theta)` and
    *   `x = r \sin(\theta)`, where `r = \sqrt(x^2 + y^2)`.
    *
    * @define OpDocMathMinimum
    *   The `minimum` op returns the element-wise minimum between two tensors. I.e., `z = x < y ? x : y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		*
		* @define OpDocMathMaximum
		*   The `maximum` op returns the element-wise maximum between two tensors. I.e., `z = x > y ? x : y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathIncompleteBeta
    *   The `incompleteBeta` op computes the regularized incomplete beta integral `I_x(a, b)`.
    *
    *   The regularized incomplete beta integral is defined as:
    *
    *   `I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}`, where:
    *
    *   `B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt`
    *
    *   is the incomplete beta function and `B(a, b)` is the *complete* beta function.
    *
    * @define OpDocMathLogicalNot
    *   The `logicalNot` op computes the truth value of `!x` element-wise.
    *
    * @define OpDocMathLogicalAnd
    *   The `logicalAnd` op computes the truth value of `x && y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathLogicalOr
    *   The `logicalOr` op computes the truth value of `x || y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathLogicalXOr
    *   The `logicalXOr` op computes the truth value of `(x || y) && !(x && y)` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathEqual
    *   The `equal` op computes the truth value of `x == y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathNotEqual
    *   The `notEqual` op computes the truth value of `x != y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathApproximatelyEqual
    *   The `approximatelyEqual` op computes the truth value of `abs(x - y) < tolerance` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathLess
    *   The `less` op computes the truth value of `x < y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathLessEqual
    *   The `lessEqual` op computes the truth value of `x <= y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathGreater
    *   The `greater` op computes the truth value of `x > y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathGreaterEqual
    *   The `greaterEqual` op computes the truth value of `x >= y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @define OpDocMathSum
    *   The `sum` op computes the sum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *     sum(x) ==> 6
    *     sum(x, 0) ==> [2, 2, 2]
    *     sum(x, 1) ==> [3, 3]
    *     sum(x, 1, keepDims = true) ==> [[3], [3]]
    *     sum(x, [0, 1]) ==> 6
    *   }}}
    *
    * @define OpDocMathMean
    *   The `mean` op computes the mean of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     mean(x) ==> 1.5
    *     mean(x, 0) ==> [1.5, 1.5]
    *     mean(x, 1) ==> [1.0, 2.0]
    *   }}}
    *
    * @define OpDocMathProd
    *   The `prod` op computes the product of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *     prod(x) ==> 1
    *     prod(x, 0) ==> [1, 1, 1]
    *     prod(x, 1) ==> [1, 1]
    *     prod(x, 1, keepDims = true) ==> [[1], [1]]
    *     prod(x, [0, 1]) ==> 1
    *   }}}
    *
    * @define OpDocMathMin
    *   The `min` op computes the minimum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     min(x) ==> 1.0
    *     min(x, 0) ==> [1.0, 1.0]
    *     min(x, 1) ==> [1.0, 2.0]
    *   }}}
    *
    * @define OpDocMathMax
    *   The `max` op computes the maximum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     max(x) ==> 2.0
    *     max(x, 0) ==> [2.0, 2.0]
    *     max(x, 1) ==> [1.0, 2.0]
    *   }}}
    *
    * @define OpDocMathAll
    *   The `all` op computes the logical AND of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[true, true], [false, false]]
    *     all(x) ==> false
    *     all(x, 0) ==> [false, false]
    *     all(x, 1) ==> [true, false]
    *   }}}
    *
    * @define OpDocMathAny
    *   The `any` op computes the logical OR of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[true, true], [false, false]]
    *     any(x) ==> true
    *     any(x, 0) ==> [true, true]
    *     any(x, 1) ==> [true, false]
    *   }}}
    *
    * @define OpDocMathLogSumExp
    *   The `logSumExp` op computes the log-sum-exp of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[0, 0, 0], [0, 0, 0]]
    *     logSumExp(x) ==> log(6)
    *     logSumExp(x, 0) ==> [log(2), log(2), log(2)]
    *     logSumExp(x, 1) ==> [log(3), log(3)]
    *     logSumExp(x, 1, keepDims = true) ==> [[log(3)], [log(3)]]
    *     logSumExp(x, [0, 1]) ==> log(6)
    *   }}}
    *
    * @define OpDocMathCountNonZero
    *   The `countNonZero` op computes the number of non-zero elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
		*   '''IMPORTANT NOTE:''' Floating point comparison to zero is done by exact floating point equality check. Small
    *   values are '''not''' rounded to zero for the purposes of the non-zero check.
		*
    *   For example:
    *   {{{
    *     // 'x' is [[0, 1, 0], [1, 1, 0]]
    *     countNonZero(x) ==> 3
    *     countNonZero(x, 0) ==> [1, 2, 0]
    *     countNonZero(x, 1) ==> [1, 2]
    *     countNonZero(x, 1, keepDims = true) ==> [[1], [2]]
    *     countNonZero(x, [0, 1]) ==> 3
    *   }}}
    *
    *   '''IMPORTANT NOTE:''' Strings are compared against zero-length empty string `""`. Any string with a size greater
    *   than zero is already considered as nonzero.
    *
    *   For example:
    *   {{{
    *     // 'x' is ["", "a", "  ", "b", ""]
    *     countNonZero(x) ==> 3 // "a", "  ", and "b" are treated as nonzero strings.
    *   }}}
    *
    * @define OpDocMathArgmax
    *   The `argmax` op returns the indices with the largest value across axes of a tensor.
    *
    *   Note that in case of ties the identity of the return value is not guaranteed.
    *
    * @define OpDocMathArgmin
    *   The `argmin` op returns the indices with the smallest value across axes of a tensor.
    *
    *   Note that in case of ties the identity of the return value is not guaranteed.
    *
    * @define OpDocMathBinCount
    *   The `binCount` op counts the number of occurrences of each value in an integer tensor.
    *
    *   If `minLength` and `maxLength` are not provided, the op returns a vector with length `max(input) + 1`, if
    *   `input` is non-empty, and length `0` otherwise.
    *
    *   If `weights` is not `null`, then index `i` of the output stores the sum of the value in `weights` at each
    *   index where the corresponding value in `input` is equal to `i`.
    *
    * @define OpDocMathCumsum
    *   The `cumsum` op computes the cumulative sum of the tensor along an axis.
    *
    *   By default, the op performs an inclusive cumulative sum, which means that the first element of the input is
    *   identical to the first element of the output:
    *   {{{
    *     cumsum([a, b, c]) ==> [a, a + b, a + b + c]
    *   }}}
    *
    *   By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed instead:
    *   {{{
    *     cumsum([a, b, c], exclusive = true) ==> [0, a, a + b]
    *   }}}
    *
    *   By setting the `reverse` argument to `true`, the cumulative sum is performed in the opposite direction:
    *   {{{
    *     cumsum([a, b, c], reverse = true) ==> [a + b + c, b + c, c]
    *   }}}
    *
    *   This is more efficient than using separate [[Basic.reverse]] ops.
    *
    *   The `reverse` and `exclusive` arguments can also be combined:
    *   {{{
    *     cumsum([a, b, c], exclusive = true, reverse = true) ==> [b + c, c, 0]
    *   }}}
    *
    * @define OpDocMathCumprod
    *   The `cumprod` op computes the cumulative product of the tensor along an axis.
    *
    *   By default, the op performs an inclusive cumulative product, which means that the first element of the input
    *   is identical to the first element of the output:
    *   {{{
    *    cumprod([a, b, c]) ==> [a, a * b, a * b * c]
    *   }}}
    *
    *   By setting the `exclusive` argument to `true`, an exclusive cumulative product is performed instead:
    *   {{{
    *     cumprod([a, b, c], exclusive = true) ==> [0, a, a * b]
    *   }}}
    *
    *   By setting the `reverse` argument to `true`, the cumulative product is performed in the opposite direction:
    *   {{{
    *     cumprod([a, b, c], reverse = true) ==> [a * b * c, b * c, c]
    *   }}}
    *
    *   This is more efficient than using separate [[Basic.reverse]] ops.
    *
    *   The `reverse` and `exclusive` arguments can also be combined:
    *   {{{
    *     cumprod([a, b, c], exclusive = true, reverse = true) ==> [b * c, c, 0]
    *   }}}
    *
    * @define OpDocMathSegmentSum
    *   The `segmentSum` op computes the sum along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \sum_{j...} data(j,...)` where the sum is over all `j` such
    *   that `segmentIndices(j) == i`. Unlike `unsortedSegmentSum`, `segmentIndices` need be sorted.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSegmentMean
    *   The `segmentMean` op computes the mean along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \frac{sum_{j...} data(j,...)}{N}` where the sum is over
    *   all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike
    *   `unsortedSegmentMean`, `segmentIndices` need to be sorted.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSegmentProd
    *   The `segmentProd` op computes the product along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \prod_{j...} data(j,...)` where the product is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentProd`, `segmentIndices` need be sorted.
    *
    *   If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSegmentMin
    *   The `segmentMin` op computes the min along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \min_{j...} data(j,...)` where the min is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentMin`, `segmentIndices` need be sorted.
    *
    *   If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSegmentMax
    *   The `segmentMax` op computes the max along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \max_{j...} data(j,...)` where the max is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentMax`, `segmentIndices` need be sorted.
    *
    *   If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentSum
    *   The `unsortedSegmentSum` op computes the sum along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \sum_{j...} data(j...)` where the sum is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentSum`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentMean
    *   The `unsortedSegmentMean` op computes the mean along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \frac{\sum_{j...} data(j...)}{N}` where the sum is over
    *   all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike
    *   `segmentSum`, `segmentIndices` need not be sorted and need not cover all values in the full range of valid
    *   values.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentProd
    *   The `unsortedSegmentProd` op computes the product along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \prod_{j...} data(j...)` where the product is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentProd`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentMin
    *   The `unsortedSegmentMin` op computes the min along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \min_{j...} data(j...)` where the min is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentMin`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentMax
    *   The `unsortedSegmentMax` op computes the max along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \max_{j...} data(j...)` where the max is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentMax`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentSqrtN
    *   The `unsortedSegmentSqrtN` op computes the sum along segments of a tensor, divided by the square root of
    *   number of elements being summed.
    *
    *   The op computes a tensor such that `output(i) = \frac{\sum_{j...} data(j...)}{\sqrt{N}}` where the sum is
    *   over all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSparseSegmentSum
    *   The `sparseSegmentSum` op computes the sum along sparse segments of a tensor.
    *
    *   The op is similar to that of [[segmentSum]], with the difference that `segmentIndices` can have rank less
    *   than `data`'s first dimension, selecting a subset of dimension `0`, specified by `indices`. `segmentIndices` is
    *   allowed to have missing indices, in which case the output will be zeros at those indices. In those cases,
    *   `numSegments` is used to determine the size of the output.
    *
    *   For example:
    *   {{{
    *     // 'c' is [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]
    *
    *     // Select two rows, one segment.
    *     sparseSegmentSum(c, Tensor(0, 1), Tensor(0, 0)) ==> [[0, 0, 0, 0]]
    *
    *     // Select two rows, two segments.
    *     sparseSegmentSum(c, Tensor(0, 1), Tensor(0, 1)) ==> [[1, 2, 3, 4], [-1, -2, -3, -4]]
    *
    *     // Select all rows, two segments.
    *     sparseSegmentSum(c, Tensor(0, 1, 2), Tensor(0, 0, 1)) ==> [[0, 0, 0, 0], [5, 6, 7, 8]]
    *     // which is equivalent to:
    *     segmentSum(c, Tensor(0, 0, 1))
    *   }}}
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSparseSegmentMean
    *   The `sparseSegmentMean` op computes the mean along sparse segments of a tensor.
    *
    *   The op is similar to that of [[segmentMean]], with the difference that `segmentIndices` can have rank less
    *   than `data`'s first dimension, selecting a subset of dimension `0`, specified by `indices`. `segmentIndices` is
    *   allowed to have missing indices, in which case the output will be zeros at those indices. In those cases,
    *   `numSegments` is used to determine the size of the output.
    *
    *   For example:
    *   {{{
    *     // 'c' is [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]
    *
    *     // Select two rows, one segment.
    *     sparseSegmentMean(c, Tensor(0, 1), Tensor(0, 0)) ==> [[0, 0, 0, 0]]
    *
    *     // Select two rows, two segments.
    *     sparseSegmentMean(c, Tensor(0, 1), Tensor(0, 1)) ==> [[1, 2, 3, 4], [-1, -2, -3, -4]]
    *
    *     // Select all rows, two segments.
    *     sparseSegmentMean(c, Tensor(0, 1, 2), Tensor(0, 0, 1)) ==> [[0, 0, 0, 0], [5, 6, 7, 8]]
    *     // which is equivalent to:
    *     segmentMean(c, Tensor(0, 0, 1))
    *   }}}
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathSparseSegmentSumSqrtN
    *   The `sparseSegmentSumSqrtN` op computes the sum along sparse segments of a tensor, divided by the square
    *   root of the number of elements being summed. `segmentIndices` is allowed to have missing indices, in which case
    *   the output will be zeros at those indices. In those cases, `numSegments` is used to determine the size of the
    *   output.
    *
    *   Similar to [[sparseSegmentSum]].
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathDiag
    *   The `diag` op constructs a diagonal tensor using the provided diagonal values.
    *
    *   Given a `diagonal`, the op returns a tensor with that `diagonal` and everything else padded with zeros. The
    *   diagonal is computed as follows:
    *
    *   Assume that `diagonal` has shape `[D1,..., DK]`. Then the output tensor, `output`, is a rank-`2K` tensor with
    *   shape `[D1, ..., DK, D1, ..., DK]`, where `output(i1, ..., iK, i1, ..., iK) = diagonal(i1, ..., iK)` and `0`
    *   everywhere else.
    *
    *   For example:
    *   {{{
    *     // 'diagonal' is [1, 2, 3, 4]
    *     diag(diagonal) ==> [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    *   }}}
    *
    *   This op is the inverse of [[diagPart]].
    *
    * @define OpDocMathDiagPart
    *   The `diagPart` op returns the diagonal part of a tensor.
    *
    *   The op returns a tensor with the `diagonal` part of the `input`. The `diagonal` part is computed as follows:
    *
    *   Assume `input` has shape `[D1, ..., DK, D1, ..., DK]`. Then the output is a rank-`K` tensor with shape
    *   `[D1,..., DK]`, where `diagonal(i1, ..., iK) = output(i1, ..., iK, i1, ..., iK)`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    *     diagPart(input) ==> [1, 2, 3, 4]
    *   }}}
    *
    *   This op is the inverse of [[diag]].
    *
    * @define OpDocMathMatrixDiag
    *   The `matrixDiag` op returns a batched diagonal tensor with the provided batched diagonal values.
    *
    *   Given a `diagonal`, the op returns a tensor with that `diagonal` and everything else padded with zeros. Assuming
    *   that `diagonal` has `k` dimensions `[I, J, K, ..., N]`, the output is a tensor of rank `k + 1` with dimensions
    *   `[I, J, K, ..., N, N]`, where: `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
    *
    *   For example:
    *   {{{
    *     // 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]] (shape = [2, 4])
    *     matrixDiag(diagonal) ==> [[[1, 0, 0, 0]
    *                                [0, 2, 0, 0]
    *                                [0, 0, 3, 0]
    *                                [0, 0, 0, 4]],
    *                               [[5, 0, 0, 0]
    *                                [0, 6, 0, 0]
    *                                [0, 0, 7, 0]
    *                                [0, 0, 0, 8]]]  // with shape [2, 4, 4]
    *   }}}
    *
    * @define OpDocMathMatrixSetDiag
    *   The `matrixSetDiag` op returns a batched matrix tensor with new batched diagonal values.
    *
    *   Given `input` and `diagonal`, the op returns a tensor with the same shape and values as `input`, except for the
    *   main diagonal of its innermost matrices. These diagonals will be overwritten by the values in `diagonal`.
    *   Assuming that `input` has `k + 1` dimensions, `[I, J, K, ..., M, N]`, and `diagonal` has `k` dimensions,
    *   `[I, J, K, ..., min(M, N)]`, then the output is a tensor of rank `k + 1` with dimensions `[I, J, K, ..., M, N]`,
    *   where:
    *     - `output[i, j, k, ..., m, n] == diagonal[i, j, k, ..., n]`, for `m == n`, and
    *     - `output[i, j, k, ..., m, n] == input[i, j, k, ..., m, n]`, for `m != n`.
    *
    * @define OpDocMathMatrixDiagPart
    *   The `matrixDiagPart` op returns the batched diagonal part of a batched tensor.
    *
    *   The op returns a tensor with the `diagonal` part of the batched `input`. Assuming that `input` has `k`
    *   dimensions, `[I, J, K, ..., M, N]`, then the output is a tensor of rank `k - 1` with dimensions
    *   `[I, J, K, ..., min(M, N)]`, where `diagonal[i, j, k, ..., n] == input[i, j, k, ..., n, n]`.
    *
    *   Note that `input` must have rank of at least `2`.
    *
    *   For example:
    *   {{{
    *     // 'input' is:
    *     //   [[[1, 0, 0, 0]
    *     //     [0, 2, 0, 0]
    *     //     [0, 0, 3, 0]
    *     //     [0, 0, 0, 4]],
    *     //    [[5, 0, 0, 0]
    *     //     [0, 6, 0, 0]
    *     //     [0, 0, 7, 0]
    *     //     [0, 0, 0, 8]]]  with shape [2, 4, 4]
    *     matrixDiagPart(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]  // with shape [2, 4]
    *   }}}
    *
    * @define OpDocMathMatrixBandPart
    *   The `matrixBandPart` op copies a tensor, while setting everything outside a central band in each innermost
    *   matrix of the tensor, to zero.
    *
    *   Assuming that `input` has `k` dimensions, `[I, J, K, ..., M, N]`, the output is a tensor with the same shape,
    *   where `band[i, j, k, ..., m, n] == indicatorBand(m, n) * input[i, j, k, ..., m, n]`. The indicator function is
    *   defined as:
    *   {{{
    *     indicatorBand(m, n) = (numSubDiagonals < 0 || m - n <= numSubDiagonals) &&
    *                           (numSuperDiagonals < 0 || n - m <= numSuperDiagonals)
    *   }}}
    *
    *   For example:
    *   {{{
    *     // 'input' is:
    *     //   [[ 0,  1,  2, 3]
    *     //    [-1,  0,  1, 2]
    *     //    [-2, -1,  0, 1]
    *     //    [-3, -2, -1, 0]]
    *     matrixBandPart(input, 1, -1) ==> [[ 0,  1,  2, 3]
    *                                       [-1,  0,  1, 2]
    *                                       [ 0, -1,  0, 1]
    *                                       [ 0,  0, -1, 0]]
    *     matrixBandPart(input, 2, 1) ==>  [[ 0,  1,  0, 0]
    *                                       [-1,  0,  1, 0]
    *                                       [-2, -1,  0, 1]
    *                                       [ 0, -2, -1, 0]]
    *   }}}
    *
    *   Useful special cases:
    *   {{{
    *     matrixBandPart(input, 0, -1) ==> Upper triangular part
    *     matrixBandPart(input, -1, 0) ==> Lower triangular part
    *     matrixBandPart(input, 0, 0)  ==> Diagonal
    *   }}}
    *
    * @define OpDocMathTrace
    *   The `trace` op computes the trace of a tensor.
    *
    *   The trace of a tensor is defined as the sum along the main diagonal of each inner-most matrix in it.
    *   If the tensor is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output is a tensor of rank
    *   `k - 2` with dimensions `[I, J, K, ..., L]` where:
    *   `output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 2], [3, 4]]
    *     trace(x) ==> 5
    *
    *     // 'x' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    *     trace(x) ==> 15
    *
    *     // 'x' is [[[ 1,  2,  3],
    *     //          [ 4,  5,  6],
    *     //          [ 7,  8,  9]],
    *     //         [[-1, -2, -3],
    *     //          [-4, -5, -6],
    *     //          [-7, -8, -9]]]
    *     trace(x) ==> [15, -15]
    *   }}}
    *
    * @define OpDocMathScalarMul
    *   The `scalarMul` op multiplies a scalar tensor with another, potentially sparse, tensor.
    *
    *   This function is intended for use in gradient code which might deal with [[OutputIndexedSlices]] objects,
    *   which are easy to multiply by a scalar but more expensive to multiply with arbitrary tensors.
    *
    * @define OpDocMathMatmul
    *   The `matmul` op multiples two matrices.
    *
    *   The inputs must, following any transpositions, be tensors of rank >= 2, where the inner 2 dimensions specify
    *   valid matrix multiplication arguments and any further outer dimensions match.
    *
    *   Note that this op corresponds to a matrix product and not an element-wise product. For example:
    *   `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`, for all indices `i` and `j`.
    *
    *   Either matrix can be transposed and/or conjugated on the fly by setting one of the corresponding flags to
    *   `true`. These are set to `false` by default.
    *
    *   If one or both of the matrices contain a lot of zeros, a more efficient multiplication algorithm can be used
    *   by setting the corresponding `aIsSparse` or `bIsSparse` flag to `true`. These are also set to `false` by
    *   default. This optimization is only available for plain matrices (i.e., rank-2 tensors) with data type
    *   `BFLOAT16` or `FLOAT32`. The break-even for using this versus a dense matrix multiply on one platform was
    *   30% zero values in the sparse matrix. The gradient computation of the sparse op will only take advantage of
    *   sparsity in the input gradient when that gradient comes from a ReLU.
    *
    *   For example:
    *   {{{
    *     // 2-D tensor 'a' is [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    *
    *     // 2-D tensor 'b' is [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    *
    *     matmul(a, b) ==> [[58.0, 64.0], [139.0, 154.0]]
    *
    *     // 3-D tensor 'a' is [[[ 1.0,  2.0,  3.0],
    *     //                     [ 4.0,  5.0,  6.0]],
    *     //                    [[ 7.0,  8.0,  9.0],
    *     //                     [10.0, 11.0, 12.0]]]
    *
    *     // 3-D tensor 'b' is [[[13.0, 14.0],
    *     //                     [15.0, 16.0],
    *     //                     [17.0, 18.0]],
    *     //                    [[19.0, 20.0],
    *     //                     [21.0, 22.0],
    *     //                     [23.0, 24.0]]]
    *
    *     matmul(a, b) ==> [[[ 94.0, 100.0], [229.0, 244.0]],
    *                       [[508.0, 532.0], [697.0, 730.0]]]
    *   }}}
    *
    * @define OpDocMathCross
    *   The `cross` op computes the pairwise cross product between two tensors.
    *
    *   `a` and `b` must have the same shape; they can either be simple 3-element vectors, or have any shape
    *   where the innermost dimension size is 3. In the latter case, each pair of corresponding 3-element vectors
    *   is cross-multiplied independently.
    *
    * @define OpDocMathTensorDot
    *   The `tensorDot` op computes the tensor contraction of two tensors along the specified axes.
    *
    *   A tensor contraction sums the product of elements from `a` and `b` over the indices specified by `axesA` and
    *   `axesB`. The axis `axesA(i)` of `a` must have the same dimension as the axis `axesB(i)` of `b` for all `i` in
    *   `[0, aAxes.size)`. The tensors/sequences (depending on whether the dynamic version of the op is being used)
    *   `axesA` and `axesB` must have identical length and consist of unique integers that specify valid axes for each
    *   of the tensors. This operation corresponds to `numpy.tensordot(a, b, axes)` in Python.
    *
    *   If `numAxes` is provided instead of `axesA` and `axesB`, then the contraction is performed over the last
    *   `numAxes` axes of `a` and the first `numAxes` axes of `b`, in order.
    *
    *   Example 1: When `a` and `b` are matrices (rank 2), the case `numAxes = 1` is equivalent to matrix
    *              multiplication.
    *   Example 2: When `a` and `b` are matrices (rank 2), the case `axesA = [1]` and `axesB = [0]` is equivalent to
    *              matrix multiplication.
    *   Example 3: Suppose that `a_{ijk}` and `b_{lmn}` represent two tensors of rank 3. Then, the case `axesA = [0]`
    *              and `axesB = [2]` results in the rank 4 tensor `c_{jklm}` whose entry corresponding to the indices
    *              `(j, k, l, m)` is given by: `c_{jklm} = \sum_i a_{ijk} b_{lmi}`. In general,
    *              `rank(result) = rank(a) + rank(b) - 2 * axesA.size`.
    *
    * @define OpDocMathComplex
    *   The `complex` op converts two real tensors to a complex tensor.
    *
    *   Given a tensor `real` representing the real part of a complex number, and a tensor `imag` representing the
    *   imaginary part of a complex number, the op returns complex numbers element-wise of the form `a + bj`, where *a*
    *   represents the `real` part and *b* represents the `imag` part. The input tensors `real` and `imag` must have the
    *   same shape and data type.
    *
    *   For example:
    *   {{{
    *     // 'real' is [2.25, 3.25]
    *     // 'imag' is [4.75, 5.75]
    *     complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
    *   }}}
    *
    * @define OpDocMathReal
    *   The `real` op returns the real part of a complex number.
    *
    *   Given a tensor `input` of potentially complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64`
    *   that is the real part of each element in `input`. If `input` contains complex numbers of the form `a + bj`,
    *   *a* is the real part returned by the op and *b* is the imaginary part.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     real(input) ==> [-2.25, 3.25]
    *   }}}
    *
    *   Note that, if `input` is already real-valued, then it is returned unchanged.
    *
    * @define OpDocMathImag
    *   The `imag` op returns the real part of a complex number.
    *
    *   Given a tensor `input` of complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that is the
    *   imaginary part of each element in `input`. If `input` contains complex numbers of the form `a + bj`, *a* is the
    *   real part and *b* is the imaginary part returned by the op.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     real(input) ==> [4.75, 5.75]
    *   }}}
    *
    * @define OpDocMathAngle
    *   The `angle` op returns the element-wise complex argument of a tensor.
    *
    *   Given a numeric tensor `input`, the op returns a tensor with numbers that are the complex angle of each element
    *   in `input`. If the numbers in `input` are of the form `a + bj`, where *a* is the real part and *b* is the
    *   imaginary part, then the complex angle returned by this operation is of the form `atan2(b, a)`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     angle(input) ==> [2.0132, 1.056]
    *   }}}
    *
    *   If `input` is real-valued, then a tensor containing zeros is returned.
    *
    * @define OpDocMathConjugate
    *   The `conjugate` op returns the element-wise complex conjugate of a tensor.
    *
    *   Given a numeric tensor `input`, the op returns a tensor with numbers that are the complex conjugate of each
    *   element in `input`. If the numbers in `input` are of the form `a + bj`, where *a* is the real part and *b* is
    *   the imaginary part, then the complex conjugate returned by this operation is of the form `a - bj`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     conjugate(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
    *   }}}
    *
    *   If `input` is real-valued, then it is returned unchanged.
    *
    * @define OpDocMathBucketize
    *   The `bucketize` op bucketizes a tensor based on the provided boundaries.
    *
    *   For example:
    *   {{{
    *     // 'input' is [[-5, 10000], [150, 10], [5, 100]]
    *     // 'boundaries' are [0, 10, 100]
    *     bucketize(input, boundaries) ==> [[0, 3], [3, 2], [1, 3]]
    *   }}}
    *
    * @define OpDocMathZerosFraction
    *   The `zerosFraction` op computes the fraction of zeros in `input`.
    *
    *   If `input` is empty, the result is `NaN`.
    *
    *   This is useful in summaries to measure and report sparsity.
    */
  private[ops] trait Documentation
}
