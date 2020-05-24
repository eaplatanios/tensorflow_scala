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

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.math.Math
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

trait MathImplicits {
  implicit def outputConvertibleToMathOps[T, OC](
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
    def logGamma(implicit ev: IsFloatOrDouble[T]): Output[T] = {
      Math.logGamma(output)
    }

    /** $OpDocMathDigamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def digamma(implicit ev: IsFloatOrDouble[T]): Output[T] = {
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
    def roundInt(implicit ev: IsHalfOrFloatOrDouble[T]): Output[T] = {
      Math.roundInt(output)
    }

    /** $OpDocMathFloor
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def floor(implicit ev: IsHalfOrFloatOrDouble[T]): Output[T] = {
      Math.floor(output)
    }

    /** $OpDocMathCeil
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ceil(implicit ev: IsHalfOrFloatOrDouble[T]): Output[T] = {
      Math.ceil(output)
    }

    /** $OpDocMathIsNaN
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isNaN(implicit ev: IsHalfOrFloatOrDouble[T]): Output[Boolean] = {
      Math.isNaN(output)
    }

    /** $OpDocMathIsInf
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isInf(implicit ev: IsHalfOrFloatOrDouble[T]): Output[Boolean] = {
      Math.isInf(output)
    }

    /** $OpDocMathIsFinite
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isFinite(implicit ev: IsHalfOrFloatOrDouble[T]): Output[Boolean] = {
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
    def igammac(other: Output[T])(implicit ev: IsFloatOrDouble[T]): Output[T] = {
      Math.igammac(output, other)
    }

    /** $OpDocMathIgamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def igamma(other: Output[T])(implicit ev: IsFloatOrDouble[T]): Output[T] = {
      Math.igamma(output, other)
    }

    /** $OpDocMathZeta
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def zeta(other: Output[T])(implicit ev: IsFloatOrDouble[T]): Output[T] = {
      Math.zeta(output, other)
    }

    /** $OpDocMathPolygamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def polygamma(other: Output[T])(implicit ev: IsFloatOrDouble[T]): Output[T] = {
      Math.polygamma(output, other)
    }

    /** $OpDocMathAtan2
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atan2(other: Output[T])(implicit ev: IsFloatOrDouble[T]): Output[T] = {
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
    def sum[I: IntDefault : TF : IsIntOrLong](
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
    def mean[I: IntDefault : TF : IsIntOrLong](
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
    def prod[I: IntDefault : TF : IsIntOrLong](
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
    def min[I: IntDefault : TF : IsIntOrLong](
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
    def max[I: IntDefault : TF : IsIntOrLong](
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
    def all[I: IntDefault : TF : IsIntOrLong](
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
    def any[I: IntDefault : TF : IsIntOrLong](
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
    def logSumExp[I: IntDefault : TF : IsIntOrLong](
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
    def countNonZero[I: IntDefault : TF : IsIntOrLong](
        axes: Output[I] = null,
        keepDims: Boolean = false
    )(implicit ev: IsNumeric[T]): Output[Long] = {
      Math.countNonZero(output, axes, keepDims)
    }

    /** $OpDocMathArgmin
      *
      * @group MathOps
      * @param  axes Integer tensor containing the axes to reduce.
      * @return Result as a new tensor.
      */
    def argmin[I: TF : IsIntOrLong](
        axes: Output[I]
    )(implicit ev: IsNotQuantized[T]): Output[Long] = {
      Math.argmin(output, axes, outputDataType = INT64)
    }

    /** $OpDocMathArgmin
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce.
      * @param  outputDataType Data type for the output tensor.
      * @return Result as a new tensor.
      */
    def argmin[I: TF : IsIntOrLong, IR: TF : IsIntOrLong](
        axes: Output[I],
        outputDataType: DataType[IR]
    )(implicit ev: IsNotQuantized[T]): Output[IR] = {
      Math.argmin(output, axes, outputDataType)
    }

    /** $OpDocMathArgmax
      *
      * @group MathOps
      * @param  axes Integer tensor containing the axes to reduce.
      * @return Result as a new tensor.
      */
    def argmax[I: TF : IsIntOrLong](
        axes: Output[I]
    )(implicit ev: IsNotQuantized[T]): Output[Long] = {
      Math.argmax(output, axes, outputDataType = INT64)
    }

    /** $OpDocMathArgmax
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce.
      * @param  outputDataType Data type for the output tensor.
      * @return Result as a new tensor.
      */
    def argmax[I: TF : IsIntOrLong, IR: TF : IsIntOrLong](
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
    def cumsum[I: TF : IsIntOrLong](
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
    def cumprod[I: TF : IsIntOrLong](
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
    def binCount[R: TF : IsIntOrLongOrFloatOrDouble](
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
    def segmentSum[I: TF : IsIntOrLong](
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
    def segmentMean[I: TF : IsIntOrLong](
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
    def segmentProd[I: TF : IsIntOrLong](
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
    def segmentMin[I: TF : IsIntOrLong](
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
    def segmentMax[I: TF : IsIntOrLong](
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
    def unsortedSegmentSum[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
    def unsortedSegmentMean[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
    def unsortedSegmentProd[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
    def unsortedSegmentMin[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
    def unsortedSegmentMax[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
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
    def sparseSegmentSum[I1: TF : IsIntOrLong, I2: IntDefault : TF : IsIntOrLong](
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
    def sparseSegmentMean[I1: TF : IsIntOrLong, I2: IntDefault : TF : IsIntOrLong](
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
    def sparseSegmentSumSqrtN[I1: TF : IsIntOrLong, I2: IntDefault : TF : IsIntOrLong](
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
    def matrixBandPart[I: TF : IsIntOrLong](
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
    )(implicit ev: IsIntOrLongOrFloatOrDouble[T]): Output[T] = {
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
    def toComplex(imag: Output[Float] = Tensor.zeros[Float](Shape()).toOutput): Output[ComplexFloat] = {
      Math.complexFloat(output.asInstanceOf[Output[Float]], imag)
    }
  }

  implicit class DoubleMathOps(val output: Output[Double]) {
    /** Creates a new complex number with the provided imaginary part.
      *
      * @param  imag Imaginary part.
      * @return Resulting complex number.
      */
    def toComplex(imag: Output[Double] = Tensor.zeros[Double](Shape()).toOutput): Output[ComplexDouble] = {
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
