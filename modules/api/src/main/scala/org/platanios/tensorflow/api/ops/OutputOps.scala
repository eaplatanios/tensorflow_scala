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

import org.platanios.tensorflow.api.types.TF

/** Type trait for defining functions operating on and returning op outputs.
  *
  * @author Emmanouil Antonios Platanios
  */
trait OutputOps[OL[A] <: OutputLike[A]] {
  type T

  /** Applies a unary function to the provided output and returns the result.
    *
    * @param  value Output-like object to apply the unary function on.
    * @param  fn    Unary function to apply.
    * @return Resulting output-like object that matches the type of `outputLike`.
    */
  @inline def applyUnary[R: TF](
      value: OL[T],
      fn: Output[T] => Output[R]
  ): OL[R]
}

/** Companion object that defines supported output ops implicit values. */
object OutputOps {
  type Aux[OL[A] <: OutputLike[A], TT] = OutputOps[OL] {
    type T = TT
  }

  implicit def outputOps[TT: TF]: OutputOps.Aux[Output, TT] = {
    new OutputOps[Output] {
      override type T = TT

      @inline override def applyUnary[R: TF](
          value: Output[T],
          fn: Output[T] => Output[R]
      ): Output[R] = {
        fn(value)
      }
    }
  }

  implicit def outputIndexedSlicesOps[TT: TF]: OutputOps.Aux[OutputIndexedSlices, TT] = {
    new OutputOps[OutputIndexedSlices] {
      override type T = TT

      @inline override def applyUnary[R: TF](
          value: OutputIndexedSlices[T],
          fn: Output[T] => Output[R]
      ): OutputIndexedSlices[R] = {
        value.copy(values = fn(value.values))
      }
    }
  }

  implicit def sparseOutputOps[TT: TF]: OutputOps.Aux[SparseOutput, TT] = {
    new OutputOps[SparseOutput] {
      override type T = TT

      @inline override def applyUnary[R: TF](
          value: SparseOutput[T],
          fn: Output[T] => Output[R]
      ): SparseOutput[R] = {
        value.copy(values = fn(value.values))
      }
    }
  }

  implicit def outputLikeOps[TT: TF]: OutputOps.Aux[OutputLike, TT] = {
    new OutputOps[OutputLike] {
      override type T = TT

      @inline override def applyUnary[R: TF](
          value: OutputLike[T],
          fn: Output[T] => Output[R]
      ): OutputLike[R] = {
        value match {
          case o: Output[T] => fn(o)
          case o: OutputIndexedSlices[T] => o.copy(values = fn(o.values))
          case o: SparseOutput[T] => o.copy(values = fn(o.values))
        }
      }
    }
  }
}
