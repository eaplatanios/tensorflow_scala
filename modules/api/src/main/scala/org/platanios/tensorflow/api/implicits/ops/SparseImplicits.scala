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
import org.platanios.tensorflow.api.core.types.{IsNumeric, IsReal, TF}
import org.platanios.tensorflow.api.ops.{Output, Sparse, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor

trait SparseImplicits {
  implicit def outputConvertibleToSparseOps[T, OC](
      value: OC
  )(implicit f: OC => SparseOutput[T]): SparseOps[T] = {
    new SparseOps(f(value))
  }

  implicit class SparseOps[T](val sparseOutput: SparseOutput[T]) {
    protected implicit val evTTF: TF[T] = {
      TF.fromDataType(sparseOutput.dataType)
    }

    def +(other: SparseOutput[T])(implicit ev: IsNumeric[T]): SparseOutput[T] = {
      add(other, threshold = Tensor.zeros[Int](Shape()).toOutput)
    }

    def +(other: Output[T])(implicit ev: IsNumeric[T]): Output[T] = {
      addDense(other)
    }

    /** $OpDocSparseSparseAdd
      *
      * @group SparseOps
      * @param  other     Tensor to add to the current one.
      * @param  threshold Sparsity threshold.
      * @param  name      Name for the created op.
      * @return Created op output.
      */
    def add[TR: TF : IsReal](
        other: SparseOutput[T],
        threshold: Output[TR],
        name: String = "SparseAdd"
    )(implicit ev: IsNumeric[T]): SparseOutput[T] = {
      Sparse.add(sparseOutput, other, threshold, name)
    }

    /** $OpDocSparseSparseDenseAdd
      *
      * @group SparseOps
      * @param  other Dense tensor to add to the current one.
      * @param  name  Name for the created op.
      * @return Created op output.
      */
    def addDense(
        other: Output[T],
        name: String = "SparseDenseAdd"
    )(implicit ev: IsNumeric[T]): Output[T] = {
      Sparse.denseAdd(sparseOutput, other, name)
    }
  }
}
