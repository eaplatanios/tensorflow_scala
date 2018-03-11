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

package org.platanios.tensorflow.api.utilities

/** Contains helper functions for manipulating collections.
  *
  * @author Emmanouil Antonios Platanios
  */
object Collections {
  /** Segments a sequence according to a provided sequence of segment lengths.
    *
    * For example.
    * {{{
    *   val xs = Seq(3, 5, 2, 77, 12, 45, 78, 21, 89, 1, 0, -1, 123)
    *   val n = Seq(3, 1, 0, 2, 5, 2)
    *   segment(xs, n) = Seq(Seq(3, 5, 2), Seq(77), Seq(), Seq(12, 45), Seq(78, 21, 89, 1, 0), Seq(-1, 123))
    * }}}
    *
    * Note that the function returns when either one of `xs` or `n` is exhausted. This means that no exception is thrown
    * if the provided segment lengths do not match the original sequence length.
    *
    * @param  xs Sequence to segment.
    * @param  n  Segment lengths.
    * @return Sequence containing the segments of `xs`.
    */
  def segment[V](xs: Seq[V], n: Seq[Int]): Seq[Seq[V]] = {
    if (xs.isEmpty && n.isEmpty) {
      Nil
    } else {
      val (ys, zs) = xs.splitAt(n.head)
      ys +: segment(zs, n.tail)
    }
  }
}
