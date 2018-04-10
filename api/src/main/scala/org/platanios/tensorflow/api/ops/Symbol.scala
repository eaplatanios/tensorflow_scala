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

/** Helper tagging trait used to group `Output`s, `OutputIndexedSlice`s, `SparseOutput`s, and `TensorArray`s under the
  * same type. It is useful for the while loop variable map function which is used in the `seq2seq` package.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Symbol
