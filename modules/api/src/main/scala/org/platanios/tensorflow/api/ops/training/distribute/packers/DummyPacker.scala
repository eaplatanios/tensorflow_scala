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

package org.platanios.tensorflow.api.ops.training.distribute.packers

import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.OutputLike

/** Represents a dummy packer that does not perform any tensor packing at all.
  *
  * @author Emmanouil Antonios Platanios
  */
object DummyPacker extends Packer[Unit] {
  /** Packs the provided values.
    *
    * @param  grouped Grouped values (per device).
    * @return Packed values, ready for reduction, along with information that is necessary for unpacking later on.
    * @throws InvalidArgumentException If the provided grouped values are inconsistent in any way.
    */
  @throws[InvalidArgumentException]
  def pack(grouped: Seq[Seq[OutputLike]]): (Seq[Seq[OutputLike]], Option[Unit]) = {
    (grouped, None)
  }

  /** Reverses the packing performed by `pack`, on the provided packed values.
    *
    * @param  packed          Packed values to unpack.
    * @param  packInformation Information from the packing process that is necessary for unpacking.
    * @return Unpacked `packed`.
    * @throws InvalidArgumentException If not pack information is provided, while it is actually necessary.
    */
  @throws[InvalidArgumentException]
  def unpack(packed: Seq[Seq[OutputLike]], packInformation: Option[Unit]): Seq[Seq[OutputLike]] = {
    packed
  }
}
