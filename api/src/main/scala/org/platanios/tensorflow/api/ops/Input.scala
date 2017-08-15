/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.using
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.jni.{Op => NativeOp}

/** Wrapper around an [[Op]] meant to represent one of its inputs. Actual op inputs have type [[Output]] since they
  * represent outputs of other ops. Currently, [[Input]] is only useful for representing consumers of an [[Op]]'s
  * outputs.
  *
  * @param  op    Op whose input this class represents.
  * @param  index Input index.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class Input private[ops](op: Op, index: Int) {
  /** Name of this op input. This is simply set to `"<op.name>:<index>"`. */
  lazy val name: String = s"${op.name}:$index"

  /** Data type of this op input. */
  lazy val dataType: DataType = using(graph.reference) { r =>
    DataType.fromCValue(NativeOp.inputDataType(r.nativeHandle, op.nativeHandle, index))
  }

  /** Graph where the op belongs. */
  def graph: Graph = op.graph

  override def toString: String = s"Op.Input(name = $name, dataType = $dataType)"
}
