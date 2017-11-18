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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.io.{CompressionType, NoCompression}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, STRING}

/** Dataset with elements read from text files (each line in each file corresponds to an element).
  *
  * **Note:** New-line characters are stripped from the output.
  *
  * @param  filenames       [[STRING]] scalar or vector tensor containing the the name(s) of the file(s) to be read.
  * @param  compressionType Compression type for the file.
  * @param  bufferSize      Number of bytes to buffer while reading from the file.
  * @param  name            Name for this dataset.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TextLinesDataset(
    filenames: Tensor,
    compressionType: CompressionType = NoCompression,
    bufferSize: Long = 256 * 1024,
    override val name: String = "TextLineDataset"
) extends Dataset[Tensor, Output, DataType, Shape](name) {
  if (filenames.dataType != STRING)
    throw new IllegalArgumentException(s"'filenames' (dataType = ${filenames.dataType}) must be a STRING tensor.")
  if (filenames.rank != -1 && filenames.rank > 1)
    throw new IllegalArgumentException(s"'filenames' (rank = ${filenames.rank}) must be at most 1.")

  override def createHandle(): Output = {
    Op.Builder(opType = "TextLineDataset", name = name)
        .addInput(Op.createWithNameScope(name)(filenames))
        .addInput(Op.createWithNameScope(name)(compressionType.name))
        .addInput(Op.createWithNameScope(name)(bufferSize))
        .build().outputs(0)
  }

  override def outputDataTypes: DataType = STRING
  override def outputShapes: Shape = Shape.scalar()
}
