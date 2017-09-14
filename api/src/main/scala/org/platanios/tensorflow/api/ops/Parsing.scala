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

import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, INT32, STRING, UINT8}

/** Contains functions for constructing ops related to parsing data.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Parsing {
  /** $OpDocParsingDecodeRaw
    *
    * @group ParsingOps
    * @param  bytes        [[STRING]] tensor interpreted as raw bytes. All the elements must have the same length.
    * @param  dataType     Output tensor data type.
    * @param  littleEndian Boolean value indicating whether the input `bytes` are stored in little-endian order. Ignored
    *                      for `dataType` values that are stored in a single byte, like [[UINT8]].
    * @param  name         Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `bytes` is not a [[STRING]] tensor.
    */
  @throws[IllegalArgumentException]
  def decodeRaw(bytes: Output, dataType: DataType, littleEndian: Boolean = true, name: String = "DecodeRaw"): Output = {
    require(bytes.dataType == STRING, s"Tensor data type was ${bytes.dataType}, while STRING was expected.")
    Op.Builder(opType = "DecodeRaw", name = name)
        .addInput(bytes)
        .setAttribute("out_type", dataType)
        .setAttribute("little_endian", littleEndian)
        .build().outputs(0)
  }

  /** $OpDocParsingDecodeCSV
    *
    * @group ParsingOps
    * @param  records            [[STRING]] tensor where each string is a record/row in the csv and all records should
    *                            have the same format.
    * @param  recordDefaults     One tensor per column of the input record, with either a scalar default value for that
    *                            column or empty if the column is required.
    * @param  dataTypes          Output tensor data types.
    * @param  delimiter          Delimiter used to separate fields in a record.
    * @param  useQuoteDelimiters If `false`, the op treats double quotation marks as regular characters inside the
    *                            string fields (ignoring RFC 4180, Section 2, Bullet 5).
    * @param  name               Name for the created op.
    * @return Created op outputs.
    * @throws IllegalArgumentException If `records` is not a [[STRING]] tensor.
    */
  @throws[IllegalArgumentException]
  def decodeCSV(
      records: Output, recordDefaults: Seq[Output], dataTypes: Seq[DataType], delimiter: String = ",",
      useQuoteDelimiters: Boolean = true, name: String = "DecodeCSV"): Seq[Output] = {
    require(records.dataType == STRING, s"Tensor data type was ${records.dataType}, while STRING was expected.")
    Op.Builder(opType = "DecodeCSV", name = name)
        .addInput(records)
        .addInputList(recordDefaults)
        .setAttribute("OUT_TYPE", dataTypes.toArray)
        .setAttribute("field_delim", delimiter)
        .setAttribute("use_quote_delim", useQuoteDelimiters)
        .build().outputs.toSeq
  }

  /** $OpDocParsingStringToNumber
    *
    * @group ParsingOps
    * @param  data     [[STRING]] tensor containing string representations of numbers.
    * @param  dataType Output tensor data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `data` is not a [[STRING]] tensor.
    */
  @throws[IllegalArgumentException]
  def stringToNumber(data: Output, dataType: DataType, name: String = "StringToNumber"): Output = {
    require(data.dataType == STRING, s"Tensor data type was ${data.dataType}, while STRING was expected.")
    Op.Builder(opType = "StringToNumber", name = name)
        .addInput(data)
        .setAttribute("out_type", dataType)
        .build().outputs(0)
  }
}

private[ops] object Parsing extends Parsing {
  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("DecodeRaw")
    GradientsRegistry.registerNonDifferentiable("DecodeCSV")
    GradientsRegistry.registerNonDifferentiable("StringToNumber")
  }

  /** @define OpDocParsingDecodeRaw
    *   The `decodeRaw` op reinterprets the bytes of a string as a vector of numbers.
    *
    * @define OpDocParsingDecodeCSV
    *   The `decodeCSV` op converts CSV records to tensors. Each column maps to one tensor.
    *
    *   The [RFC 4180](https://tools.ietf.org/html/rfc4180) format is expected for the CSV records. Note that we allow
    *   leading and trailing spaces with integer or floating-point fields.
    *
    * @define OpDocParsingStringToNumber
    *   The `stringToNumber` op converts each string in the input tensor to the specified numeric type,
    *
    *   '''NOTE:''' [[INT32]] overflow results in an error while [[FLOAT32]] overflow results in a rounded value.
    */
  private[ops] trait Documentation
}
