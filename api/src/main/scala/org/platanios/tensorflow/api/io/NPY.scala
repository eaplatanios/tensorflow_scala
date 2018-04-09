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

package org.platanios.tensorflow.api.io

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object Numpy {
  protected val dtypeParser: Regex = """^[<=>]?(\w\d*)$""".r

  case class Header(description: String, fortranOrder: Boolean, shape: Shape) {
    val dataType: DataType = description match {
      case dtypeParser(t) => numpyDTypeToDataType(t)
    }

    def byteOrder: ByteOrder = description.head match {
      case '>' => ByteOrder.BIG_ENDIAN
      case '<' => ByteOrder.LITTLE_ENDIAN
      case _ => ByteOrder.nativeOrder()
    }

    override def toString: String = {
      val fortran = if (fortranOrder) "True" else "False"
      val shapeStr = if (shape.numElements == 1L) shape(0) + "," else shape.asArray.mkString(", ")
      s"{'descr': '$description', 'fortran_order': $fortran, 'shape': ($shapeStr), }"
    }
  }

  @throws[IllegalArgumentException]
  def read(file: Path): Tensor = {
    val byteBuffer = ByteBuffer.wrap(Files.readAllBytes(file))

    // Check the first byte in the magic string.
    require(byteBuffer.get() == 0x93.toByte, s"Wrong magic string in Numpy file. File: $file.")

    // Check the rest of the magic string.
    val magicBytes = new Array[Byte](5)
    byteBuffer.get(magicBytes, 0, 5)
    val magicString = new String(magicBytes, StandardCharsets.US_ASCII)
    require(magicString == "NUMPY", s"Wrong magic string in Numpy file. File: $file.")

    // Read the version number (two unsigned bytes).
    val majorVersion = byteBuffer.get() & 0xFF
    val minorVersion = byteBuffer.get() & 0xFF
    require(majorVersion == 1 && minorVersion == 0, s"Only version 1.0 is supported for Numpy files. File: $file.")

    // Read the header length (little endian).
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
    val headerLength = byteBuffer.getShort()

    // Read the header.
    val headerBytes = new Array[Byte](headerLength)
    byteBuffer.get(headerBytes)
    val headerString = new String(headerBytes, StandardCharsets.US_ASCII)
    val headerPattern = """'descr': '([^']+)', 'fortran_order': (True|False), 'shape': \(([^)]+?),?\)""".r.unanchored
    val header = headerString match {
      case headerPattern(description, fortranOrderString, shapeString) =>
        val fortranOrder = fortranOrderString == "True"
        val shape = shapeString.trim.split(", ").map(_.toInt)
        Header(description, fortranOrder, Shape.fromSeq(shape))
      case _ => throw new Exception("wrong header")
    }

    // Read the data.
    byteBuffer.order(header.byteOrder)
    val numBytes = header.shape.numElements * header.dataType.byteSize
    if (header.fortranOrder)
      Tensor.fromBuffer(header.dataType, Shape.fromSeq(header.shape.asArray.reverse), numBytes, byteBuffer).transpose()
    else
      Tensor.fromBuffer(header.dataType, header.shape, numBytes, byteBuffer)
  }

  @throws[InvalidDataTypeException]
  def numpyDTypeToDataType(dtype: String): DataType = dtype match {
    case "b" => BOOLEAN
    case "f2" => FLOAT16
    case "f4" => FLOAT32
    case "f8" => FLOAT64
    // case "f16" => ??? // FLOAT128
    case "i1" => INT8
    case "i2" => INT16
    case "i4" => INT32
    case "i8" => INT64
    case "u1" => UINT8
    case "u2" => UINT16
    case "u4" => UINT32
    case "u8" => UINT64
    case t => throw InvalidDataTypeException(s"Numpy data type '$t' cannot be converted to a TensorFlow data type.")
  }

  @throws[InvalidDataTypeException]
  def dataTypeToNumpyDType(dataType: DataType): String = dataType match {
    case BOOLEAN => "b"
    case FLOAT16 => "f2"
    case FLOAT32 => "f4"
    case FLOAT64 => "f8"
    case INT8 => "i1"
    case INT16 => "i2"
    case INT32 => "i3"
    case INT64 => "i4"
    case UINT8 => "u1"
    case UINT16 => "u2"
    case UINT32 => "u3"
    case UINT64 => "u4"
    case t => throw InvalidDataTypeException(s"TensorFlow data type '$t' cannot be converted to a Numpy data type.")
  }
}
