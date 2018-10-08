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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardOpenOption}

import scala.util.matching.Regex

/** Contains helpers for dealing with Numpy (i.e., `.npy`) files.
  *
  * @author Emmanouil Antonios Platanios
  */
object NPY {
  /** Regular expression used to parse Numpy data types. */
  protected val dtypeParser: Regex = """^[<=>]?(\w\d*)$""".r

  /** Represents an NPY file header. */
  case class Header[T: TF](
      description: String,
      fortranOrder: Boolean,
      shape: Shape
  ) {
    val dataType: DataType[T] = description match {
      case dtypeParser(t) => numpyDTypeToDataType(t).asInstanceOf[DataType[T]]
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

  /** Reads the tensor stored in the provided Numpy (i.e., `.npy`) file. */
  @throws[InvalidDataTypeException]
  @throws[IllegalArgumentException]
  def read[T: TF](file: Path): Tensor[T] = {
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
        Header[T](description, fortranOrder, Shape.fromSeq(shape))
      case _ => throw new Exception("wrong header")
    }

    // Read the data.
    byteBuffer.order(header.byteOrder)
    val numBytes = header.shape.numElements * header.dataType.byteSize.get
    if (header.fortranOrder)
      Tensor.fromBuffer[T](Shape.fromSeq(header.shape.asArray.reverse), numBytes, byteBuffer).transpose()
    else
      Tensor.fromBuffer[T](header.shape, numBytes, byteBuffer)
  }

  /** Writes the provided tensor to the provided file, using the Numpy (i.e., `.npy`) file format. Note that this method
    * will replace the file, if it already exists. */
  @throws[InvalidDataTypeException]
  def write[T: TF](tensor: Tensor[T], file: Path, fortranOrder: Boolean = false): Unit = {
    val description = ">" + dataTypeToNumpyDType(tensor.dataType)
    val header = Header[T](description, fortranOrder, tensor.shape).toString

    val resolvedHandle = tensor.resolve()
    val buffer = NativeTensor.buffer(resolvedHandle).order(ByteOrder.nativeOrder)
    val dataBytes = buffer.array()
    tensor.NativeHandleLock synchronized {
      if (resolvedHandle != 0)
        NativeTensor.delete(resolvedHandle)
    }

    val remaining = (header.length + 11) % 16
    val padLength = if (remaining > 0) 16 - remaining else 0
    val headerLength = header.length + padLength + 1
    val size = header.length + 11 + padLength + dataBytes.length

    val array = new Array[Byte](size)
    val byteBuffer = ByteBuffer.wrap(array)
    byteBuffer.put(Array(
      0x93.toByte,
      'N'.toByte,
      'U'.toByte,
      'M'.toByte,
      'P'.toByte,
      'Y'.toByte,
      1.toByte,
      0.toByte))
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
    byteBuffer.putShort(headerLength.toShort)
    byteBuffer.order(ByteOrder.BIG_ENDIAN)
    byteBuffer.put(header.getBytes)
    byteBuffer.put(Array.fill(padLength)(' '.toByte))
    byteBuffer.put('\n'.toByte)
    byteBuffer.put(dataBytes)

    val fileWriter = Files.newOutputStream(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    fileWriter.write(array)
    fileWriter.flush()
    fileWriter.close()
  }

  /** Returns the TensorFlow data type equivalent to the provided Numpy data type string. */
  @throws[InvalidDataTypeException]
  def numpyDTypeToDataType[T](dtype: String): DataType[T] = {
    val dataType = dtype match {
      case "b" => BOOLEAN
      case "f2" => FLOAT16
      case "f4" => FLOAT32
      case "f8" => FLOAT64
      // case "f16" => ??? // FLOAT128
      case "i1" => INT8
      case "i2" => INT16
      case "i4" => Int
      case "i8" => INT64
      case "u1" => UINT8
      case "u2" => UINT16
      case "u4" => UINT32
      case "u8" => UINT64
      case t => throw InvalidDataTypeException(s"Numpy data type '$t' cannot be converted to a TensorFlow data type.")
    }
    dataType.asInstanceOf[DataType[T]]
  }

  /** Returns the Numpy data type string equivalent to the provided TensorFlow data type. */
  @throws[InvalidDataTypeException]
  def dataTypeToNumpyDType[T](dataType: DataType[T]): String = dataType match {
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
