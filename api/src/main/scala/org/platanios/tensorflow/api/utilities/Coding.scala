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

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Coding {
  def encodeFixedInt16(value: Char, littleEndian: Boolean = true): Array[Byte] = {
    val result = Array.ofDim[Byte](2)
    if (littleEndian) {
      result(0) = (value & 0xff).toByte
      result(1) = ((value >> 8) & 0xff).toByte
    } else {
      result(0) = ((value >> 8) & 0xff).toByte
      result(1) = (value & 0xff).toByte
    }
    result
  }

  def encodeFixedInt32(value: Int, littleEndian: Boolean = true): Array[Byte] = {
    val result = Array.ofDim[Byte](4)
    if (littleEndian) {
      result(0) = (value & 0xff).toByte
      result(1) = ((value >> 8) & 0xff).toByte
      result(2) = ((value >> 16) & 0xff).toByte
      result(3) = ((value >> 24) & 0xff).toByte
    } else {
      result(0) = ((value >> 24) & 0xff).toByte
      result(1) = ((value >> 16) & 0xff).toByte
      result(2) = ((value >> 8) & 0xff).toByte
      result(3) = (value & 0xff).toByte
    }
    result
  }

  def encodeFixedInt64(value: Long, littleEndian: Boolean = true): Array[Byte] = {
    val result = Array.ofDim[Byte](8)
    if (littleEndian) {
      result(0) = (value & 0xffL).toByte
      result(1) = ((value >> 8) & 0xffL).toByte
      result(2) = ((value >> 16) & 0xffL).toByte
      result(3) = ((value >> 24) & 0xffL).toByte
      result(4) = ((value >> 32) & 0xffL).toByte
      result(5) = ((value >> 40) & 0xffL).toByte
      result(6) = ((value >> 48) & 0xffL).toByte
      result(7) = ((value >> 56) & 0xffL).toByte
    } else {
      result(0) = ((value >> 56) & 0xffL).toByte
      result(1) = ((value >> 48) & 0xffL).toByte
      result(2) = ((value >> 40) & 0xffL).toByte
      result(3) = ((value >> 32) & 0xffL).toByte
      result(4) = ((value >> 24) & 0xffL).toByte
      result(5) = ((value >> 16) & 0xffL).toByte
      result(6) = ((value >> 8) & 0xffL).toByte
      result(7) = (value & 0xffL).toByte
    }
    result
  }

  def decodeFixedInt16(bytes: Array[Byte], offset: Int = 0, littleEndian: Boolean = true): Char = {
    var result: Char = 0
    if (littleEndian) {
      result = ((result | bytes(offset)) & 0xff).toChar
      result = (result | ((bytes(offset + 1) & 0xff) << 8)).toChar
    } else {
      result = (result | ((bytes(offset) & 0xff) << 8)).toChar
      result = (result | (bytes(offset + 1) & 0xff)).toChar
    }
    result
  }

  def decodeFixedInt32(bytes: Array[Byte], offset: Int = 0, littleEndian: Boolean = true): Int = {
    var result: Int = 0
    if (littleEndian) {
      result |= bytes(offset) & 0xff
      result |= (bytes(offset + 1) & 0xff) << 8
      result |= (bytes(offset + 2) & 0xff) << 16
      result |= (bytes(offset + 3) & 0xff) << 24
    } else {
      result |= (bytes(offset) & 0xff) << 24
      result |= (bytes(offset + 1) & 0xff) << 16
      result |= (bytes(offset + 2) & 0xff) << 8
      result |= bytes(offset + 3 & 0xff)
    }
    result
  }

  def decodeFixedInt64(bytes: Array[Byte], offset: Int = 0, littleEndian: Boolean = true): Long = {
    var result: Long = 0
    if (littleEndian) {
      result |= bytes(offset).toInt
      result |= (bytes(offset + 1) & 0xff) << 8
      result |= (bytes(offset + 2) & 0xff) << 16
      result |= (bytes(offset + 3) & 0xff) << 24
      result |= (bytes(offset + 4) & 0xff) << 32
      result |= (bytes(offset + 5) & 0xff) << 40
      result |= (bytes(offset + 6) & 0xff) << 48
      result |= (bytes(offset + 7) & 0xff) << 56
    } else {
      result |= (bytes(offset) & 0xffL) << 56
      result |= (bytes(offset + 1) & 0xffL) << 48
      result |= (bytes(offset + 2) & 0xffL) << 40
      result |= (bytes(offset + 3) & 0xffL) << 32
      result |= (bytes(offset + 4) & 0xffL) << 24
      result |= (bytes(offset + 5) & 0xffL) << 16
      result |= (bytes(offset + 6) & 0xffL) << 8
      result |= bytes(offset + 7) & 0xffL
    }
    result
  }

  def encodeStrings(values: String*): Array[Byte] = {
    val bytes: mutable.ArrayBuffer[Byte] = mutable.ArrayBuffer.empty
    values.foreach(value => bytes.appendAll(encodeVarInt32(value.length)))
    values.foreach(value => bytes.appendAll(value.getBytes))
    bytes.toArray
  }

  def varIntLength(value: Int): Int = {
    // We are treating 'value' as an unsigned integer.
    val B: Int = 128
    var unsigned: Long = value & 0xffffffffL
    var length: Int = 1
    while (unsigned >= B) {
      unsigned >>= 7
      length += 1
    }
    length
  }

  def encodeVarInt32(value: Int): Array[Byte] = {
    // We are treating 'value' as an unsigned integer.
    val B: Int = 128
    val bytes: Array[Byte] = Array.ofDim(5)
    var unsigned: Long = value & 0xffffffffL
    var position: Int = 0
    while (unsigned >= B) {
      bytes(position) = ((unsigned & (B - 1)) | B).toByte
      unsigned >>= 7
      position += 1
    }
    bytes(position) = unsigned.toByte
    bytes.take(position + 1)
  }

  def decodeVarInt32(bytes: Array[Byte]): (Int, Int) = {
    val B: Int = 128
    var byte: Byte = bytes(0)
    var value: Int = byte.toInt
    var position: Int = 1
    var shift: Int = 7
    while ((byte & B) != 0 && shift <= 28 && position < bytes.length) {
      // More bytes are present.
      byte = bytes(position)
      value |= ((byte & (B - 1)) << shift)
      position += 1
      shift += 7
    }
    if ((byte & 128) != 0)
      (-1, -1)
    else
      (value, position)
  }
}
