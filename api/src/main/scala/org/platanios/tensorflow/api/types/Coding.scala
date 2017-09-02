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

package org.platanios.tensorflow.api.types

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Coding {
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
