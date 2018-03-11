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

import org.scalatest.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class CRC32CSuite extends JUnitSuite {
  @Test def testValue(): Unit = {
    // From rfc3720 section B.4.
    assert(CRC32C.value(Array.fill[Byte](32)(0x00.toByte)) === 0x8a9136aa)
    assert(CRC32C.value(Array.fill[Byte](32)(0xff.toByte)) === 0x62a8ab43)
    assert(CRC32C.value((0 until 32).map(_.toByte).toArray) === 0x46dd794e)
    assert(CRC32C.value((31 to 0 by -1).map(_.toByte).toArray) === 0x113fdb5c)

    val bytes = Array(
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ).map(_.toByte)
    assert(CRC32C.value(bytes) === 0xd9963a56)

    assert(CRC32C.value("a") !== CRC32C.value("foo"))
  }

  @Test def testExtend(): Unit = {
    assert(CRC32C.value("hello world") === CRC32C.extend(CRC32C.value("hello "), "world"))
  }

  @Test def testMask(): Unit = {
    val crc = CRC32C.value("foo")
    assert(crc !== CRC32C.mask(crc))
    assert(crc !== CRC32C.mask(CRC32C.mask(crc)))
    assert(crc === CRC32C.unmask(CRC32C.mask(crc)))
    assert(crc === CRC32C.unmask(CRC32C.unmask(CRC32C.mask(CRC32C.mask(crc)))))
  }
}
