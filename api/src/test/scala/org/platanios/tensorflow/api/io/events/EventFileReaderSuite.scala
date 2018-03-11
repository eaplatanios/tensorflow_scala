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

package org.platanios.tensorflow.api.io.events

import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite
import org.tensorflow.util.Event

import java.nio.file.{Files, Path, StandardOpenOption}

/**
  * @author Emmanouil Antonios Platanios
  */
class EventFileReaderSuite extends JUnitSuite {
  private[this] val record: Array[Byte] = Array(
    0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xa3, 0x7f, 0x4b, 0x22, 0x09, 0x00, 0x00, 0xc0,
    0x25, 0xdd, 0x75, 0xd5, 0x41, 0x1a, 0x0d, 0x62,
    0x72, 0x61, 0x69, 0x6e, 0x2e, 0x45, 0x76, 0x65,
    0x6e, 0x74, 0x3a, 0x31, 0xec, 0xf3, 0x32, 0x8d).map(_.toByte)

  private[this] val _tempFolder: TemporaryFolder = new TemporaryFolder

  @Rule def tempFolder: TemporaryFolder = _tempFolder

  private[this] def writeToFile(filePath: Path, data: Array[Byte]): Unit = {
    Files.write(filePath, data, StandardOpenOption.APPEND)
  }

  @Test def testEmptyEventFile(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, Array.empty[Byte])
    val reader = EventFileReader(filePath)
    assert(reader.load().toSeq === Seq.empty[Event])
  }

  @Test def testSingleWrite(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, record)
    val reader = EventFileReader(filePath)
    val events = reader.load().toSeq
    assert(events.size === 1)
    assert(events.head.getWallTime === 1440183447.0)
  }

  @Test def testMultipleWrites(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, record)
    val reader = EventFileReader(filePath)
    assert(reader.load().toSeq.size === 1)
    writeToFile(filePath, record)
    assert(reader.load().toSeq.size === 1)
  }

  @Test def testMultipleLoads(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, record)
    val reader = EventFileReader(filePath)
    reader.load()
    reader.load()
    assert(reader.load().toSeq.size === 1)
  }

  @Test def testMultipleWritesAtOnce(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, record)
    writeToFile(filePath, record)
    val reader = EventFileReader(filePath)
    assert(reader.load().toSeq.size === 2)
  }

  @Test def testMultipleWritesWithBadWrite(): Unit = {
    val filePath = tempFolder.newFile().toPath
    writeToFile(filePath, record)
    writeToFile(filePath, record)
    // Test that we ignore partial record writes at the end of the file.
    writeToFile(filePath, Array(1, 2, 3).map(_.toByte))
    val reader = EventFileReader(filePath)
    assert(reader.load().toSeq.size === 2)
  }
}
