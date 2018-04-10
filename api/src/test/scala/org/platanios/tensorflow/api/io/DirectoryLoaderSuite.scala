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

import org.junit.{Before, Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite

import java.io.InputStream
import java.nio.file.{Files, Path, StandardOpenOption}

/**
  * @author Emmanouil Antonios Platanios
  */
class DirectoryLoaderSuite extends JUnitSuite {
  private[this] var _tempPath  : Path                  = _
  private[this] var _loader    : DirectoryLoader[Byte] = _
  private[this] val _tempFolder: TemporaryFolder       = new TemporaryFolder

  @Rule def tempFolder: TemporaryFolder = _tempFolder

  @Before def setUp(): Unit = {
    _tempPath = tempFolder.newFolder().toPath
    _loader = DirectoryLoader[Byte](_tempPath, DirectoryLoaderSuite.ByteLoader)
  }

  private[this] def writeToFile(filename: String, content: String): Unit = {
    val fileIO = FileIO(_tempPath.resolve(filename), FileIO.APPEND)
    fileIO.write(content)
    fileIO.close()
  }

  private[this] def loadAll(): Unit = _loader.load().foreach(_ => ())

  private[this] def assertLoaderYields(values: Seq[Byte]): Unit = {
    assert(_loader.load().toSeq === values)
  }

  @Test def testEmptyDirectory(): Unit = {
    assertLoaderYields(Seq.empty[Byte])
  }

  @Test def testSingleWrite(): Unit = {
    writeToFile("a", "abc")
    assertLoaderYields("abc".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testMultipleWrites(): Unit = {
    writeToFile("a", "abc")
    assertLoaderYields("abc".getBytes.toSeq)
    writeToFile("a", "xyz")
    assertLoaderYields("xyz".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testMultipleLoads(): Unit = {
    writeToFile("a", "a")
    _loader.load()
    _loader.load()
    assertLoaderYields("a".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testMultipleFilesAtOnce(): Unit = {
    writeToFile("b", "b")
    writeToFile("a", "a")
    assertLoaderYields("ab".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testFinishesLoadingFileWhenSwitchingToNewFile(): Unit = {
    writeToFile("a", "a")
    assertLoaderYields("a".getBytes.toSeq)
    writeToFile("a", "b")
    writeToFile("b", "c")
    assertLoaderYields("bc".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testIntermediateEmptyFiles(): Unit = {
    writeToFile("a", "a")
    writeToFile("b", "")
    writeToFile("c", "c")
    assertLoaderYields("ac".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testDetectsNewOldFiles(): Unit = {
    writeToFile("b", "a")
    loadAll()
    writeToFile("a", "a")
    loadAll()
    assert(_loader.outOfOrderWritesDetected)
  }

  @Test def testIgnoresNewerFiles(): Unit = {
    writeToFile("a", "a")
    loadAll()
    writeToFile("q", "a")
    loadAll()
    assert(!_loader.outOfOrderWritesDetected)
  }

  @Test def testDetectsChangingOldFiles(): Unit = {
    writeToFile("a", "a")
    writeToFile("b", "a")
    loadAll()
    writeToFile("a", "c")
    loadAll()
    assert(_loader.outOfOrderWritesDetected)
  }

  @Test def testDoesNotCrashWhenFileIsDeleted(): Unit = {
    writeToFile("a", "a")
    loadAll()
    Files.delete(_tempPath.resolve("a"))
    writeToFile("b", "b")
    assertLoaderYields("b".getBytes.toSeq)
  }

  @Test def testThrowsDirectoryDeletedExceptionWhenDirectoryIsDeleted(): Unit = {
    writeToFile("a", "a")
    loadAll()
    Files.delete(_tempPath.resolve("a"))
    Files.delete(_tempPath)
    assertThrows[DirectoryLoader.DirectoryDeletedException](loadAll())
  }

  // TODO: [TESTS] testDoesNotThrowDirectoryDeletedExceptionWhenOutageIsTransient.

  @Test def testPathFilter(): Unit = {
    _loader = DirectoryLoader[Byte](
      _tempPath, DirectoryLoaderSuite.ByteLoader, p => !p.toString.contains("do_not_watch_me"))
    writeToFile("a", "a")
    writeToFile("do_not_watch_me", "b")
    writeToFile("c", "c")
    assertLoaderYields("ac".getBytes.toSeq)
    assert(!_loader.outOfOrderWritesDetected)
  }
}

object DirectoryLoaderSuite {
  /** Dummy loader that loads individual bytes from a file. */
  private[DirectoryLoaderSuite] case class ByteLoader(path: Path) extends Loader[Byte] {
    private[this] val file: InputStream = Files.newInputStream(path, StandardOpenOption.READ)

    override def load(): Iterator[Byte] = new Iterator[Byte] {
      private[this] var nextByte: Option[Byte] = None

      private[this] def readByte(): Option[Byte] = {
        try {
          val byte = file.read()
          if (byte != -1) {
            Some(byte.toByte)
          } else {
            None
          }
        } catch {
          case _: Throwable => None
        }
      }

      override def hasNext: Boolean = {
        if (nextByte.isEmpty)
          nextByte = readByte()
        nextByte.isDefined
      }

      override def next(): Byte = {
        if (nextByte.isEmpty)
          nextByte = readByte()
        val byte = nextByte.get
        nextByte = None
        byte
      }
    }
  }
}
