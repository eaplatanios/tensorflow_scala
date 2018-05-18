///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.tensorflow.api.io.events
//
//import org.platanios.tensorflow.api.io.FileIO
//
//import org.junit.{Rule, Test}
//import org.junit.rules.TemporaryFolder
//import org.scalatest.junit.JUnitSuite
//
//import java.nio.file.Path
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//class EventMultiplexerSuite extends JUnitSuite {
//
//}
//
//object EventMultiplexerSuite {
//  private[EventMultiplexerSuite] def createCleanDirectory(path: Path): Unit = {
//    if (FileIO.isDirectory(path))
//      FileIO.deleteRecursively(path)
//    FileIO.mkDir(path)
//  }
//
//  private[EventMultiplexerSuite] def createEventsFile(path: Path): Path = {
//    if (!FileIO.isDirectory(path))
//      FileIO.mkDirs(path)
//    val filePath = path.resolve("hypothetical.tfevents.out")
//    FileIO(filePath, FileIO.WRITE).write("").close()
//    filePath
//  }
//}
