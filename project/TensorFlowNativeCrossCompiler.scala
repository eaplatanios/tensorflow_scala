/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import JniCrossPackage._

import java.nio.file.Path

import scala.sys.process._

/** Helper class for cross-compiling the TensorFlow dynamic library within Docker containers.
  *
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowNativeCrossCompiler {
  def compile(
      workingDir: Path,
      targetDir: String,
      gitRepository: String,
      gitRepositoryBranch: String,
      platform: Platform
  ): ProcessBuilder = {
    val repoDir = workingDir.resolve("tensorflow").toFile
    var processBuilder = Process("rm" :: "-rf" :: "tensorflow" :: Nil, workingDir.toFile)
    processBuilder = processBuilder #&& Process("git" :: "clone" :: gitRepository :: Nil, workingDir.toFile)
    if (gitRepositoryBranch != "master") {
      processBuilder = processBuilder #&&
          Process("git" :: "checkout" :: "-b" :: gitRepositoryBranch ::
              s"origin/$gitRepositoryBranch" :: Nil, repoDir)
    }
    val tfLibFilename = TensorFlowNativePackage.tfLibFilename(platform)
    processBuilder #&&
        Process(platform.compileScript, repoDir) #&&
        Process(
          "cp" :: s"lib_package/$tfLibFilename" :: s"$targetDir/downloads/lib/$tfLibFilename" :: Nil, repoDir) #&&
        Process("rm" :: "-rf" :: "tensorflow" :: Nil, workingDir.toFile)
  }

  implicit class RichPlatform(platform: Platform) {
    private[this] val ciBuildScript = "tensorflow/tools/ci_build/ci_build.sh"

    def compileScript: String = platform match {
      case LINUX_x86_64 => "tensorflow/tools/ci_build/linux/libtensorflow_cpu.sh"
      case LINUX_GPU_x86_64 => "tensorflow/tools/ci_build/linux/libtensorflow_gpu.sh"
      case DARWIN_x86_64 => s"$ciBuildScript CPU tensorflow/tools/ci_build/osx/libtensorflow_cpu.sh"
    }
  }
}
