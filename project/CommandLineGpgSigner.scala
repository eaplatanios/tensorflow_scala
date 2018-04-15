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

import com.typesafe.sbt.pgp.PgpSigner
import sbt._
import Keys._

/** A GpgSigner that uses the command-line to run gpg.
  *
  * @author Emmanouil Antonios Platanios
  */
class CommandLineGpgSigner(
    command: String,
    agent: Boolean,
    secRing: String,
    optKey: Option[Long],
    optPassphrase: Option[Array[Char]]
) extends PgpSigner {
  def sign(file: File, signatureFile: File, s: TaskStreams): File = {
    if (signatureFile.exists) IO.delete(signatureFile)
    val passphraseArgs = optPassphrase.map(p => p.mkString("")).map(p => Seq("--passphrase", p)).getOrElse(Seq.empty)
    val keyArgs = optKey.map(k => Seq("--default-key", "0x%x".format(k))).getOrElse(Seq.empty)
    val args = passphraseArgs ++
        Seq("--detach-sign", "--armor") ++
        (if (agent) Seq("--use-agent") else Seq.empty) ++
        keyArgs ++
        Seq("--output", signatureFile.getAbsolutePath, file.getAbsolutePath)
    sys.process.Process(command, args) ! s.log match {
      case 0 => ()
      case n => sys.error("Failure running gpg --detach-sign.  Exit code: " + n)
    }
    signatureFile
  }

  override val toString: String = "GPG-Command(" + command + ")"
}
