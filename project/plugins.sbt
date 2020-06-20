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

logLevel := Level.Warn

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.ow2.asm" % "asm" % "6.2.1",
  // The following is needed to automatically generate the eager ops.
  "org.tensorflow" % "proto" % "1.15.0")

addSbtPlugin("ch.epfl.scala" % "sbt-bloop" % "1.2.5")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.5")

// Plugins used for the documentation website.
addSbtPlugin("com.lightbend.paradox" % "sbt-paradox" % "0.8.0")
addSbtPlugin("io.github.jonas" % "sbt-paradox-material-theme" % "0.6.0")
addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.3.2")
addSbtPlugin("com.typesafe.sbt" % "sbt-ghpages" % "0.6.3")
addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "latest.release")

// Packaging and publishing related plugins.
addSbtPlugin("com.github.gseitz" % "sbt-release"  % "1.0.13")
addSbtPlugin("com.jsuereth"      % "sbt-pgp"      % "2.0.0")
addSbtPlugin("org.xerial.sbt"    % "sbt-sonatype" % "3.9.2")

// Generally useful plugins.
// addSbtPlugin("io.get-coursier" %  "sbt-coursier" % "2.0.0-RC6-1") // Provides fast dependency resolution.
