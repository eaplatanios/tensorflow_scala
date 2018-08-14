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

logLevel := Level.Warn

val tensorFlowVersion = "1.10.0"

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.ow2.asm" % "asm" % "5.0.4",
  "org.tensorflow" % "proto" % tensorFlowVersion)

// addSbtPlugin("com.geirsson" % "sbt-scalafmt" % "1.6.0-RC3")
// addSbtPlugin("org.scalameta" % "sbt-metals" % "0.1.0-M1+267-28b92d0a")

// addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "latest.release")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.3")

// Plugins used for generating the library website
addSbtPlugin("com.eed3si9n" % "sbt-unidoc"     % "0.4.1")
addSbtPlugin("com.47deg"    % "sbt-microsites" % "0.7.18")

// Packaging and publishing related plugins
addSbtPlugin("com.github.gseitz" % "sbt-release"  % "1.0.8")
addSbtPlugin("com.jsuereth"      % "sbt-pgp"      % "1.1.1")
addSbtPlugin("org.xerial.sbt"    % "sbt-sonatype" % "2.0")

// Generally useful plugins
// addSbtPlugin("io.get-coursier" %  "sbt-coursier" % "1.0.3") // Provides fast dependency resolution.
