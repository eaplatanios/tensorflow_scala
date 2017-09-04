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

logLevel := Level.Warn

val tensorFlowVersion = "1.3.0"

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.ow2.asm" % "asm" % "5.0.4",
  "org.tensorflow" % "proto" % tensorFlowVersion
)

// addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "latest.release")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.3")

// Plugins used for generating the library website
addSbtPlugin("com.eed3si9n" % "sbt-unidoc"     % "0.4.1")
addSbtPlugin("org.tpolecat" % "tut-plugin"     % "0.5.4")
addSbtPlugin("com.47deg"    % "sbt-microsites" % "0.6.2")

// Packaging and publishing related plugins
addSbtPlugin("com.github.gseitz" % "sbt-release"  % "1.0.6")
addSbtPlugin("com.jsuereth"      % "sbt-pgp"      % "1.1.0")
addSbtPlugin("org.xerial.sbt"    % "sbt-sonatype" % "2.0")

// Generally useful plugins
addSbtPlugin("io.get-coursier" %  "sbt-coursier" % "1.0.0-RC10") // Provides fast dependency resolution.
addCompilerPlugin("io.tryp"    %% "splain"       % "0.2.4")      // Prints better implicit resolution errors.
