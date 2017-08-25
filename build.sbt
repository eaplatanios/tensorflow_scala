import sbtprotobuf.{ProtobufPlugin => PB}

organization in ThisBuild := "org.platanios"
version in ThisBuild := "0.1"
scalaVersion in ThisBuild := "2.12.2"
crossScalaVersions := Seq("2.11.8", "2.12.2")
licenses in ThisBuild := Seq(("Apache License 2.0", url("https://www.apache.org/licenses/LICENSE-2.0.txt")))

val tensorFlowVersion = "1.3.0"

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-feature",
  "-language:implicitConversions",
  "-unchecked"
)

logBuffered in Test := false

scalacOptions in (ThisBuild, Compile, doc) ++= Seq("-groups", "-implicits")
//scalacOptions in (ThisBuild, Compile, doc) ++= Opts.doc.externalAPI((
//  file(s"${(packageBin in Compile).value}") -> url("http://platanios.org/")) :: Nil)
autoAPIMappings := true

lazy val loggingDependencies = Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0",
  "org.slf4j" % "slf4j-api" % "1.7.24",
  "org.apache.logging.log4j" % "log4j-api" % "2.8",
  "org.apache.logging.log4j" % "log4j-core" % "2.8",
  "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.8"
)

lazy val all = (project in file("."))
    .aggregate(jni, api, data, examples)
    .settings(
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      packagedArtifacts in RootProject(file(".")) := Map.empty
    )

lazy val jni = (project in file("./jni"))
    .enablePlugins(GenerateTensorOps, JniNative)
    .settings(
      name := "tensorflow-jni",
      libraryDependencies ++= loggingDependencies,
      // Test dependencies
      libraryDependencies += "junit" % "junit" % "4.12",
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      // Tensor op code generation settings
      target in generateTensorOps := sourceDirectory.value / "main",
      ops in generateTensorOps := Map(
        "Basic" -> Seq(
          "ZerosLike", "OnesLike",/*"Fill",*/"Rank", "Size", "Shape", "ExpandDims", "Squeeze", "Pack", "ParallelConcat",
          "Unpack", "ConcatV2",/*"ConcatOffset", "Split", "SplitV",*/"Tile", "Pad", "MirrorPad", "Reshape", "Transpose",
          "InvertPermutation", "ReverseV2", "ReverseSequence", "SpaceToBatchND", "BatchToSpaceND", "SpaceToDepth",
          "DepthToSpace",/*"Where",*/"Unique", "UniqueWithCounts", "ListDiff", "GatherV2", "GatherNd", "ScatterNd",
          "Slice", "StridedSlice", "CheckNumerics",/*"EditDistance", "OneHot",*/"BroadcastArgs", "StopGradient",
          "PreventGradient"),
        "Math" -> Seq("Cast", "Add", "Sub")
      ),
      scalaPackage in generateTensorOps := "tensors",
      // Native bindings compilation settings
      target in javah := sourceDirectory.value / "main" / "native" / "include",
      sourceDirectory in nativeCompile := sourceDirectory.value / "main" / "native",
      target in nativeCompile := target.value / "native" / nativePlatform.value,
      // Specify the order in which the different compilation tasks are executed
      nativeCompile := nativeCompile.dependsOn(generateTensorOps).value,
      compile in Compile := (compile in Compile).dependsOn(nativeCompile).value
    )

lazy val api = (project in file("./api"))
    .dependsOn(jni)
    .settings(
      name := "tensorflow-api",
      libraryDependencies ++= loggingDependencies,
      libraryDependencies += "org.typelevel" %% "spire" % "0.14.1",
      libraryDependencies += "org.tensorflow" % "proto" % tensorFlowVersion,
      libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.5",
      libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2",
      // Test dependencies
      libraryDependencies += "junit" % "junit" % "4.12",
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      // Protobuf settings
      PB.protobufSettings,
      version in PB.protobufConfig := "3.3.1",
      libraryDependencies += "com.google.protobuf" % "protobuf-java" % (version in PB.protobufConfig).value % PB.protobufConfig.name,
      sourceDirectory in PB.protobufConfig := sourceDirectory.value / "main" / "proto",
      javaSource in PB.protobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
      sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java"
    )

lazy val data = (project in file("./data"))
    .dependsOn(api)
    .settings(
      name := "data",
      libraryDependencies ++= loggingDependencies,
      // Test dependencies
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
    )

lazy val examples = (project in file("./examples"))
    .dependsOn(api, data)
    .settings(
      name := "examples",
      libraryDependencies ++= loggingDependencies
    )
