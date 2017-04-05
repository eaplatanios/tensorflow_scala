//import sbtprotobuf.{ProtobufPlugin => PB}

organization in ThisBuild := "org.platanios"
version in ThisBuild := "1.0"
scalaVersion in ThisBuild := "2.12.1"

scalacOptions in ThisBuild ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

exportJars := true

// Scaladoc options
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

lazy val tensorflow = (project in file("."))
    .aggregate(tensorflow_jni, tensorflow_api) // Ignoring tensorflow_macros for now
    .settings(
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      packagedArtifacts in file(".") := Map.empty
    )

lazy val tensorflow_jni = (project in file("./tensorflow_jni"))
    .enablePlugins(JniNative)
    .settings(
      name := "tensorflow_jni",
      // Test dependencies
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      // Native bindings compilation settings
      target in javah := sourceDirectory.value / "main" / "native" / "include",
      sourceDirectory in nativeCompile := sourceDirectory.value / "main" / "native",
      target in nativeCompile := target.value / "native" / nativePlatform.value
      // Protobuf settings
      // PB.protobufSettings,
      // version in PB.protobufConfig := "3.2.0",
      // libraryDependencies += "com.google.protobuf" % "protobuf-java" % (version in PB.protobufConfig).value % PB.protobufConfig.name,
      // sourceDirectory in PB.protobufConfig := sourceDirectory.value / "main" / "protobuf",
      // javaSource in PB.protobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
      // sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java"
    )

//lazy val tensorflow_macros = (project in file("./tensorflow_macros"))
//    .dependsOn(tensorflow_jni)
//    .settings(
//      name := "tensorflow_macros",
//      // Logging dependencies
//      libraryDependencies ++= loggingDependencies,
//      // Test dependencies
//      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
//      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
//      // Meta-programming dependencies
//      libraryDependencies += "org.scalameta" %% "scalameta" % "1.6.0",
//      addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M7" cross CrossVersion.full),
//      // Protobuf settings
//      PB.protobufSettings,
//      version in PB.protobufConfig := "3.2.0",
//      libraryDependencies += "com.google.protobuf" % "protobuf-java" % (version in PB.protobufConfig).value % PB.protobufConfig.name,
//      sourceDirectory in PB.protobufConfig := sourceDirectory.value / "main" / "protobuf",
//      javaSource in PB.protobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
//      sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java"
//    )

lazy val tensorflow_api = (project in file("./tensorflow_api"))
    .dependsOn(tensorflow_jni) // Ignoring tensorflow_macros for now
    .settings(
      name := "tensorflow_api",
      // Test dependencies
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
      // Meta-programming dependencies
      // addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M7" cross CrossVersion.full)
    )
