organization in ThisBuild := "org.platanios"
version in ThisBuild := "1.0"
scalaVersion in ThisBuild := "2.12.1"

val tensorFlowVersion = "1.1.0-rc2"

scalacOptions in ThisBuild ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

logBuffered in Test := false

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
    .aggregate(tensorflow_jni, tensorflow_api)
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
    )

lazy val tensorflow_api = (project in file("./tensorflow_api"))
    .dependsOn(tensorflow_jni)
    .settings(
      name := "tensorflow_api",
      libraryDependencies ++= loggingDependencies,
      libraryDependencies += "org.tensorflow" % "proto" % tensorFlowVersion,
      // Test dependencies
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
    )
