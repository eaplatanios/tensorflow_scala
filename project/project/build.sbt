organization in ThisBuild := "org.platanios"
version in ThisBuild := "0.1"
scalaVersion in ThisBuild := "2.12.2"
crossScalaVersions := Seq("2.11.8", "2.12.2")
licenses in ThisBuild := Seq(("Apache License 2.0", url("https://www.apache.org/licenses/LICENSE-2.0.txt")))

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-feature",
  "-Xfatal-warnings",
  "-Xlint"
)

lazy val plugins = (project in file("sbt-jni"))
    .settings(
      name := "sbt-jni",
      sbtPlugin := true,
      publishMavenStyle := false,
      libraryDependencies += "org.ow2.asm" % "asm" % "5.0.4"
    )
