logLevel := Level.Warn

val tensorFlowVersion = "1.3.0"

libraryDependencies ++= Seq(
  "org.ow2.asm" % "asm" % "5.0.4",
  "org.tensorflow" % "proto" % tensorFlowVersion
)

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "latest.release")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.6.2")
addSbtPlugin("com.47deg"  % "sbt-microsites" % "0.6.1")

//resolvers += Resolver.bintrayRepo("tek", "maven")
addCompilerPlugin("io.tryp" %% "splain" % "0.2.4")
