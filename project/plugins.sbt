logLevel := Level.Warn

libraryDependencies += "org.ow2.asm" % "asm" % "5.0.4"

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "latest.release")
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.5.5")

//resolvers += Resolver.bintrayRepo("tek", "maven")
addCompilerPlugin("io.tryp" %% "splain" % "0.2.4")
