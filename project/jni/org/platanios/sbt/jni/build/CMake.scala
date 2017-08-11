package plugin.org.platanios.sbt.jni.build

import sbt._

object CMake extends BuildTool with ConfigureMakeInstall {

  override val name = "CMake"

  override def detect(baseDirectory: File) = baseDirectory.list().contains("CMakeLists.txt")

  override protected def templateMappings = Seq(
    "/ch/jodersky/sbt/jni/templates/CMakeLists.txt" -> "CMakeLists.txt"
  )

  override def getInstance(baseDir: File, buildDir: File, logger: Logger) = new Instance {

    override def log = logger
    override def baseDirectory = baseDir
    override def buildDirectory = buildDir

    override def configure(target: File) = Process(
      // disable producing versioned library files, not needed for fat jars
      "cmake " +
          s"-DCMAKE_INSTALL_PREFIX:PATH=${target.getAbsolutePath} " +
          "-DCMAKE_BUILD_TYPE=Release " +
          "-DSBT:BOOLEAN=true " +
          baseDirectory.getAbsolutePath,
      buildDirectory
    )
  }

}