import org.objectweb.asm.{ClassReader, ClassVisitor, MethodVisitor, Opcodes}

import java.io.{File, FileInputStream}

import scala.collection.JavaConverters._
import scala.collection.mutable

import sbt._
import sbt.Keys._
import sys.process._

/** Adds functionality for generating JNI header files using the `javah` tool.
  *
  * Borrowed from [the sbt-jni plugin](https://github.com/jodersky/sbt-jni).
  *
  * @author Emmanouil Antonios Platanios
  */
object JniJavah extends AutoPlugin {
  override def requires: Plugins = plugins.JvmPlugin
  override def trigger: PluginTrigger = allRequirements

  object autoImport {
    val javahClasses: TaskKey[Set[String]] = taskKey[Set[String]](
      "Finds the fully qualified names of classes containing native declarations.")

    val javah: TaskKey[File] = taskKey[File](
      "Generates the JNI headers. Returns the directory containing generated headers.")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    javahClasses in javah := {
      import xsbti.compile._
      val compiled: CompileAnalysis = (compile in Compile).value
      val classFiles: Set[File] = compiled.readStamps().getAllProductStamps().asScala.keySet.toSet
      val nativeClasses = classFiles flatMap { file => JniJavah.nativeClasses(file) }
      nativeClasses
    },
    target in javah := target.value / "native" / "include",
    javah := {
      val directory = (target in javah).value
      // The full classpath cannot be used here since it also generates resources. In a project combining JniJavah and
      // JniPackage, we would have a chicken-and-egg problem.
      val classPath: String = ((dependencyClasspath in Compile).value.map(_.data) ++ {
        (compile in Compile).value
        Seq((classDirectory in Compile).value)
      }).mkString(sys.props("path.separator"))
      val classes = (javahClasses in javah).value
      val log = streams.value.log
      if (classes.nonEmpty)
        log.info("Headers will be generated to " + directory.getAbsolutePath)
      for (c <- classes) {
        log.info("Generating header for " + c)
        val command = s"javah -d ${directory.getAbsolutePath} -classpath $classPath $c"
        val exitCode = Process(command) ! log
        if (exitCode != 0) sys.error(s"An error occurred while running javah. Exit code: $exitCode.")
      }
      directory
    }
  )

  override lazy val projectSettings: Seq[Setting[_]] = settings

  private class NativeFinder extends ClassVisitor(Opcodes.ASM5) {
    private var fullyQualifiedName: String = ""

    /** Classes found to contain at least one @native definition. */
    private val _nativeClasses = mutable.HashSet.empty[String]

    def nativeClasses: Set[String] = _nativeClasses.toSet

    override def visit(
        version: Int, access: Int, name: String, signature: String, superName: String,
        interfaces: Array[String]): Unit = {
      fullyQualifiedName = name.replaceAll("/", ".")
    }

    override def visitMethod(
        access: Int, name: String, desc: String, signature: String, exceptions: Array[String]): MethodVisitor = {
      val isNative = (access & Opcodes.ACC_NATIVE) != 0
      if (isNative)
        _nativeClasses += fullyQualifiedName
      // Return null, meaning that we do not want to visit the method further.
      null
    }
  }

  /** Finds classes containing native implementations (i.e., `@native` definitions).
    *
    * @param  javaFile Java file from which classes are being read.
    * @return Set containing all the fully qualified names of classes that contain at least one member annotated with
    *         the `@native` annotation.
    */
  def nativeClasses(javaFile: File): Set[String] = {
    var inputStream: FileInputStream = null
    try {
      inputStream = new FileInputStream(javaFile)
      val reader = new ClassReader(inputStream)
      val finder = new NativeFinder
      reader.accept(finder, 0)
      finder.nativeClasses
    } finally {
      if (inputStream != null)
        inputStream.close()
    }
  }
}
