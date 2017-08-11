package plugin.org.platanios.sbt.jni.plugins

import sbt._
import sbt.Keys._
import org.platanios.sbt.jni.util.BytecodeUtil

/** Adds `javah` header-generation functionality to projects. */
object JniJavah extends AutoPlugin {

  override def requires = plugins.JvmPlugin
  override def trigger = allRequirements

  object autoImport {

    val javahClasses = taskKey[Set[String]](
      "Finds fully qualified names of classes containing native declarations."
    )

    val javah = taskKey[File](
      "Generate JNI headers. Returns the directory containing generated headers."
    )

  }

  import autoImport._

  lazy val mainSettings: Seq[Setting[_]] = Seq(

    javahClasses in javah := {
      val compiled: inc.Analysis = (compile in Compile).value
      val classFiles: Set[File] = compiled.relations.allProducts.toSet
      val nativeClasses = classFiles flatMap { file =>
        BytecodeUtil.nativeClasses(file)
      }
      nativeClasses
    },

    target in javah := target.value / "native" / "include",

    javah := {
      val out = (target in javah).value

      // fullClasspath can't be used here since it also generates resources. In
      // a project combining JniJavah and JniPackage, we would have a chicken-and-egg
      // problem.
      val jcp: Seq[File] = (dependencyClasspath in Compile).value.map(_.data) ++ {
        (compile in Compile).value;
        Seq((classDirectory in Compile).value)
      }

      val cp = jcp.mkString(sys.props("path.separator"))
      val log = streams.value.log

      val classes = (javahClasses in javah).value
      if (!classes.isEmpty) {
        log.info("Headers will be generated to " + out.getAbsolutePath)
      }
      for (clazz <- classes) {
        log.info("Generating header for " + clazz)
        val parts = Seq(
          "javah",
          "-d", out.getAbsolutePath,
          "-classpath", cp,
          clazz
        )
        val cmd = parts.mkString(" ")
        val ev = Process(cmd) ! streams.value.log
        if (ev != 0) sys.error(s"Error occured running javah. Exit code: ${ev}")
      }
      out
    }
  )

  override lazy val projectSettings = mainSettings

}
