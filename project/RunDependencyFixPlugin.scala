import sbt._
import sbt.Keys._
import java.io.File

/** Adds the extension method `dependsOnRun` to projects, to work around an SBT bug. The bug should be fixed in the next
  * SBT release.
  *
  * Borrowed from [the sbt-jni plugin](https://github.com/jodersky/sbt-jni).
  *
  * @author Emmanouil Antonios Platanios
  */
object RunDependencyFixPlugin extends AutoPlugin {

  override def requires = plugins.CorePlugin
  override def trigger = allRequirements

  object autoImport {

    val runClasspath = taskKey[Seq[sbt.internal.util.Attributed[File]]]("Classpath used in run task")

    def dependsOnRunSettings(project: Project) = Seq(
      runClasspath in Compile ++= (runClasspath in Compile in project).value,
      run := {
        Defaults.runTask(
          runClasspath in Compile,
          mainClass in Compile in run,
          runner in run
        ).evaluated
      }
    )

    implicit class RichProject(project: Project) {
      @deprecated("Workaround for https://github.com/sbt/sbt/issues/3425. " +
                      "Use `dependsOn(<project> % Runtime)` when fixed.", "1.3.0")
      def dependsOnRun(other: Project) = {
        project.settings(dependsOnRunSettings(other): _*)
      }
    }

  }
  import autoImport._

  override def projectSettings = Seq(
    runClasspath in Compile := (fullClasspath in Compile).value
  )

}
