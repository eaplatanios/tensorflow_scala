package plugin.org.platanios.sbt.jni.util

import java.io.{ File, FileInputStream, Closeable }
import scala.collection.mutable.HashSet

import org.objectweb.asm.{ ClassReader, ClassVisitor, MethodVisitor, Opcodes }

object BytecodeUtil {

  private class NativeFinder extends ClassVisitor(Opcodes.ASM5) {

    // classes found to contain at least one @native def
    val _nativeClasses = new HashSet[String]
    def nativeClasses = _nativeClasses.toSet

    private var fullyQualifiedName: String = ""

    override def visit(version: Int, access: Int, name: String, signature: String,
        superName: String, interfaces: Array[String]): Unit = {
      fullyQualifiedName = name.replaceAll("/", ".")
    }

    override def visitMethod(access: Int, name: String, desc: String,
        signature: String, exceptions: Array[String]): MethodVisitor = {

      val isNative = (access & Opcodes.ACC_NATIVE) != 0

      if (isNative) {
        _nativeClasses += fullyQualifiedName
      }

      null //return null, do not visit method further
    }

  }

  private def using[A >: Null <: Closeable, R](mkStream: => A)(action: A => R): R = {
    var stream: A = null
    try {
      stream = mkStream
      action(stream)
    } finally {
      if (stream != null) {
        stream.close()
      }
    }
  }

  /** Finds classes containing native implementations.
    *
    * @param classFile java class file from which classes are read
    * @return all fully qualified names of classes that contain at least one member annotated
    *         with @native
    */
  def nativeClasses(classFile: File): Set[String] = using(new FileInputStream(classFile)) { in =>
    val reader = new ClassReader(in)
    val finder = new NativeFinder
    reader.accept(finder, 0)
    finder.nativeClasses
  }

}
