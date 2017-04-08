package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.Exception.InvalidGraphElementException
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.jni.{Graph => NativeGraph}

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Graph(private var nativeHandle: Long) extends Closeable {
  private[api] val opsCache: mutable.Map[Long, Op] = mutable.LongMap[Op]()

  /** Returns the op with the specified name.
    *
    * @param  name Op name.
    * @return Option containing the op corresponding to that name (`None` if such an op does not exist in this graph).
    */
  private[api] def findOp(name: String): Option[Op] =
    NativeHandleLock.synchronized {
      val opHandle: Long = NativeGraph.findOp(nativeHandle, name)
      if (opHandle == 0)
        None
      else
        Some(opsCache.getOrElseUpdate(opHandle, Op(this, opHandle)))
    }

  /** Returns all ops of this graph.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    *
    * @return Array containing all ops of this graph.
    */
  def ops: Array[Op] =
    NativeHandleLock.synchronized {
      NativeGraph.ops(nativeHandle).map(handle => opsCache.getOrElseUpdate(handle, Op(this, handle)))
    }

  /** Returns the op referred to by the provided name, in this graph.
    *
    * If such an op cannot be found, an informative exception is thrown.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    *
    * @param  name Op name.
    * @return Op, from this graph, corresponding to that name.
    */
  @throws[InvalidGraphElementException]
  def opByName(name: String): Op =
    graphElementByName(name = name, allowOp = true, allowOpOutput = false).left.get

  /** Returns the op output referred to by the provided name, in this graph.
    *
    * If such an op output cannot be found, an informative exception is thrown.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    *
    * @param  name Op output name.
    * @return Op output, from this graph, corresponding to that name.
    */
  @throws[InvalidGraphElementException]
  def opOutputByName(name: String): Op.Output =
    graphElementByName(name = name, allowOp = false, allowOpOutput = true).right.get

  /** Returns the [[Op]] or [[Op.Output]] referred to by the provided name, in this graph.
    *
    * This function validates that `name` refers to an element of this graph, and gives an informative error message if
    * it does not. It is the canonical way to get/validate an [[Op]] or [[Op.Output]] from an external argument
    * reference in the Session API. The vast majority of this function is figuring out what an API user might be doing
    * wrong, so that we can give helpful error messages.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    *
    * @param  name          Name of the graph element being looked up.
    * @param  allowOpOutput Allow op outputs to be considered for the graph element to return.
    * @param  allowOp       Allow ops to be considered for the graph element to return.
    * @return Graph element named `name`.
    * @throws InvalidGraphElementException  If the provided name cannot be associated with an element of this graph.
    */
  @throws[InvalidGraphElementException]
  private[api] def graphElementByName(
      name: String, allowOpOutput: Boolean = true, allowOp: Boolean = true): Either[Op, Op.Output] =
    NativeHandleLock.synchronized {
      if (!allowOpOutput && !allowOp)
        throw new IllegalArgumentException("'allowOpOutput' and 'allowOp' cannot both be set to 'false'.")
      if (name.contains(':')) {
        if (allowOpOutput) {
          val nameParts = name.split(':')
          if (nameParts.length != 2 || !nameParts(1).matches("\\d+"))
            throw InvalidGraphElementException(
              s"Name '$name' looks a like an op output name, but it is not a valid one. Op output names must be of " +
                  "the form \"<op_name>:<output_index>\".")
          val opName = nameParts(0)
          val opOutputIndex = nameParts(1).toInt
          val graphOp = findOp(opName) match {
            case Some(o) => o
            case None => throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does not exist in the graph.")
          }
          if (opOutputIndex > graphOp.numOutputs - 1)
            throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does exist in the graph, but it only has ${graphOp.numOutputs} output(s).")
          Right(graphOp.outputs(opOutputIndex))
        } else {
          throw InvalidGraphElementException(
            s"Name '$name' appears to refer to an op output, but 'allowOpOutput' was set to 'false'.")
        }
      } else if (allowOp) {
        findOp(name) match {
          case Some(o) => Left(o)
          case None => throw InvalidGraphElementException(
            s"Name '$name' refers to an op which does not exist in the graph.")
        }
      } else {
        findOp(name) match {
          case Some(_) => throw InvalidGraphElementException(
            s"Name '$name' appears to refer to an op, but 'allowOp' was set to 'false'.")
          case None =>
        }
        throw InvalidGraphElementException(
          s"Name '$name' looks like an (invalid) op name, and not an op output name. Op output names must be of the " +
              "form \"<op_name>:<output_index>\".")
      }
    }

  private object NativeHandleLock
  private var referenceCount: Int = 0

  /** Release resources associated with the Graph.
    *
    * <p>Blocks until there are no active {@link Session} instances referring to this Graph. A Graph
    * is not usable after close returns.
    */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        while (referenceCount > 0) {
          try {
            NativeHandleLock.wait()
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
              // Possible leak of the graph in this case?
              return
          }
        }
        NativeGraph.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

  /** Import a serialized representation of a TensorFlow graph.
    *
    * @param graphDef the serialized representation of a TensorFlow graph.
    * @param prefix   a prefix that will be prepended to names in graphDef
    * @throws IllegalArgumentException if graphDef is not a recognized serialization of a graph.
    * @see #importGraphDef(byte[])
    */
  @throws[IllegalArgumentException]
  def importGraphDef(graphDef: Array[Byte], prefix: String): Unit = {
    if (graphDef == null || prefix == null)
      throw new IllegalArgumentException("graphDef and prefix cannot be null.")
    NativeHandleLock.synchronized {
      NativeGraph.importGraphDef(nativeHandle, graphDef, prefix)
    }
  }

  /** Import a serialized representation of a TensorFlow graph.
    *
    * <p>The serialized representation of the graph, often referred to as a <i>GraphDef</i>, can be
    * generated by {@link #toGraphDef()} and equivalents in other language APIs.
    *
    * @throws IllegalArgumentException if graphDef is not a recognized serialization of a graph.
    * @see #importGraphDef(byte[], String)
    */
  @throws[IllegalArgumentException]
  def importGraphDef(graphDef: Array[Byte]): Unit = importGraphDef(graphDef, "")

  /** Generate a serialized representation of the Graph.
    *
    * @see #importGraphDef(byte[])
    * @see #importGraphDef(byte[], String)
    */
  def toGraphDef: Array[Byte] = {
    NativeHandleLock.synchronized {
      NativeGraph.toGraphDef(nativeHandle)
    }
  }

  def reference: Reference = Reference()

  // Related native objects (such as the TF_Operation object backing an Operation instance)
  // have a validity tied to that of the Graph. The handles to those native objects are not
  // valid after Graph.close() has been invoked.
  //
  // Instances of the Reference class should be used to ensure the Graph has not been closed
  // while dependent handles are in use.
  final case class Reference() extends Closeable {
    NativeHandleLock.synchronized {
      if (Graph.this.nativeHandle == 0)
        throw new IllegalStateException("close() has been called on the Graph")
      referenceCount += 1
    }

    override def close(): Unit = {
      NativeHandleLock.synchronized {
        if (Graph.this.nativeHandle != 0) {
          referenceCount -= 1
          if (referenceCount == 0)
            NativeHandleLock.notifyAll()
        }
      }
    }

    def nativeHandle: Long = {
      NativeHandleLock.synchronized {
        if (Graph.this.nativeHandle != 0)
          Graph.this.nativeHandle
        else
          0
      }
    }
  }
}

object Graph {
  def apply(): Graph = Graph(nativeHandle = NativeGraph.allocate())
}
