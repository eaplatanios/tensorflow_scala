package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{Session => NativeSession}

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Session private (graph: Graph, private var nativeHandle: Long) extends Closeable {
  private object NativeHandleLock
  private var referenceCount: Int = 0

  def run(
      feeds: Map[Op.Output, Tensor[_]] = Map.empty, fetches: Array[Op.Output] = Array.empty,
      targets: Array[Op] = Array.empty, runOptions: Option[Array[Byte]] = None): Array[Tensor[_]] = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, runOptions = runOptions)._1
  }

  def runWithMetadata(
      feeds: Map[Op.Output, Tensor[_]] = Map.empty, fetches: Array[Op.Output] = Array.empty,
      targets: Array[Op] = Array.empty, runOptions: Option[Array[Byte]] = None): (Array[Tensor[_]], Array[Byte]) = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, runOptions = runOptions, wantMetadata = true)
  }

  private def runHelper(
      feeds: Map[Op.Output, Tensor[_]] = Map.empty, fetches: Array[Op.Output] = Array.empty,
      targets: Array[Op] = Array.empty, runOptions: Option[Array[Byte]] = None,
      wantMetadata: Boolean = false): (Array[Tensor[_]], Array[Byte]) = {
    val (inputs, inputTensors) = feeds.toArray.unzip
    val inputTensorNativeViews = inputTensors.map(_.nativeView)
    val inputTensorHandles: Array[Long] = inputTensorNativeViews.map(_.nativeHandle)
    val inputOpHandles: Array[Long] = inputs.map(_.op.nativeHandle)
    val inputOpIndices: Array[Int] = inputs.map(_.index)
    val outputOpHandles: Array[Long] = fetches.map(_.op.nativeHandle)
    val outputOpIndices: Array[Int] = fetches.map(_.index)
    val targetOpHandles: Array[Long] = targets.map(_.nativeHandle)
    val outputTensorHandles: Array[Long] = Array.ofDim[Long](fetches.length)
    NativeHandleLock.synchronized {
      if (nativeHandle == 0)
        throw new IllegalStateException("close() has been called on the session.")
      referenceCount += 1
    }
    // It's okay to use Operation.getUnsafeNativeHandle() here since the safety depends on the
    // validity of the Graph and the graph reference ensures that.
    val metadata: Array[Byte] = NativeSession.run(
      handle = nativeHandle,
      runOptions = runOptions.getOrElse(Array.empty[Byte]),
      inputTensorHandles = inputTensorHandles,
      inputOpHandles = inputOpHandles,
      inputOpIndices = inputOpIndices,
      outputOpHandles = outputOpHandles,
      outputOpIndices = outputOpIndices,
      targetOpHandles = targetOpHandles,
      wantRunMetadata = wantMetadata,
      outputTensorHandles = outputTensorHandles)
    val outputs: Array[Tensor[_]] = outputTensorHandles.map(Tensor.fromNativeHandle)
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        referenceCount -= 1
        if (referenceCount == 0)
          NativeHandleLock.notifyAll()
      }
    }
    inputTensorNativeViews.foreach(_.close())
    (outputs, metadata)
  }

  override def close(): Unit = {
    graph.reference.close()
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        while (referenceCount > 0) {
          try {
            NativeHandleLock.wait()
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
              // Possible leak of the session and graph in this case?
              return
          }
        }
        NativeSession.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

//  final case class Runner() {
//    private val inputs: ArrayBuffer[Op.Output] = ArrayBuffer[Op.Output]()
//    private val inputTensors: ArrayBuffer[Tensor] = ArrayBuffer[Tensor]()
//    private val outputs: ArrayBuffer[Op.Output] = ArrayBuffer[Op.Output]()
//    private val targets: ArrayBuffer[Op] = ArrayBuffer[Op]()
//    private var runOptions: Option[Array[Byte]] = None
//
//    private def operationByName(name: String): Op =
//      graph.findOp(name = name) match {
//        case Some(op) => op
//        case None => throw new IllegalArgumentException(s"No operation named \'$name\' in the current graph.")
//      }
//
//    def feed(opName: String, index: Int = 0, tensor: Tensor): Runner = {
//      val op: Op = operationByName(name = opName)
//      inputs += op.outputs(index)
//      inputTensors += tensor
//      this
//    }
//
//    def fetch(opName: String, index: Int = 0): Runner = {
//      val op: Op = operationByName(name = opName)
//      outputs += op.outputs(index)
//      this
//    }
//
//    def addTarget(opName: String): Runner = {
//      val op: Op = operationByName(name = opName)
//      targets += op
//      this
//    }
//
//    def setOptions(options: Array[Byte]): Runner = {
//      runOptions = Some(options)
//      this
//    }
//
//    def run(wantMetadata: Boolean = false): (List[Tensor], Array[Byte]) = {
//      val inputTensorHandles: Array[Long] = inputTensors.map(_.nativeHandle).toArray
//      val inputOpHandles: Array[Long] = inputs.map(_.op.nativeHandle).toArray
//      val inputOpIndices: Array[Int] = inputs.map(_.index).toArray
//      val outputOpHandles: Array[Long] = this.outputs.map(_.op.nativeHandle).toArray
//      val outputOpIndices: Array[Int] = this.outputs.map(_.index).toArray
//      val targetOpHandles: Array[Long] = targets.map(_.nativeHandle).toArray
//      val outputTensorHandles: Array[Long] = Array.ofDim[Long](this.outputs.length)
//
//      // It's okay to use Operation.getUnsafeNativeHandle() here since the safety depends on the
//      // validity of the Graph and graphRef ensures that.
//      val metadata: Array[Byte] = using(Reference()) { _ =>
//        NativeSession.run(
//          handle = nativeHandle,
//          runOptions = runOptions.getOrElse(Array.empty[Byte]),
//          inputTensorHandles = inputTensorHandles,
//          inputOpHandles = inputOpHandles,
//          inputOpIndices = inputOpIndices,
//          outputOpHandles = outputOpHandles,
//          outputOpIndices = outputOpIndices,
//          targetOpHandles = targetOpHandles,
//          wantRunMetadata = wantMetadata,
//          outputTensorHandles = outputTensorHandles)
//      }
//      val outputs: List[Tensor] =
//        outputTensorHandles.map(h => Tensor.fromNativeHandle(nativeHandle = h)).toList
//      (outputs, metadata)
//    }
//
//    final case class Reference() extends Closeable {
//      NativeHandleLock.synchronized {
//        if (nativeHandle == 0)
//          throw new IllegalStateException("close() has been called on the session.")
//        referenceCount += 1
//      }
//
//      override def close(): Unit = {
//        NativeHandleLock.synchronized {
//          if (nativeHandle != 0) {
//            referenceCount -= 1
//            if (referenceCount == 0)
//              NativeHandleLock.notifyAll()
//          }
//        }
//      }
//    }
//  }
}

object Session {
  def apply(graph: Graph): Session = {
    Session(graph = graph, nativeHandle = using(graph.reference)(r => NativeSession.allocate(r.nativeHandle)))
  }
}
