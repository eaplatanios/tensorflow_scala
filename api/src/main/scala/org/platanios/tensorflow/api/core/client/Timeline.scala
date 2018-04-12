/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.core.client

import com.typesafe.scalalogging.Logger
import io.circe._
import io.circe.generic.auto._
import io.circe.syntax._
import org.slf4j.LoggerFactory
import org.tensorflow.framework.StepStats

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.util.matching.Regex

/** Helper for visualizing execution timelines of TensorFlow steps.
  *
  * A timeline is used for visualizing the execution of a TensorFlow computation. It shows the timings and concurrency
  * of execution at the granularity of TensorFlow ops.
  *
  * @author Emmanouil Antonios Platanios
  */
object Timeline {
  protected[Timeline] val logger = Logger(LoggerFactory.getLogger("Core / Timeline"))

  protected[Timeline] val opLabelRegex: Regex = """(.*) = (.*)\((.*)\)""".r

  /** Generates a Chrome trace JSON string for visualizing execution timelines of TensorFlow steps.
    *
    * A timeline is used for visualizing the execution of a TensorFlow computation. It shows the timings and concurrency
    * of execution at the granularity of TensorFlow ops.
    *
    * @param  stepStatistics `StepStats` protocol buffer containing execution times.
    * @param  showDataFlow   If `true`, add flow events to the trace connecting producers and consumers of tensors.
    * @param  showMemory     If `true`, add object snapshot events to the trace showing sizes and lifetimes of tensors.
    * @param  prettyJson     If `true`, produces human-readable JSON output.
    * @return JSON string containing the Chrome trace.
    */
  def generateChromeTrace(
      stepStatistics: StepStats,
      showDataFlow: Boolean = false,
      showMemory: Boolean = false,
      prettyJson: Boolean = false
  ): String = {
    val chromeTraceFormatter = ChromeTraceFormatter()
    var devicePIDs = Map.empty[String, Int]
    var tensorsPIDs = Map.empty[String, Int]
    var threadIDs = Map.empty[Int, Int]
    var tensors = Map.empty[String, TensorTracker]
    var flowStarts = Map.empty[String, (Long, Int, Int)]
    var nextPID = 0
    var nextFlowID = 0

    // Allocate fake process ids for each device in the step statistics.
    val allocatorsPID = nextPID
    chromeTraceFormatter.emitPID("Allocators", allocatorsPID)
    nextPID += 1

    stepStatistics.getDevStatsList.asScala.foreach(deviceStats => {
      val deviceName = deviceStats.getDevice

      // Add processes in the Chrome trace to show compute and data activity.
      val devicePID = nextPID
      val tensorsPID = nextPID + 1
      devicePIDs += deviceName -> devicePID
      tensorsPIDs += deviceName -> tensorsPID
      chromeTraceFormatter.emitPID(s"${deviceStats.getDevice} Compute", devicePID)
      chromeTraceFormatter.emitPID(s"${deviceStats.getDevice} Tensors", tensorsPID)
      nextPID += 2

      // TODO: [TF_UPDATE] Genuine thread IDs in NodeExecStats might be helpful.
      val lanes = mutable.ArrayBuffer[Long]()

      deviceStats.getNodeStatsList.asScala.zipWithIndex.foreach {
        case (nodeStats, nodeIndex) =>
          // Analyze tensor references to track data flow.
          val nodeName = nodeStats.getNodeName
          val startTime = nodeStats.getAllStartMicros
          val duration = nodeStats.getAllEndRelMicros
          val endTime = nodeStats.getAllStartMicros + duration

          // Assign non-overlapping lanes for the activities on each device.
          val threadID = {
            val l = lanes.indexWhere(_ < startTime)
            if (l < 0) {
              lanes.append(endTime)
              lanes.length - 1
            } else {
              lanes(l) = endTime
              l
            }
          }
          threadIDs += nodeIndex -> threadID
      }
    })

    stepStatistics.getDevStatsList.asScala.foreach(deviceStats => {
      val deviceName = deviceStats.getDevice
      val devicePID = devicePIDs(deviceName)
      val tensorsPID = tensorsPIDs(deviceName)

      deviceStats.getNodeStatsList.asScala.zipWithIndex.foreach {
        case (nodeStats, nodeIndex) =>
          // Analyze tensor references to track data flow.
          val nodeName = nodeStats.getNodeName
          val startTime = nodeStats.getAllStartMicros
          val duration = nodeStats.getAllEndRelMicros
          val endTime = nodeStats.getAllStartMicros + duration
          val threadID = threadIDs(nodeIndex)

          nodeStats.getOutputList.asScala.zipWithIndex.foreach {
            case (output, index) =>
              val outputName = if (index == 0) nodeName else s"$nodeName:$index"
              val allocationDescription = output.getTensorDescription.getAllocationDescription
              val numBytes = allocationDescription.getRequestedBytes
              val allocatorName = allocationDescription.getAllocatorName
              val tensor = TensorTracker(outputName, tensorsPID, startTime, tensors.size, allocatorName, numBytes)
              tensors += outputName -> tensor
              tensor.addRef(startTime)
              tensor.addDeref(endTime)
              flowStarts = flowStarts.updated(outputName, (endTime, devicePID, threadID))
              if (showMemory) {
                val tensorDescription = output.getTensorDescription.toString.replace("\"", "")
                chromeTraceFormatter.emitObjectCreation(
                  "Tensor", outputName, startTime, tensorsPID, threadID, tensor.objectID)
                chromeTraceFormatter.emitObjectSnapshot(
                  "Tensor", tensor.name, endTime - 1, tensorsPID, threadID, tensor.objectID,
                  snapshot = Map("tensor_description" -> tensorDescription).asJson)
              }
          }
      }
    })

    stepStatistics.getDevStatsList.asScala.foreach(deviceStats => {
      val deviceName = deviceStats.getDevice
      val devicePID = devicePIDs(deviceName)

      // The following is `true` if this device is part of the GPU tracer logging.
      val isGPUTrace = deviceName.contains("/stream:") || deviceName.contains("/memcpy")

      deviceStats.getNodeStatsList.asScala.zipWithIndex.foreach {
        case (nodeStats, nodeIndex) =>
          val nodeName = nodeStats.getNodeName
          val threadID = threadIDs(nodeIndex)
          val startTime = nodeStats.getAllStartMicros
          val duration = nodeStats.getAllEndRelMicros
          val endTime = nodeStats.getAllStartMicros + duration

          // Emit an event to show op execution.
          val (name, op, inputs) = {
            if (isGPUTrace) {
              // Node names should always have the form "name:op".
              val fields = nodeName.split(':')
              (fields(0), fields(1), Seq.empty)
            } else if (nodeName == "RecvTensor") {
              // RPC tracing does not use the standard timeline label format.
              ("RecvTensor", "RecvTensor", Seq.empty)
            } else {
              val (op, inputs) = nodeStats.getTimelineLabel match {
                // Expects labels of the form: `name = op(arg, arg, ...)`.
                case opLabelRegex(_, o, i) if i == "" => (o, Seq.empty)
                case opLabelRegex(_, o, i) => (o, i.split(", ").toSeq)
                case _ => ("unknown", Seq.empty)
              }
              (nodeName, op, inputs)
            }
          }
          val arguments = Map("name" -> name.asJson, "op" -> op.asJson) ++
              inputs.zipWithIndex.map(i => s"input${i._2}" -> i._1.asJson)
          chromeTraceFormatter.emitRegion("Op", op, startTime, duration, devicePID, threadID, arguments)

          // Visualize the computation activity.
          inputs.foreach(inputName => {
            val name = {
              if (tensors.contains(inputName)) {
                inputName
              } else {
                // This can happen when partitioning has inserted a Send/Recv. We remove the numeric suffix so that the
                // data flow appears to come from the original node. Ideally, the step statistics would contain logging
                // for the Send and Recv nodes.
                val index = inputName.lastIndexOf("/_")
                if (index > 0) inputName.substring(0, index) else inputName
              }
            }

            if (tensors.contains(name)) {
              val tensor = tensors(name)
              tensor.addRef(startTime)
              tensor.addDeref(endTime - 1)

              if (showDataFlow) {
                // We use a different flow ID for every graph edge.
                val (createTime, createPID, createTID) = flowStarts(name)

                // We do not add flows when the producer and the consumer ops are on the same PID/TID since the
                // horizontal arrows clutter the visualization.
                if (createPID != devicePID || createTID != threadID) {
                  val flowID = nextFlowID
                  nextFlowID += 1
                  chromeTraceFormatter.emitFlowStart(inputName, createTime, createPID, createTID, flowID)
                  chromeTraceFormatter.emitFlowEnd(inputName, startTime, devicePID, threadID, flowID)
                }
              }
            } else {
              // TODO: Control dependencies currently fail here.
              logger.debug(s"Cannot find tensor '$inputName'. Maybe it was removed by the CSE.")
            }
          })
      }
    })

    // Produce a counter series for each memory allocator, if necessary.
    if (showMemory) {
      // Iterate over all tensor trackers to build a list of allocations and frees for each allocator. Then, sort the
      // lists and emit a cumulative counter series for each allocator.
      val allocations = mutable.Map.empty[String, mutable.ListBuffer[(Long, Long, String)]]
      tensors.foreach {
        case (name, tensor) =>
          chromeTraceFormatter.emitObjectDeletion(
            "Tensor", name, tensor.lastDerefTimestamp, tensor.processID, 0, tensor.objectID)
          val allocationsList = allocations.getOrElseUpdate(tensor.allocator, mutable.ListBuffer.empty)
          allocationsList.append((tensor.createTime, tensor.numBytes, name))
          allocationsList.append((tensor.lastDerefTimestamp, -tensor.numBytes, name))
      }

      // Generate a counter series showing total allocations for each allocator.
      allocations.foreach {
        case (allocator, allocationsList) =>
          var totalNumBytes = 0L
          allocationsList.sortBy(_._1).foreach {
            case (time, numBytes, _) =>
              totalNumBytes += numBytes
              chromeTraceFormatter.emitCounter(
                "Memory", allocator, time, allocatorsPID, allocator, totalNumBytes)
          }
      }
    }

    chromeTraceFormatter.toJsonString(prettyJson)
  }

  /** Helper class used for generating traces in the Chrome trace format.
    *
    * For details on the file format, please refer to
    * [[https://github.com/catapult-project/catapult/blob/master/tracing/README.md]].
    */
  private[Timeline] case class ChromeTraceFormatter() {
    case class ChromeTraceEvent(
        name: String,
        ph: String,
        cat: Option[String] = None,
        pid: Option[Int] = None,
        tid: Option[Int] = None,
        ts: Option[Long] = None,
        dur: Option[Long] = None,
        id: Option[Int] = None,
        args: Option[Map[String, Json]] = None)

    private val events  : mutable.ListBuffer[ChromeTraceEvent] = mutable.ListBuffer.empty
    private val metadata: mutable.ListBuffer[ChromeTraceEvent] = mutable.ListBuffer.empty

    /** Adds a process metadata event to the trace.
      *
      * @param  name      Event name.
      * @param  processID Identifier of the process.
      * @return This formatter after emitting the process metadata event.
      */
    def emitPID(name: String, processID: Int): ChromeTraceFormatter = {
      metadata += ChromeTraceEvent(
        name = "process_name",
        ph = "M",
        pid = Some(processID),
        args = Some(Map("name" -> name.asJson)))
      this
    }

    /** Adds a thread metadata event to the trace.
      *
      * @param  name      Event name.
      * @param  processID Identifier of the process.
      * @param  threadID  Identifier of the thread.
      * @return This formatter after emitting the thread metadata event.
      */
    def emitThread(name: String, processID: Int, threadID: Int): ChromeTraceFormatter = {
      metadata += ChromeTraceEvent(
        name = "thread_name",
        ph = "M",
        pid = Some(processID),
        tid = Some(threadID),
        args = Some(Map("name" -> name.asJson)))
      this
    }

    /** Adds a region event to the trace.
      *
      * @param  category  Event category.
      * @param  name      Event name.
      * @param  timestamp Start timestamp of this region.
      * @param  duration  Duration of this region.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  arguments Event arguments.
      * @return This formatter after emitting the region event.
      */
    def emitRegion(
        category: String,
        name: String,
        timestamp: Long,
        duration: Long,
        processID: Int,
        threadID: Int,
        arguments: Map[String, Json]
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "X",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        dur = Some(duration),
        args = Some(arguments))
      this
    }

    /** Adds an object creation event to the trace.
      *
      * @param  category  Event category.
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  objectID  Identifier of the object.
      * @return This formatter after emitting the object creation event.
      */
    def emitObjectCreation(
        category: String,
        name: String,
        timestamp: Long,
        processID: Int,
        threadID: Int,
        objectID: Int
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "N",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        id = Some(objectID))
      this
    }

    /** Adds an object deletion event to the trace.
      *
      * @param  category  Event category.
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  objectID  Identifier of the object.
      * @return This formatter after emitting the object deletion event.
      */
    def emitObjectDeletion(
        category: String,
        name: String,
        timestamp: Long,
        processID: Int,
        threadID: Int,
        objectID: Int
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "D",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        id = Some(objectID))
      this
    }

    /** Adds an object snapshot event to the trace.
      *
      * @param  category  Event category.
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  objectID  Identifier of the object.
      * @param  snapshot  Object snapshot.
      * @return This formatter after emitting the object snapshot event.
      */
    def emitObjectSnapshot(
        category: String,
        name: String,
        timestamp: Long,
        processID: Int,
        threadID: Int,
        objectID: Int,
        snapshot: Json
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "O",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        id = Some(objectID),
        args = Some(Map("snapshot" -> snapshot)))
      this
    }

    /** Adds a flow start event to the trace.
      *
      * When matched with a flow end event (with the same `flowID`) this will cause the trace viewer to draw an arrow
      * between the start and end events.
      *
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  flowID    Identifier of the flow.
      * @return This formatter after emitting the flow start event.
      */
    def emitFlowStart(
        name: String,
        timestamp: Long,
        processID: Int,
        threadID: Int,
        flowID: Int
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "s",
        cat = Some("DataFlow"),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        id = Some(flowID))
      this
    }

    /** Adds a flow end event to the trace.
      *
      * When matched with a flow start event (with the same `flowID`) this will cause the trace viewer to draw an arrow
      * between the start and end events.
      *
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  threadID  Identifier of the thread generating this event.
      * @param  flowID    Identifier of the flow.
      * @return This formatter after emitting the flow end event.
      */
    def emitFlowEnd(
        name: String,
        timestamp: Long,
        processID: Int,
        threadID: Int,
        flowID: Int
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "t",
        cat = Some("DataFlow"),
        pid = Some(processID),
        tid = Some(threadID),
        ts = Some(timestamp),
        id = Some(flowID))
      this
    }

    /** Adds a record for a single counter to the trace.
      *
      * @param  category     Event category.
      * @param  name         Event name.
      * @param  timestamp    Event timestamp.
      * @param  processID    Identifier of the process generating this event.
      * @param  counterName  Counter name.
      * @param  counterValue Counter value.
      * @return This formatter after emitting the counter record.
      */
    def emitCounter(
        category: String,
        name: String,
        timestamp: Long,
        processID: Int,
        counterName: String,
        counterValue: Long
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "C",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(0),
        ts = Some(timestamp),
        args = Some(Map(counterName -> counterValue.asJson)))
      this
    }

    /** Adds a record for multiple counters to the trace.
      *
      * @param  category  Event category.
      * @param  name      Event name.
      * @param  timestamp Event timestamp.
      * @param  processID Identifier of the process generating this event.
      * @param  counters  Map from counter names to counter values.
      * @return This formatter after emitting the counters record.
      */
    def emitCounters(
        category: String,
        name: String,
        timestamp: Long,
        processID: Int,
        counters: Map[String, Long]
    ): ChromeTraceFormatter = {
      events += ChromeTraceEvent(
        name = name,
        ph = "C",
        cat = Some(category),
        pid = Some(processID),
        tid = Some(0),
        ts = Some(timestamp),
        args = Some(counters.mapValues(_.asJson)))
      this
    }

    /** Formats the Chrome trace to a string.
      *
      * @param  pretty If `true`, produces human-readable JSON output.
      * @return JSON representation of this Chrome trace.
      */
    def toJsonString(pretty: Boolean = false): String = {
      val trace = Map("traceEvents" -> (metadata ++ events))
      if (pretty)
        trace.asJson.spaces4
      else
        trace.asJson.noSpaces
    }
  }

  /** Helper class to track the lifetime of a tensor.
    *
    * Note that this class is not thread safe and is intended only for internal use.
    *
    * @param  name       Name of the tensor.
    * @param  processID  Process identifier of the associated device.
    * @param  createTime Creation timestamp of this tensor.
    * @param  objectID   Chrome trace object identifier assigned for this tensor.
    * @param  allocator  Name of the allocator used to create the tensor.
    * @param  numBytes   Number of bytes allocated for this tensor.
    */
  private[Timeline] case class TensorTracker(
      name: String,
      processID: Int,
      createTime: Long,
      objectID: Int,
      allocator: String,
      numBytes: Long) {
    private val refTimestamps  : mutable.ListBuffer[Long] = mutable.ListBuffer.empty
    private val derefTimestamps: mutable.ListBuffer[Long] = mutable.ListBuffer.empty

    /** Returns the last dereference timestamp of this tensor. */
    def lastDerefTimestamp: Long = derefTimestamps.max

    /** Adds a reference to this tensor with the specified timestamp. */
    def addRef(timestamp: Long): TensorTracker = {
      refTimestamps += timestamp
      this
    }

    /** Adds a dereference to this tensor with the specified timestamp. */
    def addDeref(timestamp: Long): TensorTracker = {
      derefTimestamps += timestamp
      this
    }
  }
}
