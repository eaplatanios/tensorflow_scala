package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.META_GRAPH_UNBOUND_INPUT_PREFIX
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

import com.google.protobuf.ByteString
import org.tensorflow.framework.CollectionDef.{BytesList, NodeList}
import org.tensorflow.framework.MetaGraphDef.MetaInfoDef
import org.tensorflow.framework._
import org.tensorflow.util.SaverDef

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.runtime.universe.{typeOf, TypeTag}
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] object MetaGraph {
  private[this] val nodeDefNamePrefixRegex: Regex = "^\\^+".r
  private[this] val nodeDefRenameRegex    : Regex = "([\\^]|^)(.*)".r

  /** Constructs and returns a [[MetaGraphDef]] object using the provided arguments.
    *
    * @param  graph       Graph for which the [[MetaGraphDef]] is being constructed (necessary for obtaining the
    *                     specified collections).
    * @param  metaInfoDef [[MetaInfoDef]] associated with the [[MetaGraphDef]] that will be constructed.
    * @param  graphDef    [[GraphDef]] associated with the [[MetaGraphDef]] that will be constructed.
    * @param  saverDef    [[SaverDef]] associated with the [[MetaGraphDef]] that will be constructed.
    * @param  collections Graph collection keys specifying the collections to include in the [[MetaGraphDef]].
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name
    *                     scope will be included in the resulting ProtoBuf object and the export scope will be
    *                     stripped from their names to allow for easy import into new name scopes.
    * @return Constructed [[MetaGraphDef]].
    */
  private[api] def apply(
      graph: Graph, metaInfoDef: MetaInfoDef = null, graphDef: GraphDef = null, saverDef: SaverDef = null,
      collections: Set[Graph.Key[_]] = Set.empty, exportScope: String = null): MetaGraphDef = {
    val metaGraphDefBuilder = MetaGraphDef.newBuilder()

    // Add the graph definition.
    metaGraphDefBuilder.setGraphDef(if (graphDef == null) graph.toProto else graphDef)

    // Add the meta information.
    val metaInfoDefBuilder = if (metaInfoDef == null) MetaInfoDef.newBuilder() else MetaInfoDef.newBuilder(metaInfoDef)
    metaInfoDefBuilder.setTensorflowVersion(NativeLibrary.version)
    metaGraphDefBuilder.mergeMetaInfoDef(metaInfoDefBuilder.build())

    // Add the saver information.
    if (saverDef != null)
      metaGraphDefBuilder.mergeSaverDef(saverDef)

    // Add the collections.
    if (collections != null)
      collections.foreach(key => addCollectionDef(metaGraphDefBuilder, key, graph, exportScope))

    metaGraphDefBuilder.build()
  }

  /** Adds a collection named `name` in a [[MetaGraphDef.Builder]].
    *
    * @param  metaGraphDefBuilder [[MetaGraphDef.Builder]] in which to add the collection.
    * @param  key                 Collection key.
    * @param  graph               Graph from which to obtain the collections.
    * @param  exportScope         Optional string specifying the name scope to remove. Only the ops within this name
    *                             scope will be included in the resulting ProtoBuf object and the export scope will be
    *                             stripped from their names to allow for easy import into new name scopes.
    * @return Updated [[MetaGraphDef.Builder]].
    */
  private[this] def addCollectionDef[K: TypeTag](
      metaGraphDefBuilder: MetaGraphDef.Builder, key: Graph.Key[K], graph: Graph,
      exportScope: String = null): MetaGraphDef.Builder = {
    val values = graph.getCollection(key)
    val collectionDef = typeOf[K] match {
      case t if t =:= typeOf[String] =>
        val bytesListBuilder = {
          if (metaGraphDefBuilder.containsCollectionDef(key.name))
            BytesList.newBuilder(metaGraphDefBuilder.getCollectionDefOrThrow(key.name).getBytesList)
          else
            BytesList.newBuilder()
        }
        values.map(_.asInstanceOf[String]).foreach(s => bytesListBuilder.addValue(ByteString.copyFromUtf8(s)))
        CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
      case t if t =:= typeOf[Op] =>
        val nodeListBuilder = {
          if (metaGraphDefBuilder.containsCollectionDef(key.name))
            NodeList.newBuilder(metaGraphDefBuilder.getCollectionDefOrThrow(key.name).getNodeList)
          else
            NodeList.newBuilder()
        }
        values.asInstanceOf[Set[Op]]
            .filter(o => shouldIncludeNode(o.name, exportScope))
            .filter(o => exportScope == null || o.name.startsWith(exportScope)).foreach(o => {
          nodeListBuilder.addValue(Op.stripNameScope(exportScope, o.name))
        })
        CollectionDef.newBuilder().setNodeList(nodeListBuilder.build()).build()
      case t if t =:= typeOf[Op.Output] =>
        val nodeListBuilder = {
          if (metaGraphDefBuilder.containsCollectionDef(key.name))
            NodeList.newBuilder(metaGraphDefBuilder.getCollectionDefOrThrow(key.name).getNodeList)
          else
            NodeList.newBuilder()
        }
        values.asInstanceOf[Set[Op.Output]]
            .filter(o => shouldIncludeNode(o.name, exportScope))
            .filter(o => exportScope == null || o.name.startsWith(exportScope)).foreach(o => {
          nodeListBuilder.addValue(Op.stripNameScope(exportScope, o.name))
        })
        CollectionDef.newBuilder().setNodeList(nodeListBuilder.build()).build()
      case t if t =:= typeOf[Variable] =>
        val bytesListBuilder = {
          if (metaGraphDefBuilder.containsCollectionDef(key.name))
            BytesList.newBuilder(metaGraphDefBuilder.getCollectionDefOrThrow(key.name).getBytesList)
          else
            BytesList.newBuilder()
        }
        values.asInstanceOf[Set[Variable]].filter(v => shouldIncludeNode(v.name, exportScope)).foreach(v => {
          bytesListBuilder.addValue(v.toProto(exportScope).toByteString)
        })
        CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
      case t => throw new IllegalArgumentException(s"Cannot serialize collection with type '$t'.")
    }
    metaGraphDefBuilder.putCollectionDef(key.name, collectionDef)
    metaGraphDefBuilder
  }

  /** Returns `true` if a node should be included.
    *
    * @param  name        Node name.
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Boolean value indicating whether the node with the provided name should be included.
    */
  private[this] def shouldIncludeNode(name: String, exportScope: String = null): Boolean = {
    name.startsWith(META_GRAPH_UNBOUND_INPUT_PREFIX) || exportScope == null || name.startsWith(exportScope)
  }

  /** Processes a node definition according the provided arguments and returns a new node definition.
    *
    * @param nodeDef       Node definition to process.
    * @param exportScope   Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                      be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                      names to allow for easy import into new name scopes.
    * @param unboundInputs Set containing unbound input names if they exist.
    * @param clearDevices  Boolean value indicating whether to clear the device information from the returned node
    *                      definition.
    * @return New processed node definition.
    */
  private[this] def processNodeDef(
      nodeDef: NodeDef, exportScope: String = null, unboundInputs: mutable.Set[String] = mutable.Set.empty,
      clearDevices: Boolean = false): NodeDef = {
    val nodeDefBuilder = NodeDef.newBuilder(nodeDef)
    nodeDefBuilder.setName(Op.stripNameScope(exportScope, nodeDef.getName))
    val numberOfInputs = nodeDef.getInputCount
    var inputIndex = 0
    while (inputIndex < numberOfInputs) {
      val input = nodeDef.getInput(inputIndex)
      if (exportScope != null && nodeDefNamePrefixRegex.pattern.matcher(input).replaceAll("").startsWith(exportScope)) {
        // Add a prefix to the unbound name so that they are easily identifiable.
        val newInput = nodeDefRenameRegex.pattern.matcher(input).replaceFirst(s"$$1$META_GRAPH_UNBOUND_INPUT_PREFIX$$2")
        nodeDefBuilder.setInput(inputIndex, newInput)
        unboundInputs += newInput
      } else {
        nodeDefBuilder.setInput(inputIndex, Op.stripNameScope(exportScope, input))
      }
      inputIndex += 1
    }
    val attributes = nodeDef.getAttrMap.asScala
    for ((name, value) <- attributes) {
      if (name == "_class") {
        val values = value.getList.getSList.asScala
            .filter(exportScope == null || _.toStringUtf8.split("@")(1).startsWith(exportScope))
            .map(v => ByteString.copyFromUtf8(Op.stripNameScope(exportScope, v.toStringUtf8)))
        nodeDefBuilder.putAttr(
          name, AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addAllS(values.asJava)).build())
      } else {
        nodeDefBuilder.putAttr(name, AttrValue.newBuilder(value).build())
      }
    }
    if (clearDevices)
      nodeDefBuilder.setDevice("")
    nodeDefBuilder.build()
  }
}
