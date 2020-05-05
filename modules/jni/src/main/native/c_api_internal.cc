///* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//#include "tensorflow/c/c_api.h"
//#include "tensorflow/c/tf_status.h"
//#include "tensorflow/core/graph/validate.h"
//#include "tensorflow/core/platform/mutex.h"
//
////using tensorflow::ExtendSessionGraphHelper;
//
//namespace tensorflow {
//
//void RecordMutation(TF_Graph* graph, const TF_Operation& op,
//                    const char* mutation_type) {
//  // If any session has already run this node_id, mark this session as
//  // unrunnable.
//  for (auto it : graph->sessions) {
//    mutex_lock session_lock(it.first->mu);
//    if (it.first->last_num_graph_nodes > op.node.id()) {
//      it.second = strings::StrCat(
//          "Operation '", op.node.DebugString(), "' was changed by ",
//          mutation_type,
//          " after it was run by a session. This mutation will have no effect, "
//          "and will trigger an error in the future. Either don't modify "
//          "nodes after running them or create a new session.");
//    }
//  }
//}
//
//// TODO(josh11b,mrry): Change Session to be able to use a Graph*
//// directly, instead of requiring us to serialize to a GraphDef and
//// call Session::Extend().
//bool ExtendSessionGraphHelper(TF_Session* session, TF_Status* status) {
//  if (session->graph != nullptr) {
//    // Take the graph lock before the session lock to avoid deadlock. This is
//    // safe since session->graph does not change.
//    session->graph->mu.lock();
//    mutex_lock session_lock(session->mu);
//    const Graph& graph = session->graph->graph;
//
//    const string& mutation_warning = session->graph->sessions[session];
//    if (!mutation_warning.empty()) {
//      // TODO(b/74949947): turn this back into an error status
//      LOG(WARNING) << mutation_warning;
//      session->graph->sessions[session].clear();
//    }
//
//    const auto num_nodes = graph.num_node_ids();
//    if (session->last_num_graph_nodes < num_nodes) {
//      // TODO(nolivia): check this on a subset of the graph instead of all of
//      // it.
//      status->status = graph::ValidateGraphHasNoCycle(session->graph->graph);
//      if (!status->status.ok()) {
//        session->graph->mu.unlock();
//        return false;
//      }
//
//      GraphDef graph_def;
//      *graph_def.mutable_versions() = graph.versions();
//      // Fill graph_def with nodes with ids in the range
//      // [session->last_num_graph_nodes, num_nodes), that is the nodes
//      // added since the last TF_SessionRun() call.
//      for (auto id = session->last_num_graph_nodes; id < num_nodes; ++id) {
//        Node* const node = graph.FindNodeId(id);
//        if (node != nullptr && node->IsOp()) {
//          NodeDef* const node_def = graph_def.add_node();
//          *node_def = node->def();
//        }
//      }
//      *graph_def.mutable_library() = graph.flib_def().ToProto();
//      session->graph->mu.unlock();
//      status->status = session->session->Extend(std::move(graph_def));
//      if (!status->status.ok()) {
//        // Contract is we always delete input_values[i].
//        return false;
//      }
//      // Note: session->session is not modified if Extend() fails, so
//      // we only set last_num_graph_nodes if it succeeds.
//      session->last_num_graph_nodes = num_nodes;
//    } else {
//      session->graph->mu.unlock();
//    }
//  }
//  return true;
//}
//
//}  // namespace tensorflow
