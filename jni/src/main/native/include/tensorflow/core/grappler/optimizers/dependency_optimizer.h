/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Optimize TF computations by removing control dependencies or re-arranging
// them to shorten the critical path for a model step or enable other
// optimizations, such as removing nodes that are effectively noops.
class DependencyOptimizer : public GraphOptimizer {
 public:
  DependencyOptimizer() {}
  explicit DependencyOptimizer(RewriterConfig::Toggle /*unused*/) {}
  ~DependencyOptimizer() override {}

  string name() const override { return "dependency_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  // Returns true if it is safe to convert node to NoOp.
  bool SafeToConvertToNoOp(const NodeDef& node);
  // Removes all duplicate control dependencies.
  void CleanControlInputs();
  // Builds a map from the &optimized_graph_->node(i) to i.
  void BuildNodeToIdx();
  // Removes the given set of nodes from the graph.
  void DeleteNodes(const std::set<int>& nodes_to_delete);
  // Tries to optimize the node with the given index, possibly additional
  // optimizations by inserting nodes in nodes_to_simplify, and pruning nodes by
  // inserting them in nodes_to_delete.
  void OptimizeNode(int node_idx, SetVector<int>* nodes_to_simplify,
                    std::set<int>* nodes_to_delete);
  // Eliminates redundant control dependencies by computing the transitive
  // reduction of the graph.
  Status TransitiveReduction();
  // Main driver of dependency optimizations.
  Status OptimizeDependencies();

  bool fetch_nodes_known_;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<NodeMap> node_map_;
  std::unordered_map<const NodeDef*, int> node_to_idx_;
  GraphDef* optimized_graph_;  // Not owned.
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
