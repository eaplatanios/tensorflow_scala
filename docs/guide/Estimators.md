# Estimators

## Models

A `Model` needs to implement one or more of the following methods,
depending on their type:
  - **Inference Models** only need to implement a method for building
    all ops used for inference:
    ```scala
    abstract class InferenceModel[In, Out] extends Model {
      def buildInferOps(): Model.InferOps[In, Out]
    }
    ```
  - **Trainable Models** also need to implement a method for building
    all ops used for training, and one for building all ops used for
    evaluation:
    ```scala
    class TrainableModel[In, TrainIn, Out, TrainOut, Loss, EvalIn] extends InferenceModel[In, Out] {
      def buildTrainOps(): Model.TrainOps[TrainIn, TrainOut, Loss]
      def buildEvalOps(metrics: Seq[Metric[EvalIn, Output[Float]]]): Model.EvalOps[TrainIn, Out]
    }

    class SupervisedTrainableModel[In, TrainIn, Out, TrainOut, Loss] extends TrainableModel[In, (In, TrainIn), Out, TrainOut, Loss, (Out, (In, TrainIn))] {
      override def buildTrainOps(): Model.TrainOps[(In, TrainIn), TrainOut, Loss]
      override def buildEvalOps(metrics: Seq[Metric[(Out, (In, TrainIn)), Output[Float]]]): Model.EvalOps[(In, TrainIn), Out]
    }

    class UnsupervisedTrainableModel[In, Out, Loss] extends TrainableModel[In, In, Out, Out, Loss, Out] {
      override def buildTrainOps(): Model.TrainOps[In, Out, Loss]
      override def buildEvalOps(metrics: Seq[Metric[Out, Output[Float]]]): Model.EvalOps[In, Out]
    }
    ```

### Type Parameters

The type parameters of these classes can be broadly interpreted as follows:
  - `In`: Input type of the model, at inference time (i.e., when the model is 
    being used/deployed).
  - `TrainIn`: Input type of the model, at training time (e.g., this could be 
    the model inference inputs along with supervision labels).
  - `Out`: Output type of the model, at inference time.
  - `TrainOut`: Output type of the model, at training time. Note that this is 
    useful to keep separate from `Out`. Fro example, during training a model may 
    output a distribution over some labels whereas during inference it may 
    output the single label with the highest probability.
  - `Loss`: Type of the loss function value (e.g., `Float`).
  - `EvalIn`: Input type of the model, at evaluation time.

### Ops Classes

Here we describe the ops classes that the `buildXXXOps()` methods defined 
above return. In summary, these classes are wrappers over constructed ops that 
can be used to perform inference, training, and evaluation, using models.

The `InferOps` class contains all ops that are used for inference:
  - `inputIterator: DatasetIterator[In]`: Dataset iterator.
  - `input: In`: Retrieved element from the dataset iterator.
  - `output: Out`: Model inference output.

The `TrainOps` class contains all ops that are used for training:
  - `inputIterator: DatasetIterator[TrainIn]`: Train dataset iterator.
  - `input: TrainIn`: Retrieved element from the dataset iterator.
  - `output: TrainOut`: Model training output.
  - `loss: Output[Loss]`: Scalar tensor containing the loss value.
  - `gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])]`:
    Gradients of the loss along with their corresponding variables.
  - `trainOp: UntypedOp`: Op that when executed performs a single
    training step (e.g., a gradient descent step for a single batch).

The `EvalOps` class contains all ops that are used for evaluation:
  - `inputIterator: DatasetIterator[In]`: Dataset iterator.
  - `input: In`: Retrieved element from the dataset iterator.
  - `output: Out`: Model evaluation output.
  - `metricValues: Seq[Output[Float]]`: Metric value ops.
  - `metricUpdates: Seq[Output[Float]]`: Metric update ops.
  - `metricResets: Set[UntypedOp]`: Metric reset ops.
Note that the metrics have value, update, and reset ops, because
estimators only use streaming versions of the metrics.
