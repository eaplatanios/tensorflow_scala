import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.tf.learn._

// #inference_model
trait InferenceModel[In, Out] extends Model {
  def buildInferOps(): Model.InferOps[In, Out]
}
// #inference_model

// #trainable_models
trait TrainableModel[In, TrainIn, Out, TrainOut, Loss, EvalIn] extends InferenceModel[In, Out] {
  def buildTrainOps(): Model.TrainOps[TrainIn, TrainOut, Loss]
  def buildEvalOps(metrics: Seq[Metric[EvalIn, Output[Float]]]): Model.EvalOps[TrainIn, Out]
}

trait SupervisedTrainableModel[In, TrainIn, Out, TrainOut, Loss] extends TrainableModel[In, (In, TrainIn), Out, TrainOut, Loss, (Out, (In, TrainIn))] {
  override def buildTrainOps(): Model.TrainOps[(In, TrainIn), TrainOut, Loss]
  override def buildEvalOps(metrics: Seq[Metric[(Out, (In, TrainIn)), Output[Float]]]): Model.EvalOps[(In, TrainIn), Out]
}

trait UnsupervisedTrainableModel[In, Out, Loss] extends TrainableModel[In, In, Out, Out, Loss, Out] {
  override def buildTrainOps(): Model.TrainOps[In, Out, Loss]
  override def buildEvalOps(metrics: Seq[Metric[Out, Output[Float]]]): Model.EvalOps[In, Out]
}
// #trainable_models
