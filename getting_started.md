# Getting Started

Tensors are the main data structure used in TensorFlow and are thus a good place to start.

## Tensors




A tensor is the main data structure used in TensorFlow. It represents a multi-dimensional array and can hold elements of 
various data types. For example, the following code creates an integer tensor filled with zeros with shape `[2, 5]` 
(i.e., a two-dimensional array holding integer values, where the first dimension size is 2 and the second is 5):
```scala
scala> val tensor = Tensor.zeros(INT32, Shape(2, 5))
tensor: org.platanios.tensorflow.api.tensors.Tensor = INT32[2, 5]
```
You can print the contents of a tensor as follows:
```scala
scala> tensor.summarize()
res0: String =
INT32[2, 5]
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
```


## Graph

