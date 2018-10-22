# Graph Collections

## Creating a new graph collection key

1. Create a key that extends the `Graph.Key[T]` trait, where `T` is the
   type of values the corresponding collection holds.

   Note that, given the complexity that may be involved in serializing
   certain value types, a few helper sub-traits of `Graph.Key[T]` are
   provided, that you can extend and avoid having to write the
   serialization code. Those are:

   - `StringCollectionKey extends Key[String]`
   - `IntCollectionKey extends Key[Int]`
   - `OpCollectionKey extends Key[Op]`
   - `OutputCollectionKey extends Key[Output]`
   - `VariableCollectionKey extends Key[Variable]`
   - `SaverCollectionKey extends Key[Saver]`
   - `ResourceCollectionKey extends Key[Resource]`

   If extending one of these traits, your key implementation can look as
   simple as this:

   ```scala
   object GLOBAL_VARIABLES extends VariableCollectionKey {
     override def name: String = "variables"
   }
   ```

   Note that the helper sub-traits are also compatible with
   loading/saving from/to code that uses the TensorFlow Python API.

2. Register the new key in a static code block that will be called when
   your library is coded. For example, if you are contributing within
   the TensorFlow for Scala API package, add a call such as:

   `Graph.Keys.register(Graph.Keys.GLOBAL_VARIABLES)`

   in the `api` package object (i.e.,
   `org/platanios/tensorflow/api/package.scala`).
