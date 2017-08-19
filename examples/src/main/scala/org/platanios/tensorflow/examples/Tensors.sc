import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api._
import org.slf4j.LoggerFactory

val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))

val session = tf.Session()

val myTensor = tf.zeros(Shape(3, 4), INT32)

println(session.run(fetches = myTensor).toOutput)
println(session.run(fetches = myTensor).toString)


val myVar = tf.variable("myVar", FLOAT32, Shape(3, 4), tf.zerosInitializer)

session.run(targets = tf.globalVariablesInitializer())

println()
//println(session.run(fetches = myVar))