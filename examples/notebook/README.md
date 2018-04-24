# [Toree](https://toree.apache.org/) 
The only [Jupyter Kernel](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) that supports both Scala and Python Language compiler backend, backedup by Apache.

However this comes with an extra setup i.e Spark intallation, 
which is worth your time!
Since you can use both the language frameworks in one Jupyter notebook.

 
# Jupyter Notebook Setup

- Step 1:
    - **Option 1:**
      - Download Spark binary and extract to known location and 
      update the same in SPARK_HOME environment variable
      - Link: https://spark.apache.org/downloads.html 
    - **Option 2:** 
       - Use following commands to build from scratch (not recomended)
        ```bash
        cd ~
        wget http://d3kbcqa49mib13.cloudfront.net/spark-2.1.0.tgz
        tar -xvzf ./spark-2.1.0.tgz
        cd spark-2.1.0/
        build/mvn -DskipTests clean package #I am skeptical at this step :)
        ```
- Step 2:
```bash
#Set appropriate path here
export SPARK_HOME=~/spark-2.1.0/
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$SPARK_HOME/python/lib
pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
jupyter toree install

#link the locally bundled/assembled TensorFlow Scala jar file to Toree kernel
export SPARK_OPTS="--master local[*] --jars ~/tensorflow_scala/examples/examples-assembly-0.1.jar"

jupyter notebook 
```


Reference Links
- https://toree.apache.org/
- http://blog.thedataincubator.com/2017/04/spark-2-0-on-jupyter-with-toree/
- https://gist.github.com/mikecroucher/b57a9e5a4c1a1a2045f30a901b186bdf
- https://github.com/apache/incubator-toree/blob/master/etc/examples/notebooks/magic-tutorial.ipynb
- https://github.com/asimjalis/apache-toree-quickstart

