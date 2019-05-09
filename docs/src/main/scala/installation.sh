// #clone_repository
git clone https://github.com/tensorflow/tensorflow.git <repository_directory>
cd <repository_directory>
git checkout 1aaa68d93c6b2f4151446eb211399b4330c96a09
// #clone_repository

// #compile_tf
./configure
bazel build --config=opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow.so
// #compile_tf

// #apt_get_install_protobuf
apt-get install protobuf-compiler
// #apt_get_install_protobuf

// #brew_install_protobuf
brew install protobuf
// #brew_install_protobuf
