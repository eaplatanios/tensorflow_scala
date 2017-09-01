scalaVersion := "2.12.3"
crossScalaVersions in ThisBuild := Seq("2.11.8", "2.12.3")

organization in ThisBuild := "org.platanios"
version in ThisBuild := "0.1"
licenses in ThisBuild := Seq("Apache License 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt"))

val tensorFlowVersion = "1.3.0"

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked"
)

logBuffered in Test := false

//scalacOptions in (ThisBuild, Compile, doc) ++= Opts.doc.externalAPI((
//  file(s"${(packageBin in Compile).value}") -> url("http://platanios.org/")) :: Nil)

lazy val loggingDependencies = Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",
  "ch.qos.logback" % "logback-classic" % "1.2.3"
)

lazy val all = (project in file("."))
    .aggregate(jni, api, data, examples, site)
    .settings(
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      packagedArtifacts in RootProject(file(".")) := Map.empty
    )

lazy val jni = (project in file("./jni"))
    .enablePlugins(JniNative, TensorFlowGenerateTensorOps, TensorFlowNativePackage)
    .settings(
      name := "tensorflow-jni",
      libraryDependencies ++= loggingDependencies,
      // Test dependencies
      libraryDependencies += "junit" % "junit" % "4.12",
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      // Tensor op code generation settings
      target in generateTensorOps := sourceDirectory.value / "main",
      ops in generateTensorOps := Map(
        "Basic" -> Seq(
          "ZerosLike", "OnesLike", "Fill", "Rank", "Size", "Shape", "ExpandDims", "Squeeze", "Pack", "ParallelConcat",
          "Unpack", "ConcatV2", "ConcatOffset", "Split", "SplitV", "Tile", "Pad", "MirrorPad", "Reshape", "Transpose",
          "InvertPermutation", "ReverseV2", "ReverseSequence", "SpaceToBatchND", "BatchToSpaceND", "SpaceToDepth",
          "DepthToSpace", "Where", "Unique", "UniqueWithCounts", "ListDiff", "GatherV2", "GatherNd", "ScatterNd",
          "Slice", "StridedSlice", "CheckNumerics", "EditDistance", "OneHot", "BroadcastArgs", "StopGradient",
          "PreventGradient", "Identity", "IdentityN", "ScatterNdNonAliasingAdd", "QuantizeAndDequantizeV3",
          "QuantizeV2", "Dequantize", "QuantizedConcat", "QuantizedReshape", "QuantizedInstanceNorm",
          "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsPerChannel"),
        "Math" -> Seq(
          "Select", "Range", "LinSpace", "Cast", "Bitcast", "AddN", "Abs", "ComplexAbs", "Neg", "Reciprocal", "Square",
          "Sqrt", "Rsqrt", "Exp", "Expm1", "Log", "Log1p", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh",
          "Tanh", "Asinh", "Acosh", "Atanh", "Lgamma", "Digamma", "Erf", "Erfc", "Sigmoid", "Sign", "Round", "Rint",
          "Floor", "Ceil", "IsNan", "IsInf", "IsFinite", "Add", "Sub", "Mul", "Div", "FloorDiv", "TruncateDiv",
          "RealDiv", "SquaredDifference", "Mod", "FloorMod", "TruncateMod", "Pow", "Igammac", "Igamma", "Zeta",
          "Polygamma", "Atan2", "Maximum", "Minimum", "Betainc", "LogicalNot", "LogicalAnd", "LogicalOr", "Equal",
          "NotEqual", "ApproximateEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "Sum", "Mean", "Prod", "Min",
          "Max", "All", "Any", "ArgMax", "ArgMin", "Bincount", "Cumsum", "Cumprod", "SegmentSum", "SegmentMean",
          "SegmentProd", "SegmentMin", "SegmentMax", "UnsortedSegmentSum", "UnsortedSegmentMax", "SparseSegmentSum",
          "SparseSegmentMean", "SparseSegmentSqrtN", "Diag", "DiagPart", "MatrixDiag", "MatrixSetDiag",
          "MatrixDiagPart", "MatrixBandPart", "MatMul", "BatchMatMul", "SparseMatMul", "Cross", "Complex", "Real",
          "Imag", "Angle", "Conj", "Bucketize", "QuantizedAdd", "QuantizedMul", "QuantizedMatMul",
          "QuantizeDownAndShrinkRange", "Requantize", "RequantizationRange", "CompareAndBitpack"),
        "NN" -> Seq(
          "BiasAdd", "Relu", "Relu6", "Elu", "Selu", "Softplus", "Softsign", "Softmax", "LogSoftmax", "L2Loss",
          "SoftmaxCrossEntropyWithLogits", "SparseSoftmaxCrossEntropyWithLogits", "TopKV2", "InTopKV2", "AvgPool",
          "AvgPool3D", "MaxPoolV2", "MaxPool3D", "MaxPoolWithArgmax", "FractionalAvgPool", "FractionalMaxPool",
          "Conv2D", "FusedResizeAndPadConv2D", "FusedPadConv2D", "DepthwiseConv2dNative", "Conv3D", "Dilation2D", "LRN",
          "BatchNormWithGlobalNormalization", "FusedBatchNorm", "QuantizedBiasAdd", "QuantizedRelu", "QuantizedRelu6",
          "QuantizedReluX", "QuantizedAvgPool", "QuantizedMaxPool", "QuantizedConv2D",
          "QuantizedBatchNormWithGlobalNormalization")
      ),
      scalaPackage in generateTensorOps := "tensors",
      // Native bindings compilation settings
      target in javah := sourceDirectory.value / "main" / "native" / "include",
      sourceDirectory in nativeCompile := sourceDirectory.value / "main" / "native",
      target in nativeCompile := target.value / "native" / nativePlatform.value,
      target in nativeCrossCompile := target.value / "native",
      nativePlatforms in nativeCrossCompile := Set("linux-x86_64", "darwin-x86_64"/*", windows-x86_64"*/),
      tensorFlowBinaryVersion in nativeCrossCompile := "nightly", // tensorFlowVersion,
      // Specify the order in which the different compilation tasks are executed
      // nativeCompile := nativeCompile.dependsOn(generateTensorOps).value,
      compile in Compile := (compile in Compile).dependsOn(nativeCompile).value
    )

lazy val api = (project in file("./api"))
    .dependsOn(jni)
    .enablePlugins(ProtobufPlugin)
    .settings(publishSettings)
    .settings(
      name := "tensorflow-api",
      libraryDependencies ++= loggingDependencies,
      libraryDependencies += "org.typelevel" %% "spire" % "0.14.1",
      libraryDependencies += "org.tensorflow" % "proto" % tensorFlowVersion,
      libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2",
      // Test dependencies
      libraryDependencies += "junit" % "junit" % "4.12",
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      // Protobuf settings
      version in ProtobufConfig := "3.4.0",
      sourceDirectory in ProtobufConfig := sourceDirectory.value / "main" / "proto",
      javaSource in ProtobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
      sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java",
      unmanagedResourceDirectories in Compile += (sourceDirectory in ProtobufConfig).value
    )

lazy val data = (project in file("./data"))
    .dependsOn(api)
    .settings(
      name := "data",
      libraryDependencies ++= loggingDependencies,
      // Test dependencies
      libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1",
      libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
    )

lazy val examples = (project in file("./examples"))
    .dependsOn(api, data)
    .settings(
      name := "examples",
      libraryDependencies ++= loggingDependencies
    )

lazy val site = (project in file("./site"))
    .dependsOn(api)
    .enablePlugins(ScalaUnidocPlugin, MicrositesPlugin)
    .settings(publishSettings)
    .settings(noPublishSettings)
    .settings(
      name := "tensorflow-site",
      autoAPIMappings := true,
      siteSubdirName in ScalaUnidoc := "api",
      unidocProjectFilter in (ScalaUnidoc, unidoc) := inProjects(jni, api, data, examples),
      addMappingsToSiteDir(mappings in (ScalaUnidoc, packageDoc), siteSubdirName in ScalaUnidoc),
      ghpagesNoJekyll := false,
      fork in (ScalaUnidoc, unidoc) := true,
      scalacOptions in (ScalaUnidoc, unidoc) ++= Seq(
        //"-Xfatal-warnings",
        "-doc-source-url", scmInfo.value.get.browseUrl + "/tree/master€{FILE_PATH}.scala",
        "-sourcepath", baseDirectory.in(LocalRootProject).value.getAbsolutePath,
        // "=diagrams",
        "-groups",
        "-implicits-show-all"
      ),
      // libraryDependencies += "org.scalameta" %% "scalameta" % "1.8.0" % Provided,
      // libraryDependencies += "org.scalameta" %% "contrib" % "1.8.0",
      tutSourceDirectory := (sourceDirectory in Compile).value / "site",
      fork in tut := true,
      scalacOptions in Tut ~= (_.filterNot(Set("-Ywarn-unused-import", "-Ywarn-dead-code"))),
      micrositeName := "TensorFlow for Scala",
      micrositeDescription := "Scala API for TensorFlow",
      micrositeBaseUrl := "/tensorflow_scala",
      micrositeDocumentationUrl := "/tensorflow_scala/docs",
      micrositeAuthor := "Emmanouil Antonios Platanios",
      micrositeHomepage := "http://eaplatanios.github.io/tensorflow_scala/",
      micrositeOrganizationHomepage := "http://eaplatanios.github.io",
      micrositeGithubOwner := "eaplatanios",
      micrositeGithubRepo := "tensorflow_scala",
      micrositePushSiteWith := GHPagesPlugin,
      micrositeGitterChannel := false,
      micrositeHighlightTheme := "hybrid",
      micrositeImgDirectory := (resourceDirectory in Compile).value / "site" / "img",
      micrositeCssDirectory := (resourceDirectory in Compile).value / "site" / "css",
      micrositeJsDirectory := (resourceDirectory in Compile).value / "site" / "js",
      micrositePalette := Map(
        "brand-primary"     -> "#E05236",
        "brand-secondary"   -> "#455A64",
        "brand-tertiary"    -> "#39474E", // "#303C42",
        "gray-dark"         -> "#453E46",
        "gray"              -> "#837F84",
        "gray-light"        -> "#E3E2E3",
        "gray-lighter"      -> "#F4F3F4",
        "white-color"       -> "#FFFFFF"),
      includeFilter in makeSite := "*.html" | "*.css" | "*.png" | "*.jpg" | "*.gif" | "*.js" | "*.swf" | "*.yml" | "*.md" | "*.svg",
      includeFilter in Jekyll := (includeFilter in makeSite).value
    )

lazy val noPublishSettings = Seq(
  publish := (),
  publishLocal := (),
  publishArtifact := false
)

lazy val publishSettings = Seq(
  homepage := Some(url("https://github.com/eaplatanios/tensorflow_scala")),
  licenses := Seq("Apache License 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt")),
  scmInfo := Some(ScmInfo(url("https://github.com/eaplatanios/tensorflow_scala"),
                          "scm:git:git@github.com:eaplatanios/tensorflow_scala.git")),
  autoAPIMappings := true,
  apiURL := Some(url("http://eaplatanios.github.io/tensorflow_scala/api/"))
)
