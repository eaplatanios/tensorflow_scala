/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import ReleaseTransformations._
import TensorFlowNativePackage._
import sbtrelease.Vcs

scalaVersion in ThisBuild := "2.12.3"
crossScalaVersions in ThisBuild := Seq("2.11.8", "2.12.3")

organization in ThisBuild := "org.platanios"

val tensorFlowVersion = "1.3.0"

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked",
  // "-Xfatal-warnings",
  "-Yno-adapted-args",
  "-Ywarn-dead-code",
  // "-Ywarn-numeric-widen",
  // "-Ywarn-value-discard",
  "-Xfuture"
)

//scalacOptions in (ThisBuild, Compile, doc) ++= Opts.doc.externalAPI((
//  file(s"${(packageBin in Compile).value}") -> url("http://platanios.org/")) :: Nil)

lazy val loggingSettings = Seq(
  libraryDependencies ++= Seq(
    "com.typesafe.scala-logging" %% "scala-logging"   % "3.7.2",
    "ch.qos.logback"             %  "logback-classic" % "1.2.3")
)

lazy val testSettings = Seq(
  libraryDependencies ++= Seq(
    "junit"         %  "junit" %   "4.12",
    "org.scalactic" %% "scalactic" % "3.0.1",
    "org.scalatest" %% "scalatest" % "3.0.1" % "test"),
  logBuffered in Test := false,
  fork in test := true,
  testForkedParallel in Test := true,
  parallelExecution in Test := true,
  testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oDF")
)

lazy val all = (project in file("."))
    .aggregate(jni, api, data, examples, site)
    .settings(moduleName := "tensorflow", name := "TensorFlow for Scala")
    .settings(publishSettings)
    .settings(
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      packagedArtifacts := (packagedArtifacts in api).value ++ (packagedArtifacts in jni).value,
      nativeCompile := Seq.empty,
      nativeCrossCompile in CrossCompile := Map.empty,
      commands ++= Seq(
        publishCrossCompiled,
        publishLocalCrossCompiled)
    )

lazy val jni = (project in file("./jni"))
    .enablePlugins(JniNative, TensorFlowGenerateTensorOps, TensorFlowNativePackage)
    .configs(CrossCompile)
    .settings(moduleName := "tensorflow-jni", name := "TensorFlow for Scala JNI Bindings")
    .settings(loggingSettings)
    .settings(testSettings)
    .settings(noPublishSettings)
    .settings(
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
      target in CrossCompile := target.value / "native",
      nativePlatforms in CrossCompile := Set(LINUX_x86_64, DARWIN_x86_64/*", WINDOWS_x86_64"*/),
      tensorFlowBinaryVersion in CrossCompile := "nightly", // tensorFlowVersion
      // Specify the order in which the different compilation tasks are executed
      nativeCompile := nativeCompile.dependsOn(generateTensorOps).value,
      publishArtifact := true
    )

lazy val api = (project in file("./api"))
    .dependsOn(jni)
    .enablePlugins(ProtobufPlugin)
    .settings(moduleName := "tensorflow-api", name := "TensorFlow for Scala API")
    .settings(loggingSettings)
    .settings(testSettings)
    .settings(noPublishSettings)
    .settings(
      libraryDependencies += "org.typelevel" %% "spire" % "0.14.1",
      libraryDependencies += "org.tensorflow" % "proto" % tensorFlowVersion,
      libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2",
      // Protobuf settings
      version in ProtobufConfig := "3.4.0",
      sourceDirectory in ProtobufConfig := sourceDirectory.value / "main" / "proto",
      javaSource in ProtobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
      sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java",
      unmanagedResourceDirectories in Compile += (sourceDirectory in ProtobufConfig).value,
      publishArtifact := true
    )

lazy val data = (project in file("./data"))
    .dependsOn(api)
    .settings(moduleName := "tensorflow-data", name := "TensorFlow for Scala Data")
    .settings(loggingSettings)
    .settings(testSettings)
    .settings(noPublishSettings)

lazy val examples = (project in file("./examples"))
    .dependsOn(api, data)
    .settings(moduleName := "tensorflow-examples", name := "TensorFlow for Scala Examples")
    .settings(loggingSettings)
    .settings(noPublishSettings)

lazy val site = (project in file("./site"))
    .dependsOn(api)
    .enablePlugins(ScalaUnidocPlugin, MicrositesPlugin)
    .settings(moduleName := "tensorflow-site", name := "TensorFlow for Scala Site")
    .settings(publishSettings)
    .settings(noPublishSettings)
    .settings(
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
      includeFilter in makeSite :=
          "*.html" | "*.css" | "*.png" | "*.jpg" | "*.gif" | "*.js" | "*.swf" | "*.yml" | "*.md" | "*.svg",
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
  developers := List(
    Developer(
      id="eaplatanios",
      name="Emmanouil Antonios Platanios",
      email="e.a.platanios@gmail.com",
      url=url("http://platanios.org/"))
  ),
  autoAPIMappings := true,
  apiURL := Some(url("http://eaplatanios.github.io/tensorflow_scala/api/")),
  releaseCrossBuild := true,
  releaseTagName := s"v${if (releaseUseGlobalVersion.value) (version in ThisBuild).value else version.value}",
  releaseVersionBump := sbtrelease.Version.Bump.Next,
  releasePublishArtifactsAction := PgpKeys.publishSigned.value,
  releaseVcs := Vcs.detect(baseDirectory.value),
  releaseVcsSign := true,
  releaseIgnoreUntrackedFiles := true,
  useGpg := true,  // Bouncy Castle has bugs with sub-keys, so we use gpg instead
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := Function.const(false),
  publishTo := Some(
    if (isSnapshot.value)
      Opts.resolver.sonatypeSnapshots
    else
      Opts.resolver.sonatypeStaging
  ),
  releaseProcess := Seq[ReleaseStep](
    checkSnapshotDependencies,
    inquireVersions,
    runClean,
    runTest,
    ReleaseStep(reapply(Seq(
      nativeCompile := Seq.empty,
      nativeCrossCompile in CrossCompile := (nativeCrossCompile in CrossCompile).dependsOn(nativeCompile).value,
      resourceGenerators in Compile += Def.task {
        jniLibraries((nativeCrossCompile in CrossCompile).value, (resourceManaged in Compile).value)
      }.taskValue
    ), _)),
    setReleaseVersion,
    commitReleaseVersion,
    tagRelease,
    releaseStepCommand("publishSigned"),
    setNextVersion,
    commitNextVersion,
    ReleaseStep(action = Command.process("sonatypeReleaseAll", _), enableCrossBuild = true),
    pushChanges
  ),
  // For Travis CI - see http://www.cakesolutions.net/teamblogs/publishing-artefacts-to-oss-sonatype-nexus-using-sbt-and-travis-ci
  credentials ++= (for {
    username <- Option(System.getenv().get("SONATYPE_USERNAME"))
    password <- Option(System.getenv().get("SONATYPE_PASSWORD"))
  } yield Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", username, password)).toSeq
)

lazy val publishCrossCompiled = Command.command("publishCrossCompiled") { state =>
  val newState = reapply(Seq(
    nativeCompile in jni := Seq.empty,
    nativeCrossCompile in CrossCompile in jni := {
      (nativeCrossCompile in CrossCompile in jni).dependsOn(nativeCompile in jni).value
    },
    resourceGenerators in Compile in jni += Def.task {
      jniLibraries((nativeCrossCompile in CrossCompile in jni).value, (resourceManaged in Compile in jni).value)
    }.taskValue
  ), state)
  val extracted = Project.extract(newState)
  extracted.runAggregated(publish in extracted.get(thisProjectRef), newState)
  state
}

lazy val publishLocalCrossCompiled = Command.command("publishLocalCrossCompiled") { state =>
  val newState = reapply(Seq(
    nativeCompile in jni := Seq.empty,
    nativeCrossCompile in CrossCompile in jni := {
      (nativeCrossCompile in CrossCompile in jni).dependsOn(nativeCompile in jni).value
    },
    resourceGenerators in Compile in jni += Def.task {
      jniLibraries((nativeCrossCompile in CrossCompile in jni).value, (resourceManaged in Compile in jni).value)
    }.taskValue
  ), state)
  val extracted = Project.extract(newState)
  extracted.runAggregated(publishLocal in extracted.get(thisProjectRef), newState)
  state
}

// TODO: Add command for the signed publish tasks.
