/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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
import JniCrossPackage._
import sbtrelease.Vcs

import scala.sys.process.Process

scalaVersion in ThisBuild := "2.12.7"
crossScalaVersions in ThisBuild := Seq("2.11.12", "2.12.7")

organization in ThisBuild := "org.platanios"

val tensorFlowVersion = "1.11.0"
val circeVersion = "0.10.0" // Use for working with JSON.

autoCompilerPlugins in ThisBuild := true

// addCompilerPlugin(MetalsPlugin.semanticdbScalac)

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",        // Existential types (besides wildcard types) can be written and inferred.
  "-language:higherKinds",         // Allow higher-kinded types.
  "-language:implicitConversions", // Allow definition of implicit functions called views.
  "-unchecked",                    // Enable additional warnings where generated code depends on assumptions.
  // "-Xfatal-warnings",
  // "-Xlog-implicits",
  "-Yno-adapted-args",
  // "-Ypartial-unification",
  // "-Ywarn-dead-code",
  // "-Ywarn-numeric-widen",
  // "-Ywarn-value-discard",
  "-Yrangepos",
  "-Xfuture", // Turn on future language features.
  // "-P:splain:all",
  // "-P:splain:infix",
  // "-P:splain:foundreq",
  // "-P:splain:implicits",
  // "-P:splain:color",
  // "-P:splain:tree",
  // "-P:splain:boundsimplicits:false"
)

nativeCrossCompilationEnabled in ThisBuild := false

lazy val loggingSettings = Seq(
  libraryDependencies ++= Seq(
    "com.typesafe.scala-logging" %% "scala-logging"   % "3.9.0",
    "ch.qos.logback"             %  "logback-classic" % "1.2.3"))

lazy val commonSettings = loggingSettings ++ Seq(
  // Plugin that prints better implicit resolution errors.
  // addCompilerPlugin("io.tryp"  % "splain" % "0.3.3" cross CrossVersion.patch)
)

lazy val testSettings = Seq(
  libraryDependencies ++= Seq(
    "junit"         %  "junit"     % "4.12",
    "org.scalactic" %% "scalactic" % "3.0.5",
    "org.scalatest" %% "scalatest" % "3.0.5" % "test"),
  logBuffered in Test := false,
  fork in test := false,
  testForkedParallel in Test := false,
  parallelExecution in Test := false,
  testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oDF"))

lazy val all = (project in file("."))
    .aggregate(jni, api, data, examples, site)
    .dependsOn(jni, api)
    .settings(moduleName := "tensorflow", name := "TensorFlow Scala")
    .settings(commonSettings)
    .settings(publishSettings)
    .settings(
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      publishArtifact := true,
      packagedArtifacts ++= {
        (nativeCrossCompile in JniCross in jni).value
            .map(p => p._1 -> getPackagedArtifacts(p._1, p._2))
            .filter(_._2.isDefined).map {
          case (platform, file) => Artifact((nativeArtifactName in JniCross in jni).value, platform.tag) -> file.get
        }
      })

lazy val jni = (project in file("./modules/jni"))
    .enablePlugins(JniNative, TensorFlowGenerateTensorOps, JniCrossPackage, TensorFlowNativePackage)
    .settings(moduleName := "tensorflow-jni", name := "TensorFlow Scala - JNI Bindings")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(publishSettings)
    .settings(
      // Tensor op code generation settings
      target in generateTensorOps := sourceDirectory.value / "main",
      ops in generateTensorOps := Map(
        "Basic" -> Seq(
          "ZerosLike", "OnesLike", "Fill", "Rank", "Size", "Shape", "ExpandDims", "Squeeze", "Pack", "ParallelConcat",
          "Unpack", "ConcatV2", "ConcatOffset", "Split", "SplitV", "Tile", "Pad", "PadV2", "MirrorPad", "Reshape",
          "Transpose", "ConjugateTranspose", "InvertPermutation", "ReverseV2", "ReverseSequence", "SpaceToBatchND",
          "BatchToSpaceND", "SpaceToDepth", "DepthToSpace", "Where", "Unique", "UniqueWithCounts", "ListDiff",
          "GatherV2", "GatherNd", "ScatterNd", "Slice", "StridedSlice", "CheckNumerics", "EditDistance", "OneHot",
          "BroadcastArgs", "StopGradient", "PreventGradient", "Identity", "IdentityN", "ScatterNdNonAliasingAdd",
          "QuantizeAndDequantizeV3", "QuantizeV2", "Dequantize", "QuantizedConcat", "QuantizedReshape",
          "QuantizedInstanceNorm", "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars",
          "FakeQuantWithMinMaxVarsPerChannel"),
        "Math" -> Seq(
          "Select", "Range", "LinSpace", "Cast", "Bitcast", "AddN", "Abs", "ComplexAbs", "Neg", "Reciprocal", "Square",
          "Sqrt", "Rsqrt", "Exp", "Expm1", "Log", "Log1p", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh",
          "Tanh", "Asinh", "Acosh", "Atanh", "Lgamma", "Digamma", "Erf", "Erfc", "Sigmoid", "Sign", "Round", "Rint",
          "Floor", "Ceil", "IsNan", "IsInf", "IsFinite", "Add", "Sub", "Mul", "Div", "FloorDiv", "TruncateDiv",
          "RealDiv", "SquaredDifference", "Mod", "FloorMod", "TruncateMod", "Pow", "Igammac", "Igamma", "Zeta",
          "Polygamma", "Atan2", "Maximum", "Minimum", "Betainc", "LogicalNot", "LogicalAnd", "LogicalOr", "Equal",
          "NotEqual", "ApproximateEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "Sum", "Mean", "Prod", "Min",
          "Max", "All", "Any", "ArgMax", "ArgMin", "Bincount", "Cumsum", "Cumprod", "SegmentSum", "SegmentMean",
          "SegmentProd", "SegmentMin", "SegmentMax", "UnsortedSegmentSum", "UnsortedSegmentMax",
          "SparseSegmentSum", "SparseSegmentMean", "SparseSegmentSqrtN",
          "SparseSegmentSumWithNumSegments", "SparseSegmentMeanWithNumSegments", "SparseSegmentSqrtNWithNumSegments",
          "Diag", "DiagPart", "MatrixDiag", "MatrixSetDiag",
          "MatrixDiagPart", "MatrixBandPart", "MatMul", "BatchMatMul", "SparseMatMul", "Cross", "Complex", "Real",
          "Imag", "Angle", "Conj", "Bucketize", "QuantizedAdd", "QuantizedMul", "QuantizedMatMul",
          "QuantizeDownAndShrinkRange", "Requantize", "RequantizationRange", "CompareAndBitpack"),
        "NN" -> Seq(
          "BiasAdd", "Relu", "Relu6", "Elu", "Selu", "Softplus", "Softsign", "Softmax", "LogSoftmax", "L2Loss",
          "SoftmaxCrossEntropyWithLogits", "SparseSoftmaxCrossEntropyWithLogits", "TopKV2", "InTopKV2", "AvgPool",
          "AvgPool3D", "MaxPool", "MaxPoolGrad", "MaxPoolGradGrad", "MaxPool3D", "MaxPoolWithArgmax",
          "FractionalAvgPool", "FractionalMaxPool", "Conv2D", "Conv2DBackpropInput", "Conv2DBackpropFilter",
          "FusedResizeAndPadConv2D", "FusedPadConv2D", "DepthwiseConv2dNative", "Conv3D", "Dilation2D", "LRN",
          "BatchNormWithGlobalNormalization", "FusedBatchNorm", "QuantizedBiasAdd", "QuantizedRelu", "QuantizedRelu6",
          "QuantizedReluX", "QuantizedAvgPool", "QuantizedMaxPool", "QuantizedConv2D",
          "QuantizedBatchNormWithGlobalNormalization"),
        "Random" -> Seq(
          "RandomShuffle", "RandomUniform", "RandomUniformInt", "RandomStandardNormal", "TruncatedNormal"),
        "Sparse" -> Seq("SparseToDense"),
        "Text" -> Seq(
          "StringJoin", "StringSplit", "EncodeBase64", "DecodeBase64", "StringToHashBucket", "StringToHashBucketFast",
          "StringToHashBucketStrong")
      ),
      scalaPackage in generateTensorOps := "tensors",
      // Native bindings compilation settings
      target in javah := sourceDirectory.value / "main" / "native" / "include",
      sourceDirectory in nativeCompile := sourceDirectory.value / "main" / "native",
      target in nativeCompile := target.value / "native" / nativePlatform.value,
      target in JniCross := target.value / "native",
      nativePlatforms in JniCross := Set(LINUX_x86_64, LINUX_GPU_x86_64, DARWIN_x86_64),
      tfBinaryVersion in JniCross := "nightly", // tensorFlowVersion,
      tfLibCompile in JniCross := false,
      tfLibRepository in JniCross := "https://github.com/tensorflow/tensorflow.git",
      tfLibRepositoryBranch in JniCross := "master",
      // Specify the order in which the different compilation tasks are executed
      nativeCompile := nativeCompile.dependsOn(generateTensorOps).value)

lazy val api = (project in file("./modules/api"))
    .dependsOn(jni)
    .enablePlugins(ProtobufPlugin)
    .settings(moduleName := "tensorflow-api", name := "TensorFlow Scala - API")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(publishSettings)
    .settings(
      libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % "0.14.1",
        "org.tensorflow" % "proto" % tensorFlowVersion,
        "com.chuusai" %% "shapeless" % "2.3.3",
        compilerPlugin("com.github.ghik" %% "silencer-plugin" % "0.6"),
        "com.github.ghik" %% "silencer-lib" % "0.6"),
      libraryDependencies ++= Seq(
        "io.circe" %% "circe-core",
        "io.circe" %% "circe-generic",
        "io.circe" %% "circe-parser"
      ).map(_ % circeVersion),
      // Protobuf settings
      version in ProtobufConfig := "3.5.1",
      sourceDirectory in ProtobufConfig := sourceDirectory.value / "main" / "proto",
      javaSource in ProtobufConfig := ((sourceDirectory in Compile).value / "generated" / "java"),
      sourceDirectories in Compile += sourceDirectory.value / "main" / "generated" / "java",
      unmanagedResourceDirectories in Compile += (sourceDirectory in ProtobufConfig).value)

lazy val horovod = (project in file("./modules/horovod"))
    .dependsOn(jni, api)
    .enablePlugins(JniNative, JniCrossPackage)
    .settings(moduleName := "tensorflow-horovod", name := "TensorFlow Scala - Horovod")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(publishSettings)
    .settings(
      // Native bindings compilation settings
      target in javah := sourceDirectory.value / "main" / "native" / "include",
      sourceDirectory in nativeCompile := sourceDirectory.value / "main" / "native",
      target in nativeCompile := target.value / "native" / nativePlatform.value,
      dockerImagePrefix in JniCross := "tensorflow-jni",
      nativeArtifactName in JniCross := "horovod",
      nativeLibPath in JniCross := {
        (nativeCrossCompile in JniCross in jni).value
        val tfVersion = (tfBinaryVersion in JniCross in jni).value
        val tfJniTarget = (target in JniCross in jni).value
        val log = streams.value.log
        val targetDir = (target in nativeCrossCompile in JniCross).value
        IO.createDirectory(targetDir)
        (nativePlatforms in nativeCrossCompile in JniCross).value.map(platform => {
          val platformTargetDir = targetDir / platform.name
          IO.createDirectory(platformTargetDir / "downloads")
          IO.createDirectory(platformTargetDir / "downloads" / "lib")

          // Download the native TensorFlow library
          log.info(s"Downloading the TensorFlow native library.")
          val exitCode = TensorFlowNativePackage.downloadTfLib(
            platform, (tfJniTarget / platform.name).getPath, tfVersion
          ).map(_ ! log)

          if (exitCode.getOrElse(0) != 0) {
            sys.error(
              s"An error occurred while preparing the native TensorFlow libraries for '$platform'. " +
                  s"Exit code: $exitCode.")
          }

          platform -> tfJniTarget / platform.name
        }).toMap
      })

lazy val data = (project in file("./modules/data"))
    .dependsOn(api)
    .settings(moduleName := "tensorflow-data", name := "TensorFlow Scala - Data")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(publishSettings)
    .settings(
      libraryDependencies += "org.apache.commons" % "commons-compress" % "1.15")

lazy val examples = (project in file("./modules/examples"))
    .dependsOn(api, data)
    .settings(moduleName := "tensorflow-examples", name := "TensorFlow Scala - Examples")
    .settings(commonSettings)
    .settings(publishSettings)

lazy val site = (project in file("./docs/site"))
    .dependsOn(api)
    .enablePlugins(ScalaUnidocPlugin, MicrositesPlugin)
    .settings(moduleName := "tensorflow-site", name := "TensorFlow Scala - Site")
    .settings(commonSettings)
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
      micrositeBaseUrl := "tensorflow_scala",
      micrositeDocumentationUrl := "api/",
      micrositeAuthor := "Emmanouil Antonios Platanios",
      micrositeHomepage := "http://eaplatanios.github.io/tensorflow_scala/",
      micrositeOrganizationHomepage := "http://eaplatanios.github.io",
      micrositeGithubOwner := "eaplatanios",
      micrositeGithubRepo := "tensorflow_scala",
      micrositePushSiteWith := GHPagesPlugin,
      micrositeGitterChannel := true,
      micrositeGitterChannelUrl := "eaplatanios/tensorflow_scala",
      micrositeHighlightTheme := "hybrid",
      micrositeImgDirectory := (resourceDirectory in Compile).value / "site" / "img",
      micrositeCssDirectory := (resourceDirectory in Compile).value / "site" / "css",
      micrositeJsDirectory := (resourceDirectory in Compile).value / "site" / "js",
      micrositePalette := Map(
        "brand-primary"     -> "rgb(239, 108, 0)",
        "brand-secondary"   -> "#455A64",
        "brand-tertiary"    -> "#39474E", // "#303C42",
        "gray-dark"         -> "#453E46",
        "gray"              -> "#837F84",
        "gray-light"        -> "#E3E2E3",
        "gray-lighter"      -> "#F4F3F4",
        "white-color"       -> "#FFFFFF"),
      micrositeFooterText := None,
      includeFilter in makeSite :=
          "*.html" | "*.css" | "*.png" | "*.jpg" | "*.gif" | "*.js" | "*.swf" | "*.yml" | "*.md" | "*.svg",
      includeFilter in Jekyll := (includeFilter in makeSite).value)

lazy val noPublishSettings = Seq(
  publish := Unit,
  publishLocal := Unit,
  publishArtifact := false,
  skip in publish := true,
  releaseProcess := Nil)

val deletedPublishedSnapshots = taskKey[Unit]("Delete published snapshots.")

lazy val publishSettings = Seq(
  publishArtifact := true,
  homepage := Some(url("https://github.com/eaplatanios/tensorflow_scala")),
  licenses := Seq("Apache License 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt")),
  scmInfo := Some(ScmInfo(url("https://github.com/eaplatanios/tensorflow_scala"),
                          "scm:git:git@github.com:eaplatanios/tensorflow_scala.git")),
  developers := List(
    Developer(
      id="eaplatanios",
      name="Emmanouil Antonios Platanios",
      email="e.a.platanios@gmail.com",
      url=url("http://platanios.org/"))),
  autoAPIMappings := true,
  apiURL := Some(url("http://eaplatanios.github.io/tensorflow_scala/api/")),
  releaseCrossBuild := true,
  releaseTagName := {
    val buildVersionValue = (version in ThisBuild).value
    val versionValue = version.value
    s"v${if (releaseUseGlobalVersion.value) buildVersionValue else versionValue}"
  },
  releaseVersionBump := sbtrelease.Version.Bump.Next,
  releaseVersionFile := baseDirectory.value / "version.sbt",
  releaseUseGlobalVersion := true,
  releasePublishArtifactsAction := PgpKeys.publishSigned.value,
  releaseVcs := Vcs.detect(baseDirectory.value),
  releaseVcsSign := true,
  releaseIgnoreUntrackedFiles := true,
  useGpg := true,  // Bouncy Castle has bugs with sub-keys, so we use gpg instead
  PgpKeys.pgpSigner := new CommandLineGpgSigner(
    command = "gpg",
    agent = true,
    secRing = file("~/.gnupg/secring.gpg").getPath,
    optKey = pgpSigningKey.value,
    optPassphrase = sys.env.get("PGP_PASSWORD").map(_.toCharArray)),
  publishMavenStyle := true,
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
    setReleaseVersion,
    commitReleaseVersion,
    tagRelease,
    releaseStepCommandAndRemaining("publishSigned"),
    setNextVersion,
    commitNextVersion,
    releaseStepCommand("sonatypeReleaseAll"),
    pushChanges),
  // The following 2 lines are needed to get around this: https://github.com/sbt/sbt/issues/4275
  publishConfiguration := publishConfiguration.value.withOverwrite(true),
  publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true),
  // For Travis CI - see http://www.cakesolutions.net/teamblogs/publishing-artefacts-to-oss-sonatype-nexus-using-sbt-and-travis-ci
  credentials ++= (for {
    username <- Option(System.getenv().get("SONATYPE_USERNAME"))
    password <- Option(System.getenv().get("SONATYPE_PASSWORD"))
  } yield Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", username, password)).toSeq,
  deletedPublishedSnapshots := {
    Process(
      "curl" :: "--request" :: "DELETE" :: "--write" :: "%{http_code} %{url_effective}\\n" ::
          "--user" :: s"${System.getenv().get("SONATYPE_USERNAME")}:${System.getenv().get("SONATYPE_PASSWORD")}" ::
          "--output" :: "/dev/null" :: "--silent" ::
          s"${Opts.resolver.sonatypeSnapshots.root}/${organization.value.replace(".", "/")}/" :: Nil) ! streams.value.log
  }
)

