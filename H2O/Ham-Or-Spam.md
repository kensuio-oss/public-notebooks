### Intro:

This example is the Ham-Or-Spam demo [1] of H2O Sparkling-water that is integrated into the spark-notebook.

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2] 

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/scripts/hamOrSpam.script.scala


### Dependencies :

Imports the H2O (sparkling-water) depencies that we need.  Note that these dependencies comes with many of the spark dependencies as well, so in order to avoid conflict, we need to remove some of these friend dependencies.

```scala
:dp
    ai.h2o % sparkling-water-core_2.11 % 2.0.2
    - org.apache.hadoop % hadoop-client %   _
    - org.apache.spark  % spark-core_2.11    %   _
    - org.apache.spark % spark-mllib_2.11 % _
    - org.apache.spark % spark-repl_2.11 % _
    - org.scala-lang    %     _         %   _
    - org.scoverage     %     _         %   _
    - org.eclipse.jetty.aggregate % jetty-servlet % _
```


><pre>
> globalScope.jars: Array[String] = [Ljava.lang.String;@18c08b01
> res2: List[String] = List(/tmp/spark-notebook/org/apache/avro/avro/1.8.0/avro-1.8.0.jar, /tmp/spark-notebook/org/apache/spark/spark-catalyst_2.11/2.0.1/spark-catalyst_2.11-2.0.1.jar, /tmp/spark-notebook/com/google/code/gson/gson/2.3.1/gson-2.3.1.jar, /tmp/spark-notebook/org/eclipse/jetty/orbit/javax.transaction/1.1.1.v201105210645/javax.transaction-1.1.1.v201105210645.jar, /tmp/spark-notebook/org/apache/spark/spark-sql_2.11/2.0.1/spark-sql_2.11-2.0.1.jar, /tmp/spark-notebook/org/eclipse/jetty/orbit/javax.activation/1.1.0.v201105071233/javax.activation-1.1.0.v201105071233.jar, /tmp/spark-notebook/org/apache/parquet/parquet-hadoop/1.7.0/parquet-hadoop-1.7.0.jar, /tmp/spark-notebook/com/google/guava/guava/16.0.1/guava-16.0.1.ja...
> </pre>



### Imports:
Imports the H2O packages that we will need

```scala
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.ModelMetricsBinomial
import org.apache.spark.h2o._
import org.apache.spark.{SparkFiles, mllib}
import org.apache.spark.mllib.feature.{IDFModel, IDF, HashingTF}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}
import water.Key
import water.support.{H2OFrameSupport, SparkContextSupport, ModelMetricsSupport}
```


><pre>
> import _root_.hex.deeplearning.DeepLearningModel
> import _root_.hex.ModelMetricsBinomial
> import org.apache.spark.h2o._
> import org.apache.spark.{SparkFiles, mllib}
> import org.apache.spark.mllib.feature.{IDFModel, IDF, HashingTF}
> import org.apache.spark.rdd.RDD
> import org.apache.spark.sql.{SQLContext, DataFrame}
> import water.Key
> import water.support.{H2OFrameSupport, SparkContextSupport, ModelMetricsSupport}
> </pre>



### Define helper classes and functions:

- SMS class: to hold features and target for a message
- load function: to load the file from the internet and split each lines based on tabulation  (the target is first element, rest of the message is the second element)
- tokenize function: to convert a message into a meaningful list of words
- buildIDFModel function: to build a Term frequency-inverse document frequency (TF-IDF) model --> More details on http://spark.apache.org/docs/latest/mllib-feature-extraction.html
- buildDLModel function: to build a DeepLearning (MLP) model --> Mode details on http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

```scala
case class SMS(target: String, fv: org.apache.spark.mllib.linalg.Vector)
```


><pre>
> defined class SMS
> </pre>




```scala
def load(dataFile: String): RDD[Array[String]] = {
  val data = scala.io.Source.fromURL(dataFile)("ISO-8859-1").getLines.toList 
  sc.parallelize(data).cache.map(l => l.split("\t")).filter(r => !r(0).isEmpty)
}
```


><pre>
> load: (dataFile: String)org.apache.spark.rdd.RDD[Array[String]]
> </pre>




```scala
def tokenize(data: RDD[String]): RDD[Seq[String]] = {
  val ignoredWords = Seq("the", "a", "", "in", "on", "at", "as", "not", "for")
  val ignoredChars = Seq(',', ':', ';', '/', '<', '>', '"', '.', '(', ')', '?', '-', '\'','!','0', '1')

  val texts = data.map( r=> {
    var smsText = r.toLowerCase
    for( c <- ignoredChars) {
      smsText = smsText.replace(c, ' ')
    }

    val words =smsText.split(" ").filter(w => !ignoredWords.contains(w) && w.length>2).distinct
    words.toSeq
  })
  texts
}
```


><pre>
> tokenize: (data: org.apache.spark.rdd.RDD[String])org.apache.spark.rdd.RDD[Seq[String]]
> </pre>




```scala
def buildIDFModel(tokens: RDD[Seq[String]], minDocFreq:Int = 4, hashSpaceSize:Int = 1 << 10): (HashingTF, IDFModel, RDD[mllib.linalg.Vector]) = {
  // Hash strings into the given space
  val hashingTF = new HashingTF(hashSpaceSize)
  val tf = hashingTF.transform(tokens)
  // Build term frequency-inverse document frequency
  val idfModel = new IDF(minDocFreq = minDocFreq).fit(tf)
  val expandedText = idfModel.transform(tf)
  (hashingTF, idfModel, expandedText)
}
```


><pre>
> buildIDFModel: (tokens: org.apache.spark.rdd.RDD[Seq[String]], minDocFreq: Int, hashSpaceSize: Int)(org.apache.spark.mllib.feature.HashingTF, org.apache.spark.mllib.feature.IDFModel, org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector])
> </pre>




```scala
def buildDLModel(train: Frame, valid: Frame, epochs: Int = 10, l1: Double = 0.001, l2: Double = 0.0, hidden: Array[Int] = Array[Int](200, 200)) (implicit h2oContext: H2OContext): DeepLearningModel = {
  import h2oContext.implicits._
  // Build a model
  import _root_.hex.deeplearning.DeepLearning
  import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
  val dlParams = new DeepLearningParameters()
  dlParams._train = train
  dlParams._valid = valid
  dlParams._response_column = "target"
  dlParams._epochs = epochs
  dlParams._l1 = l1
  dlParams._hidden = hidden

  // Create a job
  val dl = new DeepLearning(dlParams, Key.make("dlModel.hex"))
  dl.trainModel.get
}
```


><pre>
> buildDLModel: (train: org.apache.spark.h2o.Frame, valid: org.apache.spark.h2o.Frame, epochs: Int, l1: Double, l2: Double, hidden: Array[Int])(implicit h2oContext: org.apache.spark.h2o.H2OContext)hex.deeplearning.DeepLearningModel
> </pre>



### Ready to start

We are done with definiting all helper functions, and we can now proceed with the code that will actually do the job.

We define the session, sqlContext and H2O Context

```scala
//handle to the spark Session
val spark = SparkSession.builder().getOrCreate()

// Create SQL support
implicit val sqlContext = spark.sqlContext
import sqlContext.implicits._

// Start H2O services
import org.apache.spark.h2o._
implicit val h2oContext = H2OContext.getOrCreate(sc)
import h2oContext.implicits._
```


><pre>
> spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@19e7939f
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@27feef99
> import sqlContext.implicits._
> import org.apache.spark.h2o._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_-564160527
>  * cluster size: 1
>  * list of used nodes:
>   (executorId, host, port)
>   ------------------------
>   (driver,localhost,54323)
>   ------------------------
> 
>   Open H2O Flow in browser: http://127.0.0.1:54323 (CMD + click in Mac OSX)
> 
> import h2oContext.implicits._
> </pre>



### Prepare the data

```scala
//val data = load("smsData.txt")
val data = load("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/smsData.txt")
val hamSpam = data.map( r => r(0)) //spam or ham
val message = data.map( r => r(1)) //message as string
val tokens = tokenize(message) //message as word tokens
```


><pre>
> data: org.apache.spark.rdd.RDD[Array[String]] = MapPartitionsRDD[6] at filter at <console>:80
> hamSpam: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[7] at map at <console>:97
> message: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[8] at map at <console>:98
> tokens: org.apache.spark.rdd.RDD[Seq[String]] = MapPartitionsRDD[9] at map at <console>:82
> </pre>



### Build the TF-IDF model

This requires to loop on the data to fit the model (can take sometime for larger datasets).

```scala
var (hashingTF, idfModel, tfidf) = buildIDFModel(tokens,hashSpaceSize=1 << 10)
```


><pre>
> hashingTF: org.apache.spark.mllib.feature.HashingTF = org.apache.spark.mllib.feature.HashingTF@77b2b1eb
> idfModel: org.apache.spark.mllib.feature.IDFModel = org.apache.spark.mllib.feature.IDFModel@1f68a374
> tfidf: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = MapPartitionsRDD[15] at mapPartitions at IDF.scala:178
> </pre>



### Build the H2O Frame
- Merge back target with TF-IDF extracted vectors
- Convert the RDD to a spark dataframe
- Convert the spark dataframe to a H2OFrame
- Turn the target string to a categorical vector

```scala

val resultRDD: DataFrame = hamSpam.zip(tfidf).map(v => SMS(v._1, v._2)).toDF
val table:H2OFrame = resultRDD
table.replace(table.find("target"), table.vec("target").toCategoricalVec).remove()  // Transform target column into a categorical vector
```


><pre>
> resultRDD: org.apache.spark.sql.DataFrame = [target: string, fv: vector]
> table: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_21
>    cols: 1025
>    rows: 1324
>  chunks: 8
>    size: 760448
> </pre>



### Create train/validation datasets
The table is split into a training (80%) and a validation (20%) dataset.  

We are all done with the H2OFrame table, so we can release it.

```scala

val keys = Array[String]("train.hex", "valid.hex")
val ratios = Array[Double](0.8)
val frs = H2OFrameSupport.split(table, keys, ratios)
val (train, valid) = (frs(0), frs(1))
table.delete()
```


><pre>
> keys: Array[String] = Array(train.hex, valid.hex)
> ratios: Array[Double] = Array(0.8)
> frs: Array[water.fvec.Frame] =
> Array(Frame key: train.hex
>    cols: 1025
>    rows: 1059
>  chunks: 7
>    size: 658019
> , Frame key: valid.hex
>    cols: 1025
>    rows: 265
>  chunks: 2
>    size: 183192
> )
> train: water.fvec.Frame =
> Frame key: train.hex
>    cols: 1025
>    rows: 1059
>  chunks: 7
>    size: 658019
> 
> valid: water.fvec.Frame =
> Frame key: valid.hex
>    cols: 1025
>    rows: 265
>  chunks: 2
>    size: 183192
> </pre>



### Build the deeplearning model

The model is trained to predict the target based on the TF-IDF Feature vector.  The model has 2 hidden layers of 200 neurons each.
It is trained on 10 epochs and uses a L1 regularization.

Warning:  This step can take some time on a large dataset.

```scala
val epochs: Int = 10
val l1: Double = 0.001
val l2: Double = 0.0
val hidden: Array[Int] = Array[Int](200, 200) 

val dlModel = buildDLModel(train, valid, epochs, l1, l2, hidden)
```


><pre>
> epochs: Int = 10
> l1: Double = 0.001
> l2: Double = 0.0
> hidden: Array[Int] = Array(200, 200)
> dlModel: hex.deeplearning.DeepLearningModel =
> Model Metrics Type: Binomial
>  Description: Metrics reported on full training frame
>  model id: dlModel.hex
>  frame id: train.hex
>  MSE: 0.004366137
>  RMSE: 0.066076756
>  AUC: 0.99994326
>  logloss: 0.017004592
>  mean_per_class_error: 0.0018726592
>  default threshold: 0.8418402671813965
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>         ham  spam   Error       Rate
>    ham  792     0  0.0000    0 / 792
>   spam    1   266  0.0037    1 / 267
> Totals  793   266  0.0009  1 / 1,059
> Gains/Lift Table (Avg response rate: 25.21 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture...
> </pre>



### Collect model metrics and evaluate model quality

```scala
val trainMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, train)
val validMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, valid)
println("Training AUC: " + trainMetrics.auc)
println("Validation AUC: " + validMetrics.auc)
```


><pre>
> Training AUC: 0.9999432527522415
> Validation AUC: 0.9837229437229438
> trainMetrics: hex.ModelMetricsBinomial =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: dlModel.hex
>  frame id: train.hex
>  MSE: 0.004366137
>  RMSE: 0.066076756
>  AUC: 0.99994326
>  logloss: 0.017004592
>  mean_per_class_error: 0.0018726592
>  default threshold: 0.8418402671813965
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>         ham  spam   Error       Rate
>    ham  792     0  0.0000    0 / 792
>   spam    1   266  0.0037    1 / 267
> Totals  793   266  0.0009  1 / 1,059
> Gains/Lift Table (Avg response rate: 25.21 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture Rate  Cumulative Capture Rate         Gain  Cumulative Gain
>       1                0.01038716         1.000000  3.966292         3...
> </pre>



### Spark Detector

The model is trained and we can now use it to identify (unlabelled) spam messages.
We create an helper function isSpam to to the job.


```scala
// Spam detector
def isSpam(msg: String, dlModel: DeepLearningModel, hashingTF: HashingTF, idfModel: IDFModel, h2oContext: H2OContext, hamThreshold: Double = 0.5):Boolean = {
  val msgRdd = sc.parallelize(Seq(msg))
  val msgVector: DataFrame = idfModel.transform( hashingTF.transform (tokenize (msgRdd))).map(v => SMS("?", v)).toDF
  val msgTable: H2OFrame = h2oContext.asH2OFrame(msgVector)
  msgTable.remove(0) // remove first column
  val prediction = dlModel.score(msgTable)
  prediction.vecs()(1).at(0) < hamThreshold
}
```


><pre>
> isSpam: (msg: String, dlModel: hex.deeplearning.DeepLearningModel, hashingTF: org.apache.spark.mllib.feature.HashingTF, idfModel: org.apache.spark.mllib.feature.IDFModel, h2oContext: org.apache.spark.h2o.H2OContext, hamThreshold: Double)Boolean
> </pre>



### Testing the detector on new messages

```scala
//pass each messages through the spam detector
Array[String](
  "Michal, h2oworld party tonight in MV?",
  "We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?"
).foreach( sms => 
  println("Spam=%5b <--> Message='%s'".format(isSpam(sms, dlModel, hashingTF, idfModel, h2oContext), sms ) )
)
```


><pre>
> Spam=false <--> Message='Michal, h2oworld party tonight in MV?'
> Spam= true <--> Message='We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?'
> </pre>