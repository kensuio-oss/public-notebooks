### Intro:

This example is the Sentiment Analysis demo using the AmazonFineFood dataset [1] of H2O Sparkling-water that is integrated into the spark-notebook. In this notebook we will see:

- how to download and process streamed input data using sliding window
- how to perform some text parsing
- how to convert a text into a feature vector using tokenizer, hashing and a TF-IDF Model.
- how to build H2OFrame/DataFrame from simple object classes
- how to perform operation on the spark dataframe  (i.e. by using UDF functions)
- how to split a dataset into a training, validation and test sets
- how to define, train and use Deeplearning and Gradient Boosting models
- how to use the built model to perform sentimental analysis on an english text

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2]

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/src/main/scala/org/apache/spark/examples/h2o/AmazonFineFood.scala


### Metadata

In order to work properly, this notebook requires some edition of the metadata, in particular you should add the custom dependencies (as bellow) in order to load H2O and avoid interferences with spark versions. 
You also need to pass custom spark config parameters inorder to disable H2O REPL and to specify the port of the H2O Flow UI.

```
"customLocalRepo": "/tmp/spark-notebook",
"customDeps": [
  "ai.h2o % sparkling-water-core_2.11 % 2.0.2",  
  "- org.apache.hadoop % hadoop-client %   _",
  "- org.apache.spark  % spark-core_2.11    %   _",
  "- org.apache.spark % spark-mllib_2.11 % _",
  "- org.apache.spark % spark-repl_2.11 % _",
  "- org.scala-lang    %     _         %   _",
  "- org.scoverage     %     _         %   _",
  "- org.eclipse.jetty.aggregate % jetty-servlet % _"
],
"customSparkConf": {
  "spark.ext.h2o.repl.enabled": "false",
  "spark.ext.h2o.port.base": 54321
},
```

Don't forget to restart the kernel after you changed the metadata
### Imports:
Imports the H2O packages that we will need

```scala
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkFiles
import org.apache.spark.h2o._
import org.apache.spark.sql.{DataFrame, SQLContext, Row}
import org.apache.spark.mllib.feature.{IDFModel, HashingTF, IDF}

import water.Key
import water.fvec.Vec
import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport}
import water.support.H2OFrameSupport._

import java.io.File
import scala.util.Try
```


><pre>
> import org.apache.commons.io.FileUtils
> import org.apache.spark.SparkFiles
> import org.apache.spark.h2o._
> import org.apache.spark.sql.{DataFrame, SQLContext, Row}
> import org.apache.spark.mllib.feature.{IDFModel, HashingTF, IDF}
> import water.Key
> import water.fvec.Vec
> import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport}
> import water.support.H2OFrameSupport._
> import java.io.File
> import scala.util.Try
> </pre>



### Contexts

We prepare the SQL and H2O Contexts that we will need later

```scala
implicit val sqlContext = sparkSession.sqlContext
import sqlContext.implicits._

val h2oContext = H2OContext.getOrCreate(sparkSession.sparkContext)
import h2oContext._

import h2oContext.implicits._
```


><pre>
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@694d08fb
> import sqlContext.implicits._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_-733657802
>  * cluster size: 1
>  * list of used nodes:
>   (executorId, host, port)
>   ------------------------
>   (driver,192.168.0.13,54321)
>   ------------------------
> 
>   Open H2O Flow in browser: http://192.168.0.13:54321 (CMD + click in Mac OSX)
> 
> import h2oContext._
> import h2oContext.implicits._
> </pre>



### Amazon Fine Foods reviews data

The data are taken from SNAP: https://snap.stanford.edu/data/web-FineFoods.html .
Be carefull that the file size is about 100MB GZipped, so download and processing may take sometime

The dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review.

The data are structured in block of 9 lines within the text file:
```
product/productId: B001E4KFG0 
review/userId: A3SGXH7AUHU8GW 
review/profileName: delmartian 
review/helpfulness: 1/1 
review/score: 5.0 
review/time: 1303862400 
review/summary: Good Quality Dog Food 
review/text: I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than most.
<EMPTY-LINE>
```

There is a bit of gymnastic needed in order to read the streamed file in blocks (a sliding window is used) and keep only the fields with the data.
This is what is done here bellow.

- We create a case class to contain the review data (and some helper parsing functions)
- We get an input stream of the amazon data hosted on SNAP
- We read the file by blocks of 9 lines (corresponding to a review)
- We create an RDD and a DataFrame from this sample (and we cache it)
- We show the first 10 rows of the table


```scala
case class amazonReview(productId:Option[String], userId:Option[String], profileName:Option[String], helpfulness:Option[String], score:Option[Float], time:Option[Long], summary:Option[String], text:Option[String]){
      def isWrongRow():Boolean = (0 until productArity).map( idx => productElement(idx)).forall(e => e==None)
}

import scala.util.Try
def str(s: String): Option[String] = Try(s.toString).toOption
def long(s: String): Option[Long] = Try(s.toLong).toOption
def float(s: String): Option[Float] = Try(s.toFloat).toOption
```


><pre>
> defined class amazonReview
> import scala.util.Try
> str: (s: String)Option[String]
> long: (s: String)Option[Long]
> float: (s: String)Option[Float]
> </pre>




```scala
val amazonFileURL = new java.net.URL("https://snap.stanford.edu/data/finefoods.txt.gz")
val amazonBlocks = scala.io.Source.fromInputStream(new java.util.zip.GZIPInputStream(amazonFileURL.openStream))("ISO-8859-1").getLines.sliding(9,9)
val amazonRec = amazonBlocks
  .take(25000)  //only consider 25000 reviews to keep the demo fast
  .map(reccord => { reccord.map(_.split(": ",2)).filter(_.length>=2).map(_(1)) })
  .filter(_.length==8)
  .map(fields => amazonReview(str(fields(0)), str(fields(1)), str(fields(2)), str(fields(3)), float(fields(4)), long(fields(5)), str(fields(6)), str(fields(7))) )
val amazonRDD = sc.parallelize(amazonRec.toList).cache()
println("Number of review considered = %d".format(amazonRDD.count())) //trigger the dataset caching
val amazonDF = amazonRDD.toDF
amazonDF.show(10)
```


><pre>
> Number of review considered = 25000
> +----------+--------------+--------------------+-----------+-----+----------+--------------------+--------------------+
> | productId|        userId|         profileName|helpfulness|score|      time|             summary|                text|
> +----------+--------------+--------------------+-----------+-----+----------+--------------------+--------------------+
> |B001E4KFG0|A3SGXH7AUHU8GW|          delmartian|        1/1|  5.0|1303862400|Good Quality Dog ...|I have bought sev...|
> |B00813GRG4|A1D87F6ZCVE5NK|              dll pa|        0/0|  1.0|1346976000|   Not as Advertised|Product arrived l...|
> |B000LQOCH0| ABXLMWJIXXAIN|Natalia Corres "N...|        1/1|  4.0|1219017600|"Delight" says it...|This is a confect...|
> |B000UA0QIQ|A395BORC6FGVXV|                Karl|        3/3|  2.0|1307923200|      Cough Medicine|If you are lookin...|
> |B006K2ZZ7K|A1UQRSCLF8GW1T|Michael D. Bigham...|        0/0|  5.0|1350777600|         Great taffy|Great taffy at a ...|
> |B006K2ZZ7K| ADT0SRK1MGOEU|      Twoapennything|        0/0|  4.0|1342051200|          Nice Taffy|I got a wild hair...|
> |B006K2ZZ7K|A1SP2KVKFXXRU1|   David C. Sullivan|        0/0|  5.0|1340150400|Great!  Just as g...|This saltwater ta...|
> |B006K2ZZ7K|A3JRGQVEQN31IQ|  Pamela G. Williams|        0/0|  5.0|1336003200|Wonderful, tasty ...|This taffy is so ...|
> |B000E7L2R4|A1MZYO9TZK0BBI|            R. James|        1/1|  5.0|1322006400|          Yay Barley|Right now I'm mos...|
> |B00171APVA|A21BT40VZCCYT4|       Carol A. Reed|        0/0|  5.0|1351209600|    Healthy Dog Food|This is a very he...|
> +----------+--------------+--------------------+-----------+-----+----------+--------------------+--------------------+
> only showing top 10 rows
> 
> amazonFileURL: java.net.URL = https://snap.stanford.edu/data/finefoods.txt.gz
> amazonBlocks: Iterator[String]#GroupedIterator[String] = non-empty iterator
> amazonRec: Iterator[amazonReview] = empty iterator
> amazonRDD: org.apache.spark.rdd.RDD[amazonReview] = ParallelCollectionRDD[4] at parallelize at <console>:108
> amazonDF: org.apache.spark.sql.DataFrame = [productId: string, userId: string ... 6 more fields]
> </pre>



### Rare words

We identify the words that are rarely used (<2 time) in the dataset.  Both the review summary and review text are considered. 

```scala
val rareWordsSet = amazonRDD
  .map(review => review.summary + " " + review.text)
  .flatMap(text => text.split("""\W+""").map(_.toLowerCase))
  .filter(word => """[^0-9]*""".r.pattern.matcher(word).matches)
  .map(w => (w, 1))
  .reduceByKey(_+_)
  .filter({ case (k, v) => v < 2 })
  .map({ case (k, v) => k })
  .collect.toSet
rareWordsSet.toList
```


><pre>
> rareWordsSet: scala.collection.immutable.Set[String] = Set(professed, renfros, purifies, quotient, lauryl, brink, genererics, kethcup, boutiques, comply, seeder, insinuates, mario, poing, fryed, exprensive, moistruizing, elequent, mmmmmmmmmmm, cadged, woodinville, frutini, fobs, havig, snackin, harrowing, assert, yummu, verison, spokes, internist, annd, floaters, brul, termination, kentuckygirl, papillion, brewe, pareve, finick, lsfjewldf, remindes, kinks, redued, goodbars, easiness, crossbred, fne, argentine, fantastics, ribonucleotides, hisbiscus, fanastic, foll, vegis, insufficiency, piment, rouse, roated, associations, disappionted, underwear, glean, nestum, priouct, dogster, theor, conceivably, centimeters, officemate, badass, chao, practitioner, eraserhead, gulcosamine, doxee, pra...
> </pre>



### Transforming functions

We define functions that we will need to transform either RDD variables or Dataframe variables (using UDF functions).
These functions are embedded into a serializable object, to make it parallelizable.

```scala

object tokenizer extends Serializable {
  val rarewords = rareWordsSet
  
  val stopwords = Set("ax","i","you","edu","s","t","m","subject","can","lines","re","what"
    ,"there","all","we","one","the","a","an","of","or","in","for","by","on"
    ,"but", "is", "in","a","not","with", "as", "was", "if","they", "are", "this", "and", "it", "have"
  , "from", "at", "my","be","by","not", "that", "to","from","com","org","like","likes","so")
  stopwords.toList
  
  val toTokens = { text:String => text
      .split("""\W+""")
      .map(_.toLowerCase)
      .filter(word => """[^0-9]*""".r.pattern.matcher(word).matches)
      .filterNot(word => stopwords.contains(word))
      .filter(word => word.size >= 2)
      .filterNot(word => rarewords.contains(word))
  }  
  val toTokensUDF = udf { text:String => toTokens(text)  }  
  
  val hashingTF = new HashingTF(1024)  
  val toNumeric = { terms: Seq[_] => hashingTF.transform(terms) }  
  val toNumericUDF = udf { terms: Seq[_] => toNumeric(terms) }  
  val toNumericFeaturesUDF = udf { text:String => toNumeric(toTokens(text))}
  val toNumericFeaturesConcatUDF = udf { (summary:String, text:String) => toNumeric(toTokens(summary+" "+text))}
  
}

val toBinaryScore = udf { score:Float => if (score < 3.toByte) "negative" else "positive" }
```


><pre>
> defined object tokenizer
> toBinaryScore: org.apache.spark.sql.expressions.UserDefinedFunction = UserDefinedFunction(<function1>,StringType,Some(List(FloatType)))
> </pre>



### Dataframe Transformation

- We filter the dataset to remove neutral reviews (score==3)
- We turn the score column to a string variable:  positive or negative
- We add new columns to hold numerical vectors of the summary, text and summary+text variables.  The length of the vector is defined by the hashing space.

```scala
val proccessedDF = amazonDF
   .where("score !=3")
   .withColumn("score", toBinaryScore(col("score")))  
   .withColumn("summaryFeatures", tokenizer.toNumericFeaturesUDF(col("summary")))
   .withColumn("textFeatures"   , tokenizer.toNumericFeaturesUDF(col("text")))
   .withColumn("bothFeatures"   , tokenizer.toNumericFeaturesConcatUDF(col("summary"),col("text")))

proccessedDF.show(25)
```


><pre>
> +----------+--------------+--------------------+-----------+--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
> | productId|        userId|         profileName|helpfulness|   score|      time|             summary|                text|     summaryFeatures|        textFeatures|        bothFeatures|
> +----------+--------------+--------------------+-----------+--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
> |B001E4KFG0|A3SGXH7AUHU8GW|          delmartian|        1/1|positive|1303862400|Good Quality Dog ...|I have bought sev...|(1024,[143,301,79...|(1024,[101,112,14...|(1024,[101,112,14...|
> |B00813GRG4|A1D87F6ZCVE5NK|              dll pa|        0/0|negative|1346976000|   Not as Advertised|Product arrived l...|  (1024,[775],[1.0])|(1024,[10,112,145...|(1024,[10,112,145...|
> |B000LQOCH0| ABXLMWJIXXAIN|Natalia Corres "N...|        1/1|positive|1219017600|"Delight" says it...|This is a confect...|(1024,[34,687],[1...|(1024,[41,77,78,1...|(1024,[34,41,77,7...|
> |B000UA0QIQ|A395BORC6FGVXV|                Karl|        3/3|negative|1307923200|      Cough Medicine|If you are lookin...|(1024,[21,1007],[...|(1024,[33,120,123...|(1024,[21,33,120,...|
> |B006K2ZZ7K|A1UQRSCLF8GW1T|Michael D. Bigham...|        0/0|positive|1350777600|         Great taffy|Great taffy at a ...|(1024,[116,538],[...|(1024,[78,112,116...|(1024,[78,112,116...|
> |B006K2ZZ7K| ADT0SRK1MGOEU|      Twoapennything|        0/0|positive|1342051200|          Nice Taffy|I got a wild hair...|(1024,[538,842],[...|(1024,[33,37,41,1...|(1024,[33,37,41,1...|
> |B006K2ZZ7K|A1SP2KVKFXXRU1|   David C. Sullivan|        0/0|positive|1340150400|Great!  Just as g...|This saltwater ta...|(1024,[116,480,77...|(1024,[91,116,120...|(1024,[91,116,120...|
> |B006K2ZZ7K|A3JRGQVEQN31IQ|  Pamela G. Williams|        0/0|positive|1336003200|Wonderful, tasty ...|This taffy is so ...|(1024,[78,538,569...|(1024,[23,42,120,...|(1024,[23,42,78,1...|
> |B000E7L2R4|A1MZYO9TZK0BBI|            R. James|        1/1|positive|1322006400|          Yay Barley|Right now I'm mos...|(1024,[181,497],[...|(1024,[112,144,26...|(1024,[112,144,18...|
> |B00171APVA|A21BT40VZCCYT4|       Carol A. Reed|        0/0|positive|1351209600|    Healthy Dog Food|This is a very he...|(1024,[143,301,47...|(1024,[120,143,25...|(1024,[120,143,25...|
> |B0001PB9FE|A3HDKO7OW0QNK4|        Canadian Fan|        1/1|positive|1107820800|The Best Hot Sauc...|I don't know if i...|(1024,[139,359,47...|(1024,[10,14,18,3...|(1024,[10,14,18,3...|
> |B0009XLVG0|A2725IB4YY9JEB|A Poeng "SparkyGo...|        4/4|positive|1282867200|My cats LOVE this...|One of my boys ne...|(1024,[112,205,21...|(1024,[1,4,14,18,...|(1024,[1,4,14,18,...|
> |B0009XLVG0| A327PCT23YH90|                  LT|        1/1|negative|1339545600|My Cats Are Not F...|My cats have been...|(1024,[249,270,30...|(1024,[24,68,103,...|(1024,[24,68,103,...|
> |B001GVISJM|A18ECVX2RJ7HUE|     willie "roadie"|        2/2|positive|1288915200|   fresh and greasy!|good flavor! thes...|(1024,[358,747],[...|(1024,[112,126,24...|(1024,[112,126,24...|
> |B001GVISJM|A2MUGFV2TDQ47K| Lynrie "Oh HELL no"|        4/5|positive|1268352000|Strawberry Twizzl...|The Strawberry Tw...|(1024,[78,126,568...|(1024,[78,126,268...|(1024,[78,126,268...|
> |B001GVISJM|A1CZX3CP8IKQIJ|        Brian A. Lee|        4/5|positive|1262044800|Lots of twizzlers...|My daughter loves...|(1024,[126,252,33...|(1024,[14,51,126,...|(1024,[14,51,126,...|
> |B001GVISJM|A3KLWF6WQ5BNYO|      Erica Neathery|        0/0|negative|1348099200|          poor taste|I love eating the...|(1024,[326,743],[...|(1024,[112,123,26...|(1024,[112,123,26...|
> |B001GVISJM| AFKW14U97Z6QO|               Becca|        0/0|positive|1345075200|            Love it!|I am very satisfi...|  (1024,[112],[1.0])|(1024,[72,120,268...|(1024,[72,112,120...|
> |B001GVISJM|A2A9X58G2GTBLP|             Wolfee1|        0/0|positive|1324598400|  GREAT SWEET CANDY!|Twizzlers, Strawb...|(1024,[116,833,99...|(1024,[1,26,45,91...|(1024,[1,26,45,91...|
> |B001GVISJM|A3IV7CL2C13K2U|                Greg|        0/0|positive|1318032000|Home delivered tw...|Candy was deliver...|(1024,[218,485],[...|(1024,[120,199,21...|(1024,[120,199,21...|
> |B001GVISJM|A1WO0KGLPR5PV6|            mom2emma|        0/0|positive|1313452800|        Always fresh|My husband is a T...|(1024,[277,358],[...|(1024,[39,45,78,8...|(1024,[39,45,78,8...|
> |B001GVISJM| AZOF9E17RGZH8|      Tammy Anderson|        0/0|positive|1308960000|           TWIZZLERS|I bought these fo...|  (1024,[126],[1.0])|(1024,[24,102,107...|(1024,[24,102,107...|
> |B001GVISJM| ARYVQL4N737A1|       Charles Brown|        0/0|positive|1304899200|  Delicious product!|I can remember bu...|(1024,[112,601],[...|(1024,[52,112,333...|(1024,[52,112,333...|
> |B001GVISJM| AJ613OLZZUG7V|              Mare's|        0/0|positive|1304467200|           Twizzlers|I love this candy...|  (1024,[126],[1.0])|(1024,[37,112,174...|(1024,[37,112,126...|
> |B001GVISJM|A22P2J09NJ9HKE|S. Cabanaugh "jil...|        0/0|positive|1295481600|Please sell these...|I have lived out ...|(1024,[490,521,56...|(1024,[66,86,88,1...|(1024,[66,86,88,1...|
> +----------+--------------+--------------------+-----------+--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
> only showing top 25 rows
> 
> proccessedDF: org.apache.spark.sql.DataFrame = [productId: string, userId: string ... 9 more fields]
> </pre>



### TF-IDF Model

- We build a TF-IDF model and we fit it on the numerical vector of Summary+Text
- We create a new dataframe with the TF-IDF transformed feature and the score label  (this is done using the new case class LabelFeature)
- We show the first 10 entries of this dataframe.

```scala
case class LabelFeature(label: String, feature: org.apache.spark.mllib.linalg.Vector)
```


><pre>
> defined class LabelFeature
> </pre>




```scala
val idfModel = new IDF(minDocFreq = 1).fit(
  proccessedDF.select("bothFeatures").rdd.map { case Row(v: org.apache.spark.mllib.linalg.Vector) => v}
)

val finalDF = proccessedDF.select("score", "summaryFeatures").rdd.map(row => LabelFeature(row.getAs[String]("score"), idfModel.transform(row.getAs[org.apache.spark.mllib.linalg.Vector]("summaryFeatures"))) ).toDF
finalDF.take(10)
```


><pre>
> idfModel: org.apache.spark.mllib.feature.IDFModel = org.apache.spark.mllib.feature.IDFModel@554bfd2
> finalDF: org.apache.spark.sql.DataFrame = [label: string, feature: vector]
> res12: Array[org.apache.spark.sql.Row] = Array([positive,(1024,[143,301,792,1009],[2.287815059066819,2.1007612929686497,1.0811292500408394,2.2626913190716236])], [negative,(1024,[775],[3.3583642816680186])], [positive,(1024,[34,687],[3.685820679607611,3.147115087967885])], [negative,(1024,[21,1007],[3.940881833890126,3.9386321141561105])], [positive,(1024,[116,538],[0.8917582431132202,3.498566572554681])], [positive,(1024,[538,842],[3.498566572554681,2.640371102521543])], [positive,(1024,[116,480,774,792,915],[0.8917582431132202,3.300926382080024,2.9466295605462594,1.0811292500408394,1.3602896993581357])], [positiv...
> </pre>



### H2O Frames

- We convert the dataframe to a H2OFrame
- We split it into a train/validation sub samples


```scala
val finalH2O:H2OFrame = h2oContext.asH2OFrame(finalDF, "dataset.hex")
finalH2O.replace(finalH2O.find("label"), finalH2O.vec("label").toCategoricalVec).remove()  // Transform label column into a categorical vector
val frs = H2OFrameSupport.split(finalH2O, Array[String]("train.hex", "valid.hex"), Array[Double](0.8))
val (train, valid) = (frs(0), frs(1))
finalH2O.delete()
```


><pre>
> finalH2O: org.apache.spark.h2o.H2OFrame =
> Frame key: dataset.hex
>    cols: 0
>    rows: 22849
>  chunks: 8
>    size: 0
> 
> frs: Array[water.fvec.Frame] =
> Array(Frame key: train.hex
>    cols: 1025
>    rows: 18279
>  chunks: 7
>    size: 1099861
> , Frame key: valid.hex
>    cols: 1025
>    rows: 4570
>  chunks: 2
>    size: 300326
> )
> train: water.fvec.Frame =
> Frame key: train.hex
>    cols: 1025
>    rows: 18279
>  chunks: 7
>    size: 1099861
> 
> valid: water.fvec.Frame =
> Frame key: valid.hex
>    cols: 1025
>    rows: 4570
>  chunks: 2
>    size: 300326
> </pre>



### Grandient Boosting Model configuration

- We create a GBM model to predict the review score based on the feature summary+text.
- We train the model
- We get the model evaluations for training and validation datasets

```scala
import _root_.hex.tree.gbm.GBM
import _root_.hex.tree.gbm.GBMModel.GBMParameters
import _root_.hex.genmodel.utils.DistributionFamily
import _root_.hex.{Model, ModelMetricsBinomial}
import water.support.ModelMetricsSupport

val gbmParams = new GBMParameters()
gbmParams._train = train
gbmParams._valid = valid
gbmParams._response_column = "label"
gbmParams._ntrees = 10
gbmParams._max_depth = 6
gbmParams._distribution = DistributionFamily.bernoulli

val gbm = new GBM(gbmParams)
val gbmModel = gbm.trainModel.get

val trainMetricsGBM = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel,train)
val validMetricsGBM = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel, valid)
```


><pre>
> import _root_.hex.tree.gbm.GBM
> import _root_.hex.tree.gbm.GBMModel.GBMParameters
> import _root_.hex.genmodel.utils.DistributionFamily
> import _root_.hex.{Model, ModelMetricsBinomial}
> import water.support.ModelMetricsSupport
> gbmParams: hex.tree.gbm.GBMModel.GBMParameters = hex.tree.gbm.GBMModel$GBMParameters@2a69cc21
> gbmParams._train: water.Key[water.fvec.Frame] = train.hex
> gbmParams._valid: water.Key[water.fvec.Frame] = valid.hex
> gbmParams._response_column: String = label
> gbmParams._ntrees: Int = 10
> gbmParams._max_depth: Int = 6
> gbmParams._distribution: hex.genmodel.utils.DistributionFamily = bernoulli
> gbm: hex.tree.gbm.GBM = hex.tree.gbm.GBM@1bbda4c4
> gbmModel: hex.tree.gbm.GBMModel =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: GBM_model_1483551724239_1
>  frame id: train.hex
> ...
> </pre>



### Deeplearning Model configuration

- We create a deeplearning model to predict the review score based on the feature summary+text.
- We train the model
- We get the model evaluations for training and validation datasets

```scala
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import _root_.hex.{Model, ModelMetricsBinomial}
import water.support.ModelMetricsSupport

val dlParams = new DeepLearningParameters()
dlParams._train = train
dlParams._valid = valid
dlParams._response_column = "label"
dlParams._epochs = 10
dlParams._l1 = 0.0001
dlParams._l2 = 0.0001
dlParams._activation = Activation.RectifierWithDropout
dlParams._hidden = Array(200,200)

// Create a job
val dl = new DeepLearning(dlParams)
val dlModel = dl.trainModel.get

val trainMetricsDL = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel,train)
val validMetricsDL = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, valid)
```


><pre>
> import _root_.hex.deeplearning.DeepLearning
> import _root_.hex.deeplearning.DeepLearningModel
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
> import _root_.hex.{Model, ModelMetricsBinomial}
> import water.support.ModelMetricsSupport
> dlParams: hex.deeplearning.DeepLearningModel.DeepLearningParameters = hex.deeplearning.DeepLearningModel$DeepLearningParameters@143514d0
> dlParams._train: water.Key[water.fvec.Frame] = train.hex
> dlParams._valid: water.Key[water.fvec.Frame] = valid.hex
> dlParams._response_column: String = label
> dlParams._epochs: Double = 10.0
> dlParams._l1: Double = 1.0E-4
> dlParams._l2: Double = 1.0E-4
> dlParams._activation: hex.deeplearning.DeepLearningModel.DeepLearningParame...
> </pre>



### Model evaluation

We print the performances of the two models

```scala
println(
  s"""Model performance:
  GBM:
    train AUC = ${trainMetricsGBM.auc}
    test  AUC = ${validMetricsGBM.auc}
  DL:
    train AUC = ${trainMetricsDL.auc}
    test  AUC = ${validMetricsDL.auc}
  """.stripMargin)
```


><pre>
> Model performance:
>   GBM:
>     train AUC = 0.7469506804001735
>     test  AUC = 0.7273827511643896
>   DL:
>     train AUC = 0.8517564884944392
>     test  AUC = 0.8244533677873518
>   
> </pre>



# Model testing

We can now make use of the model the sentiment behind whatever english text.<br>
Of course, given that the training set is related to amazon product, the sentiment prediction might be quite biased.<br>
I don't suggest to use it on philosophical disertations or poems.

### Scoring helper function

We define a function that takes as input a text and a model and returns the identified text sentiment and its probability

```scala
def textSentiment(text: String, model: Model[_,_,_]) (implicit sqlContext: SQLContext, h2oContext: H2OContext) = {
    import h2oContext.implicits._
    import sqlContext.implicits._
  
    val features = idfModel.transform(tokenizer.toNumeric(tokenizer.toTokens(text)))
      
    // Create a single row table
    val srdd: DataFrame = sqlContext.sparkContext.parallelize(Seq(LabelFeature("negative", features) )).toDF
    // Join table with census data
    val row: H2OFrame = srdd


    val predictTable = model.score(row)
    
    val positive  = predictTable.vec("positive").at(0).toFloat
    val negative  = predictTable.vec("negative").at(0).toFloat
    if(positive>negative){
      "text is POSITIVE with %6.2f %% probability".format(100.0*positive)
    }else{
      "text is NEGATIVE with %6.2f %% probability".format(100.0*negative)
    }  
}
```


><pre>
> textSentiment: (text: String, model: hex.Model[_, _, _])(implicit sqlContext: org.apache.spark.sql.SQLContext, implicit h2oContext: org.apache.spark.h2o.H2OContext)String
> </pre>



### List of text to be processed

We manually enter a list of text review to be abalyzed.


```scala
val reviewExamples = Seq[String](
  "I like this product so much !!!",
  "Bad, nasty and expensive."
)
```


><pre>
> reviewExamples: Seq[String] = List(I like this product so much !!!, Bad, nasty and expensive.)
> </pre>



### Sentiments

We loop on the texts and use the textSentiment function to get the sentiment and probability of each text

```scala
for (review <- reviewExamples) {
  val resultGBM = textSentiment(review, gbmModel)(sqlContext, h2oContext)
  val resultDL = textSentiment(review, dlModel)(sqlContext, h2oContext)
  println(
    s"""
Text: "$review"
  DeepLearning: ${resultDL} 
  GBM         : ${resultGBM} 
    """.stripMargin)
}
```


><pre>
> 
> Text: "I like this product so much !!!"
>   DeepLearning: text is POSITIVE with  84.22 % probability 
>   GBM         : text is POSITIVE with  85.67 % probability 
>     
> 
> Text: "Bad, nasty and expensive."
>   DeepLearning: text is NEGATIVE with  74.98 % probability 
>   GBM         : text is POSITIVE with  66.11 % probability 
>     
> </pre>



### All done