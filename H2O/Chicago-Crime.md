### Intro:

This example is the Chicago Crime demo [1] of H2O Sparkling-water that is integrated into the spark-notebook. In this notebook we will see:

- how to perform some text parsing
- how to build H2OFrame/DataFrame from complex object classes
- how to join two tables using spark SQL
- how to split a dataset into a training, validation and test sets
- how to define, train and use Deeplearning and Gradient Boosting models
- how to merge back model predictions and features into a single table

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2]

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/scripts/chicagoCrimeSmall.script.scala


### Metadata

In order to work properly, this notebook requires some edition of the metadata, in particular you should add the custom dependencies (as bellow) in order to load H2O and avoid interferences with spark versions. 
Note that we also have to load dependencies from sparkling-water-examples, as scala REPL isn't able to handle new MRTask classes.
You also need to pass custom spark config parameters inorder to disable H2O REPL and to specify the port of the H2O Flow UI.

```
"customLocalRepo": "/tmp/spark-notebook",
"customDeps": [
  "ai.h2o % sparkling-water-core_2.11 % 2.0.2",
  "ai.h2o % sparkling-water-examples_2.11 % 2.0.2",
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

import org.joda.time.DateTimeConstants._
import org.joda.time.format.DateTimeFormat
import org.joda.time.{DateTimeZone, MutableDateTime}
import water.Key
import water.MRTask
import water.fvec.{Chunk, NewChunk, Vec}
import water.parser.{BufferedString, ParseSetup}
import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport, SparklingWaterApp}
import water.support.H2OFrameSupport._

import java.io.File
import scala.util.Try

import org.apache.spark.examples.h2o.{Crime, ChicagoCrimeApp, RefineDateColumn} //needed to define MRTask
```


><pre>
> import org.apache.commons.io.FileUtils
> import org.apache.spark.SparkFiles
> import org.apache.spark.h2o._
> import org.apache.spark.sql.{DataFrame, SQLContext, Row}
> import org.joda.time.DateTimeConstants._
> import org.joda.time.format.DateTimeFormat
> import org.joda.time.{DateTimeZone, MutableDateTime}
> import water.Key
> import water.MRTask
> import water.fvec.{Chunk, NewChunk, Vec}
> import water.parser.{BufferedString, ParseSetup}
> import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport, SparklingWaterApp}
> import water.support.H2OFrameSupport._
> import java.io.File
> import scala.util.Try
> import org.apache.spark.examples.h2o.{Crime, ChicagoCrimeApp, RefineDateColumn}
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
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@6895d181
> import sqlContext.implicits._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_-1207059075
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



### Helper functions

We define a helper function to lock and update H2O Frames


```scala
def withLockAndUpdate[T <: Frame](fr: T)(f: T => Any): T = {
  fr.write_lock()
  f(fr)
  // Update frame in DKV
  fr.update()
  fr.unlock()
  fr
}
```


><pre>
> withLockAndUpdate: [T <: org.apache.spark.h2o.Frame](fr: T)(f: T => Any)T
> </pre>



### Weather Data

- We download the weather data from the URL to a temporary local file
- We create a H2OFrame from the local file using the super-fast advanced H2O CSV parser
- We drop the first column of the Frame
- We create and register a SQL table from the H2OFrame
- We show the first 10 rows of the table


```scala
val weatherURL = new java.net.URL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoAllWeather.csv")
val weatherFile = new File("/tmp/spark-notebook/chicagoAllWeather.csv");
FileUtils.copyURLToFile(weatherURL, weatherFile) //download the file from URL
val weatherFrame:H2OFrame = new H2OFrame(weatherFile)
withLockAndUpdate(weatherFrame){  
  _.remove(0).remove() // Remove first column since we do not need it
}
val weatherTable = asDataFrame(weatherFrame)(sqlContext)
weatherTable.createOrReplaceTempView("chicagoWeather")

weatherTable.take(10)
```


><pre>
> weatherURL: java.net.URL = https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoAllWeather.csv
> weatherFile: java.io.File = /tmp/spark-notebook/chicagoAllWeather.csv
> weatherFrame: org.apache.spark.h2o.H2OFrame =
> Frame key: chicagoAllWeather.hex
>    cols: 6
>    rows: 5162
>  chunks: 32
>    size: 41861
> 
> weatherTable: org.apache.spark.sql.DataFrame = [month: tinyint, day: tinyint ... 4 more fields]
> res7: Array[org.apache.spark.sql.Row] = Array([1,1,2001,23,14,6], [1,2,2001,18,12,6], [1,3,2001,28,18,8], [1,4,2001,30,24,19], [1,5,2001,36,30,21], [1,6,2001,33,26,19], [1,7,2001,34,28,21], [1,8,2001,26,20,14], [1,9,2001,23,16,10], [1,10,2001,34,26,19])
> </pre>



### Census Data

- We download the census data from the URL to a temporary local file
- We create a H2OFrame from the local file using the super-fast advanced H2O CSV parser
- We clean the column titles to avoid white space or + symbols in it.
- We create and register a SQL table from the H2OFrame
- We show the first 10 rows of the table

```scala
val censusURL = new java.net.URL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCensus.csv")
val censusFile = new File("/tmp/spark-notebook/chicagoCensus.csv");
FileUtils.copyURLToFile(censusURL, censusFile) //download the file from URL
val censusFrame:H2OFrame = new H2OFrame(censusFile)
withLockAndUpdate(censusFrame){  fr =>
  // Rename columns: replace ' ' by '_'
  val colNames = fr.names().map( n => n.trim.replace(' ', '_').replace('+','_'))
  fr._names = colNames
}
val censusTable = asDataFrame(censusFrame)(sqlContext)
censusTable.createOrReplaceTempView("chicagoCensus")

censusTable.take(10)
```


><pre>
> censusURL: java.net.URL = https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCensus.csv
> censusFile: java.io.File = /tmp/spark-notebook/chicagoCensus.csv
> censusFrame: org.apache.spark.h2o.H2OFrame =
> Frame key: chicagoCensus.hex
>    cols: 9
>    rows: 78
>  chunks: 1
>    size: 3119
> 
> censusTable: org.apache.spark.sql.DataFrame = [Community_Area_Number: tinyint, COMMUNITY_AREA_NAME: string ... 7 more fields]
> res9: Array[org.apache.spark.sql.Row] = Array([1,Rogers Park,7.7,23.6,8.700000000000001,18.2,27.5,23939,39], [2,West Ridge,7.800000000000001,17.2,8.8,20.8,38.5,23040,46], [3,Uptown,3.8000000000000003,24.0,8.9,11.8,22.200000000000003,35787,20], [4,Lincoln Square,3.4000000000000004,10.9,8.200000000000001,13.4,25.5,37524,17], [5,North Center,0.30000000000000...
> </pre>



### Crimes Data

- We download the crimes data from the URL to a temporary local file
- We create a H2OFrame from the local file using the super-fast advanced H2O CSV parser
- We split the string date into several columns using the RefineDateColumn MRTask that is defined  <a href="https://github.com/h2oai/sparkling-water/blob/master/examples/src/main/scala/org/apache/spark/examples/h2o/ChicagoCrimeApp.scala">here</a>
- We clean the column titles to avoid white space or + symbols in it
- We create and register a SQL table from the H2OFrame
- We show the first 10 rows of the table

```scala
import org.joda.time.DateTimeConstants._
val datePattern: String = "MM/dd/yyyy hh:mm:ss a"
val dateTimeZone: String = "Etc/UTC"

val crimeURL = new java.net.URL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCrimes10k.csv")
val crimeFile = new File("/tmp/spark-notebook/chicagoCrimes10k.csv");
FileUtils.copyURLToFile(crimeURL, crimeFile) //download the file from URL

//adapt the parse setup
val parseSetup = water.fvec.H2OFrame.parserSetup(crimeFile)
val colNames = parseSetup.getColumnNames
val typeNames = parseSetup.getColumnTypes
colNames.indices.foreach { idx =>
  if (colNames(idx) == "Date") typeNames(idx) = Vec.T_STR
}
val crimeFrame:H2OFrame = new H2OFrame(parseSetup, crimeFile)

withLockAndUpdate(crimeFrame){  fr =>
  // Refine date into multiple columns
  val dateCol = fr.vec(2)
  fr.add(new RefineDateColumn(datePattern, dateTimeZone).doIt(dateCol))  
  // Update names, replace all ' ' by '_'
  val colNames = fr.names().map(n => n.trim.replace(' ', '_'))
  fr._names = colNames
  // Remove Date column
  fr.remove(2).remove()
}

val crimeTable = h2oContext.asDataFrame(crimeFrame)(sqlContext)
crimeTable.createOrReplaceTempView("chicagoCrime")

crimeTable.take(10)
```


><pre>
> import org.joda.time.DateTimeConstants._
> datePattern: String = MM/dd/yyyy hh:mm:ss a
> dateTimeZone: String = Etc/UTC
> crimeURL: java.net.URL = https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/chicagoCrimes10k.csv
> crimeFile: java.io.File = /tmp/spark-notebook/chicagoCrimes10k.csv
> parseSetup: water.parser.ParseSetup = ParserInfo{name='CSV', prior=2147483647, isParallelParseSupported=true}
> colNames: Array[String] = Array(ID, Case Number, Date, Block, IUCR, Primary Type, Description, Location Description, Arrest, Domestic, Beat, District, Ward, Community Area, FBI Code, X Coordinate, Y Coordinate, Year, Updated On, Latitude, Longitude, Location)
> typeNames: Array[Byte] = Array(3, 2, 2, 4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 4)
> crimeFrame: org.apa...
> </pre>



### Table Join

We use spark SQL to join crime data with weather and census tables.  We display the first entries of the joined table.

```scala

val crimeWeather = sqlContext.sql(
  """SELECT
a.Year, a.Month, a.Day, a.WeekNum, a.HourOfDay, a.Weekend, a.Season, a.WeekDay,
a.IUCR, a.Primary_Type, a.Location_Description, a.Community_Area, a.District,
a.Arrest, a.Domestic, a.Beat, a.Ward, a.FBI_Code,
b.minTemp, b.maxTemp, b.meanTemp,
c.PERCENT_AGED_UNDER_18_OR_OVER_64, c.PER_CAPITA_INCOME, c.HARDSHIP_INDEX,
c.PERCENT_OF_HOUSING_CROWDED, c.PERCENT_HOUSEHOLDS_BELOW_POVERTY,
c.PERCENT_AGED_16__UNEMPLOYED, c.PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA
FROM chicagoCrime a
JOIN chicagoWeather b
ON a.Year = b.year AND a.Month = b.month AND a.Day = b.day
JOIN chicagoCensus c
ON a.Community_Area = c.Community_Area_Number""".stripMargin)

crimeWeather.take(10)
```


><pre>
> crimeWeather: org.apache.spark.sql.DataFrame = [Year: smallint, Month: tinyint ... 26 more fields]
> res13: Array[org.apache.spark.sql.Row] = Array([2015,1,23,4,22,0,Winter,5,null,WEAPONS VIOLATION,ALLEY,31,12,true,false,1234,25,15,29,31,30,32.6,16444,76,9.600000000000001,25.8,15.8,40.7], [2015,1,23,4,19,0,Winter,5,4625,OTHER OFFENSE,SIDEWALK,31,10,true,false,1034,25,26,29,31,30,32.6,16444,76,9.600000000000001,25.8,15.8,40.7], [2015,1,23,4,19,0,Winter,5,320,ROBBERY,SMALL RETAIL STORE,31,10,false,false,1034,25,3,29,31,30,32.6,16444,76,9.600000000000001,25.8,15.8,40.7], [2015,1,23,4,18,0,Winter,5,1310,CRIMINAL DAMAGE,RESTAURANT,31,12,false,false,1235,25,14,29,31,30,32.6,16444,76,9.600000000000001,25.8,15.8,40.7], [2015,1,23,4,18,0,Winter,5,610,BURGLARY,RESIDENCE,31,12,false,false,1234,25,5,...
> </pre>



### Training/Validation datasets

- We convert the joined table back to an H2O Frame
- String columns are converted to categorical variables
- We split the H2OFrame into a training (80%) and a validation (20%) datasets

```scala
val crimeWeatherDF:H2OFrame = crimeWeather
withLockAndUpdate(crimeWeatherDF){allStringVecToCategorical}

val frs = splitFrame(crimeWeatherDF, Array[String]("train.hex", "test.hex"), Array[Double](0.8, 0.2))
val (train, test) = (frs(0), frs(1))
```


><pre>
> crimeWeatherDF: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_58
>    cols: 28
>    rows: 9999
>  chunks: 200
>    size: 666569
> 
> frs: Array[water.fvec.Frame] =
> Array(Frame key: train.hex
>    cols: 28
>    rows: 7993
>  chunks: 199
>    size: 618368
> , Frame key: test.hex
>    cols: 28
>    rows: 2006
>  chunks: 199
>    size: 480713
> )
> train: water.fvec.Frame =
> Frame key: train.hex
>    cols: 28
>    rows: 7993
>  chunks: 199
>    size: 618368
> 
> test: water.fvec.Frame =
> Frame key: test.hex
>    cols: 28
>    rows: 2006
>  chunks: 199
>    size: 480713
> </pre>



### Grandient Boosting Model configuration

- We create a GBM model to predict the value of the "Arrest" column based on the other data.
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
gbmParams._valid = test
gbmParams._response_column = "Arrest"
gbmParams._ntrees = 10
gbmParams._max_depth = 6
gbmParams._distribution = DistributionFamily.bernoulli

val gbm = new GBM(gbmParams)
val gbmModel = gbm.trainModel.get

val trainMetricsGBM = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel,train)
val testMetricsGBM = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel, test)
```


><pre>
> import _root_.hex.tree.gbm.GBM
> import _root_.hex.tree.gbm.GBMModel.GBMParameters
> import _root_.hex.genmodel.utils.DistributionFamily
> import _root_.hex.{Model, ModelMetricsBinomial}
> import water.support.ModelMetricsSupport
> gbmParams: hex.tree.gbm.GBMModel.GBMParameters = hex.tree.gbm.GBMModel$GBMParameters@1d0d8af4
> gbmParams._train: water.Key[water.fvec.Frame] = train.hex
> gbmParams._valid: water.Key[water.fvec.Frame] = test.hex
> gbmParams._response_column: String = Arrest
> gbmParams._ntrees: Int = 10
> gbmParams._max_depth: Int = 6
> gbmParams._distribution: hex.genmodel.utils.DistributionFamily = bernoulli
> gbm: hex.tree.gbm.GBM = hex.tree.gbm.GBM@521a87a9
> gbmModel: hex.tree.gbm.GBMModel =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: GBM_model_1485170375149_1
>  frame id: train.hex
> ...
> </pre>



### Deeplearning Model configuration

- We create a deeplearning model to predict the value of the "Arrest" column based on the other data.
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
dlParams._valid = test
dlParams._response_column = "Arrest"
dlParams._epochs = 10
dlParams._l1 = 0.0001
dlParams._l2 = 0.0001
dlParams._activation = Activation.RectifierWithDropout
dlParams._hidden = Array(200,200)

// Create a job
val dl = new DeepLearning(dlParams)
val dlModel = dl.trainModel.get

val trainMetricsDL = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel,train)
val testMetricsDL = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, test)
```


><pre>
> import _root_.hex.deeplearning.DeepLearning
> import _root_.hex.deeplearning.DeepLearningModel
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
> import _root_.hex.{Model, ModelMetricsBinomial}
> import water.support.ModelMetricsSupport
> dlParams: hex.deeplearning.DeepLearningModel.DeepLearningParameters = hex.deeplearning.DeepLearningModel$DeepLearningParameters@6c4bfc0b
> dlParams._train: water.Key[water.fvec.Frame] = train.hex
> dlParams._valid: water.Key[water.fvec.Frame] = test.hex
> dlParams._response_column: String = Arrest
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
    test  AUC = ${testMetricsGBM.auc}
  DL:
    train AUC = ${trainMetricsDL.auc}
    test  AUC = ${testMetricsDL.auc}
  """.stripMargin)
```


><pre>
> Model performance:
>   GBM:
>     train AUC = 0.9331019683973867
>     test  AUC = 0.9006151473283248
>   DL:
>     train AUC = 0.9164267214594475
>     test  AUC = 0.8900188777908906
>   
> </pre>



# Model testing

We can now make use of the model to predict the probability of being arrest as function of the date, place and type of crime.

### Scoring helper function

We define a function that takes as input a Crime object and a model and returns the probability of arrestlebt for that crime

```scala
def scoreEvent(crime: Crime, model: Model[_,_,_], censusTable: DataFrame) (implicit sqlContext: SQLContext, h2oContext: H2OContext): Float = {
    import h2oContext.implicits._
    import sqlContext.implicits._
    // Create a single row table
    val srdd: DataFrame = sqlContext.sparkContext.parallelize(Seq(crime)).toDF
    // Join table with census data
    val row: H2OFrame = censusTable.join(srdd).where('Community_Area === 'Community_Area_Number) //.printSchema
    // Transform all string columns into categorical
    withLockAndUpdate(row){allStringVecToCategorical}

    val predictTable = model.score(row)
    val probOfArrest = predictTable.vec("true").at(0)

    probOfArrest.toFloat
}
```


><pre>
> scoreEvent: (crime: org.apache.spark.examples.h2o.Crime, model: hex.Model[_, _, _], censusTable: org.apache.spark.sql.DataFrame)(implicit sqlContext: org.apache.spark.sql.SQLContext, implicit h2oContext: org.apache.spark.h2o.H2OContext)Float
> </pre>



### List of crime

We manually enter a list of crime to be tested.  Those are encoded under the crime format defined <a href="https://github.com/h2oai/sparkling-water/blob/915f4fb7627befd5a6384c90e6d6f71a53b61d02/examples/src/main/scala/org/apache/spark/examples/h2o/ChicagoCrimeApp.scala#L315"> here</a>


```scala
val crimeExamples = Seq(
  Crime("02/08/2015 11:43:58 PM", 1811, "NARCOTICS", "STREET",false, 422, 4, 7, 46, 18),
  Crime("02/08/2015 11:00:39 PM", 1150, "DECEPTIVE PRACTICE", "RESIDENCE",false, 923, 9, 14, 63, 11)
)
```


><pre>
> crimeExamples: Seq[org.apache.spark.examples.h2o.Crime] = List(Crime(2015,2,8,6,23,1,Winter,7,1811,NARCOTICS,STREET,false,422,4,7,46,18,None,None,None), Crime(2015,2,8,6,23,1,Winter,7,1150,DECEPTIVE PRACTICE,RESIDENCE,false,923,9,14,63,11,None,None,None))
> </pre>



### Arrest Probability

We loop on the crimes and use the scoreEvent function to get the probability of arrest for each crime

```scala
for (crime <- crimeExamples) {
  val arrestProbGBM = 100*scoreEvent(crime, gbmModel, censusTable)(sqlContext, h2oContext)
  val arrestProbDL = 100*scoreEvent(crime, dlModel, censusTable)(sqlContext, h2oContext)
  println(
    s"""
Crime: $crime
  Probability of arrest best on DeepLearning: ${arrestProbDL} %
  Probability of arrest best on GBM: ${arrestProbGBM} %

    """.stripMargin)
}
```


><pre>
> 
> Crime: Crime(2015,2,8,6,23,1,Winter,7,1811,NARCOTICS,STREET,false,422,4,7,46,18,None,None,None)
>   Probability of arrest best on DeepLearning: 99.992905 %
>   Probability of arrest best on GBM: 75.25859 %
> 
>     
> 
> Crime: Crime(2015,2,8,6,23,1,Winter,7,1150,DECEPTIVE PRACTICE,RESIDENCE,false,923,9,14,63,11,None,None,None)
>   Probability of arrest best on DeepLearning: 1.4065968 %
>   Probability of arrest best on GBM: 12.035014 %
> 
>     
> </pre>



### All done