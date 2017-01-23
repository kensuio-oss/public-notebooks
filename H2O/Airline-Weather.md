### Intro:

This example is the Strata Airlines With Weather demo [1] of H2O Sparkling-water that is integrated into the spark-notebook. In this notebook we will see:

- how to perform some text parsing
- how to build H2OFrame/DataFrame from complex object classes
- how to join two tables using spark SQL
- how to split a dataset into a training, validation and test sets
- how to define, train and use Deeplearning and Gradient Boosting models
- how to merge back model predictions and features into a single table

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2]

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/scripts/StrataAirlines.script.scala


### Metadata

In order to work properly, this notebook requires some edition of the metadata, in particular you should add the custom dependencies (as bellow) in order to load H2O and avoid interferences with spark versions.  You also need to pass custom spark config parameters inorder to disable H2O REPL and to specify the port of the H2O Flow UI.

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
import org.apache.spark.SparkFiles
import org.apache.spark.h2o._
import org.apache.spark.sql.{DataFrame, SQLContext, Row}
import water.Key
import java.io.File

import water.support.SparkContextSupport.addFiles
import water.support.H2OFrameSupport._

import scala.util.Try
```


><pre>
> import org.apache.spark.SparkFiles
> import org.apache.spark.h2o._
> import org.apache.spark.sql.{DataFrame, SQLContext, Row}
> import water.Key
> import java.io.File
> import water.support.SparkContextSupport.addFiles
> import water.support.H2OFrameSupport._
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
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@48ca3275
> import sqlContext.implicits._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_915088449
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



### Airlines data

- We define a Airlines class that is used to store all the airlines data
- We create an object to parse the data into Airlines objects
- We read the airlines data from the internet
- We select all data for which the destination is Chigago ORD
- We show the first 10 rows of the table

```scala
import scala.util.Try

class Airlines (
                val Year              :Option[Int],
                val Month             :Option[Int],
                val DayofMonth        :Option[Int],
                val DayOfWeek         :Option[Int],
                val DepTime           :Option[Int],
                val CRSDepTime        :Option[Int],
                val ArrTime           :Option[Int],
                val CRSArrTime        :Option[Int],
                val UniqueCarrier     :Option[String],
                val FlightNum         :Option[Int],
                val TailNum           :Option[Int],
                val ActualElapsedTime :Option[Int],
                val CRSElapsedTime    :Option[Int],
                val AirTime           :Option[Int],
                val ArrDelay          :Option[Int],
                val DepDelay          :Option[Int],
                val Origin            :Option[String],
                val Dest              :Option[String],
                val Distance          :Option[Int],
                val TaxiIn            :Option[Int],
                val TaxiOut           :Option[Int],
                val Cancelled         :Option[Int],
                val CancellationCode  :Option[Int],
                val Diverted          :Option[Int],
                val CarrierDelay      :Option[Int],
                val WeatherDelay      :Option[Int],
                val NASDelay          :Option[Int],
                val SecurityDelay     :Option[Int],
                val LateAircraftDelay :Option[Int],
                val IsArrDelayed      :Option[Boolean],
                val IsDepDelayed      :Option[Boolean]
                
                ) extends Product with Serializable {
  
  override def canEqual(that: Any):Boolean = that.isInstanceOf[Airlines]
  override def productArity: Int = 31
  override def productElement(n: Int) = n match {
    case  0 => Year
    case  1 => Month
    case  2 => DayofMonth
    case  3 => DayOfWeek
    case  4 => DepTime
    case  5 => CRSDepTime
    case  6 => ArrTime
    case  7 => CRSArrTime
    case  8 => UniqueCarrier
    case  9 => FlightNum
    case 10 => TailNum
    case 11 => ActualElapsedTime
    case 12 => CRSElapsedTime
    case 13 => AirTime
    case 14 => ArrDelay
    case 15 => DepDelay
    case 16 => Origin
    case 17 => Dest
    case 18 => Distance
    case 19 => TaxiIn
    case 20 => TaxiOut
    case 21 => Cancelled
    case 22 => CancellationCode
    case 23 => Diverted
    case 24 => CarrierDelay
    case 25 => WeatherDelay
    case 26 => NASDelay
    case 27 => SecurityDelay
    case 28 => LateAircraftDelay
    case 29 => IsArrDelayed
    case 30 => IsDepDelayed
    case  _ => throw new IndexOutOfBoundsException(n.toString)
  }
  override def toString: String = {
    val sb = new StringBuffer
    for( i <- 0 until productArity )
      sb.append(productElement(i)).append(',')
    sb.toString
  }

  def isWrongRow():Boolean = (0 until productArity).map( idx => productElement(idx)).forall(e => e==None)
}


object AirlinesParse extends Serializable {

  private def int(s: String): Option[Int] = Try(s.toInt).toOption
  private def str(s: String): Option[String] = Try(s.toString).toOption
  private def bool(s: String): Option[Boolean] = Try(s=="YES").toOption  

  def apply(row: Array[String]): Airlines = {
    new Airlines(int (row( 0)), // Year
      int (row( 1)), // Month
      int (row( 2)), // DayofMonth
      int (row( 3)), // DayOfWeek
      int (row( 4)), // DepTime
      int (row( 5)), // CRSDepTime
      int (row( 6)), // ArrTime
      int (row( 7)), // CRSArrTime
      str (row( 8)), // UniqueCarrier
      int (row( 9)), // FlightNum
      int (row(10)), // TailNum
      int (row(11)), // ActualElapsedTime
      int (row(12)), // CRSElapsedTime
      int (row(13)), // AirTime
      int (row(14)), // ArrDelay
      int (row(15)), // DepDelay
      str (row(16)), // Origin
      str (row(17)), // Dest
      int (row(18)), // Distance
      int (row(19)), // TaxiIn
      int (row(20)), // TaxiOut
      int (row(21)), // Cancelled
      int (row(22)), // CancellationCode
      int (row(23)), // Diverted
      int (row(24)), // CarrierDelay
      int (row(25)), // WeatherDelay
      int (row(26)), // NASDelay
      int (row(27)), // SecurityDelay
      int (row(28)), // LateAircraftDelay
      bool(row(29)), // IsArrDelayed
      bool(row(30)) // IsDepDelayed     
      )
  }
}
```


><pre>
> import scala.util.Try
> defined class Airlines
> defined object AirlinesParse
> </pre>




```scala
val airlinesFileURL = new java.net.URL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/year2005.csv.gz")
val airlinesData = sc.parallelize(scala.io.Source.fromInputStream(new java.util.zip.GZIPInputStream(airlinesFileURL.openStream)).getLines.toList).cache()
val airlinesTable = airlinesData.map(_.split(",")).map(row => AirlinesParse(row)).filter(!_.isWrongRow())
val flightsToORD = airlinesTable.filter(f => f.Dest==Some("ORD"))
flightsToORD.take(10)
```


><pre>
> airlinesFileURL: java.net.URL = https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/year2005.csv.gz
> airlinesData: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[4] at parallelize at <console>:96
> airlinesTable: org.apache.spark.rdd.RDD[Airlines] = MapPartitionsRDD[7] at filter at <console>:97
> flightsToORD: org.apache.spark.rdd.RDD[Airlines] = MapPartitionsRDD[8] at filter at <console>:98
> res4: Array[Airlines] = Array(Some(2005),Some(1),Some(1),Some(6),Some(1224),Some(1217),Some(1750),Some(1759),Some(AA),Some(1400),None,Some(206),Some(222),Some(171),Some(-9),Some(7),Some(LAS),Some(ORD),Some(1515),Some(12),Some(23),Some(0),None,Some(0),Some(0),Some(0),Some(0),Some(0),Some(0),Some(false),Some(true),, Some(2005),Some(5),Some(31),Some(2),Some(821),Some(...
> </pre>



### Weather data

- We define a Weather class that is used to store all the weather data from Chicago Airport
- We create an object to parse the data into Weather objects
- We read the weather data from the internet
- We show the first 10 rows of the table

```scala
import scala.util.Try

case class Weather(Year:Option[Int],Month:Option[Int],DayofMonth:Option[Int], TmaxF:Option[Int],TminF:Option[Int],TmeanF:Option[Float],
                   PrcpIn:Option[Float],SnowIn:Option[Float],CDD:Option[Float],HDD:Option[Float],GDD:Option[Float]){
    def isWrongRow():Boolean = (0 until productArity).map( idx => productElement(idx)).forall(e => e==None)
}

object WeatherParse extends Serializable {   
  type DATE = (Option[Int], Option[Int], Option[Int]) // Year, Month, Day
  val datePattern1 = """(\d\d\d\d)-(\d\d)-(\d\d)""".r("year", "month", "day")
  val datePattern2 = """(\d+)/(\d+)/(\d\d\d\d)""".r("month", "day", "year")

  private def int(s: String): Option[Int] = Try(s.toInt).toOption
  private def float(s: String): Option[Float] = Try(s.toFloat).toOption
  
  private def parseDate(s: String): Option[DATE] = s match {
      case datePattern1(y,m,d) => Some( (int(y),int(m),int(d)) )
      case datePattern2(m,d,y) => Some( (int(y),int(m),int(d)) )
      case _ => None
  } 
  
  def apply(row: Array[String]): Weather = {   
    val b = if(row.length==9) 0 else 1 // base index
    val d = parseDate(row(b)).getOrElse( (None, None, None) )
    Weather(d._1,
            d._2,
            d._3,
            int  (row(b + 1)),
            int  (row(b + 2)),
            float(row(b + 3)),
            float(row(b + 4)),
            float(row(b + 5)),
            float(row(b + 6)),
            float(row(b + 7)),
            float(row(b + 8))
    )   
  }
}
```


><pre>
> import scala.util.Try
> defined class Weather
> defined object WeatherParse
> </pre>




```scala
val weatherURL  = " https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/Chicago_Ohare_International_Airport.csv" 
val weatherData = sc.parallelize(scala.io.Source.fromURL(weatherURL)("ISO-8859-1").getLines.toList).cache()
val weatherTable = weatherData.map(_.split(",")).map(row => WeatherParse(row)).filter(!_.isWrongRow())
weatherTable.toDF.take(10)
```


><pre>
> weatherURL: String = " https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/Chicago_Ohare_International_Airport.csv"
> weatherData: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[9] at parallelize at <console>:98
> weatherTable: org.apache.spark.rdd.RDD[Weather] = MapPartitionsRDD[12] at filter at <console>:99
> res7: Array[org.apache.spark.sql.Row] = Array([2005,1,1,41,25,33.0,0.31,0.0,0.0,32.0,0.0], [2005,1,2,54,33,43.5,0.08,0.0,0.0,21.5,0.0], [2005,1,3,36,32,34.0,0.36,0.0,0.0,31.0,0.0], [2005,1,4,35,30,32.5,0.05,1.2,0.0,32.5,0.0], [2005,1,5,31,26,28.5,0.38,6.2,0.0,36.5,0.0], [2005,1,6,27,12,19.5,0.19,2.4,0.0,45.5,0.0], [2005,1,7,28,7,17.5,null,null,0.0,47.5,0.0], [2005,1,8,27,24,25.5,null,null,0.0,39.5,0.0], [2005,1,9,34,26,30.0,null,0.0,0.0,35.0,0.0]...
> </pre>



### Table Join

We use spark SQL to join flight and weather data in spark based on the data.  We display the first entry of the joined table.

```scala
flightsToORD.toDF.createOrReplaceTempView("FlightsToORD")
weatherTable.toDF.createOrReplaceTempView("WeatherORD")

// Perform SQL Join on both tables
val bigTable = sqlContext.sql(
  """SELECT
f.Year,f.Month,f.DayofMonth,
f.CRSDepTime,f.CRSArrTime,f.CRSElapsedTime,
f.UniqueCarrier,f.FlightNum,f.TailNum,
f.Origin,f.Distance,
w.TmaxF,w.TminF,w.TmeanF,w.PrcpIn,w.SnowIn,w.CDD,w.HDD,w.GDD,
f.IsDepDelayed
FROM FlightsToORD f
JOIN WeatherORD w
ON f.Year=w.Year AND f.Month=w.Month AND f.DayofMonth=w.DayofMonth""".stripMargin)

bigTable.take(10)
```


><pre>
> bigTable: org.apache.spark.sql.DataFrame = [Year: int, Month: int ... 18 more fields]
> res9: Array[org.apache.spark.sql.Row] = Array([2005,1,15,1050,1151,121,DH,1158,null,IAD,589,15,4,9.5,0.0,0.0,0.0,55.5,0.0,true], [2005,1,15,835,1000,85,MQ,4042,null,SGF,438,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], [2005,1,15,1228,1315,107,MQ,4168,null,BUF,473,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], [2005,1,15,647,822,95,MQ,3954,null,MEM,491,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], [2005,1,15,1100,1236,156,AA,321,null,LGA,733,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], [2005,1,15,1812,1814,62,MQ,4147,null,TVC,224,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], [2005,1,15,1330,1740,190,AA,1298,null,SLC,1249,15,4,9.5,0.0,0.0,0.0,55.5,0.0,true], [2005,1,15,1219,1814,235,AA,2016,null,SNA,1726,15,4,9.5,0.0,0.0,0.0,55.5,0.0,false], ...
> </pre>



### H2O Frames

We convert the SQL table to a H2O Frame and we convert the column "isDepDelayed" to a categorical variable (needed for deeplearning training).
The joined table is then split into three subsets used for training, validation and testing of the model.

```scala

def withLockAndUpdate[T <: Frame](fr: T)(f: T => Any): T = {
  fr.write_lock()
  f(fr)
  // Update frame in DKV
  fr.update()
  fr.unlock()
  fr
}

val joinedH2OFrame:H2OFrame = bigTable
withLockAndUpdate(joinedH2OFrame){ fr => fr.replace(19, fr.vec("IsDepDelayed").toCategoricalVec)}
```


><pre>
> withLockAndUpdate: [T <: org.apache.spark.h2o.Frame](fr: T)(f: T => Any)T
> joinedH2OFrame: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_44
>    cols: 20
>    rows: 4690
>  chunks: 200
>    size: 468905
> 
> res11: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_44
>    cols: 20
>    rows: 4690
>  chunks: 200
>    size: 468905
> </pre>




```scala
import water.support.H2OFrameSupport
val frs = H2OFrameSupport.split(joinedH2OFrame, Array[String]("train.hex", "valid.hex", "test.hex"), Array[Double](0.7,0.2))
val (train, valid, test) = (frs(0), frs(1), frs(2))
joinedH2OFrame.delete()
```


><pre>
> import water.support.H2OFrameSupport
> frs: Array[water.fvec.Frame] =
> Array(Frame key: train.hex
>    cols: 20
>    rows: 3283
>  chunks: 146
>    size: 344618
> , Frame key: valid.hex
>    cols: 20
>    rows: 938
>  chunks: 36
>    size: 90034
> , Frame key: test.hex
>    cols: 20
>    rows: 469
>  chunks: 20
>    size: 46376
> )
> train: water.fvec.Frame =
> Frame key: train.hex
>    cols: 20
>    rows: 3283
>  chunks: 146
>    size: 344618
> 
> valid: water.fvec.Frame =
> Frame key: valid.hex
>    cols: 20
>    rows: 938
>  chunks: 36
>    size: 90034
> 
> test: water.fvec.Frame =
> Frame key: test.hex
>    cols: 20
>    rows: 469
>  chunks: 20
>    size: 46376
> </pre>



### Deeplearning Model configuration

We create a deeplearning model to predict the value of the "isDepDelayed" column based on the other data.

```scala
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation

val dlParams = new DeepLearningParameters()
dlParams._train = train
dlParams._response_column = "IsDepDelayed"
dlParams._variable_importances = true
dlParams._valid = valid
dlParams._epochs = 100
dlParams._activation = Activation.RectifierWithDropout
dlParams._hidden = Array[Int](100, 100)
dlParams._reproducible = true
dlParams._force_load_balance = false

```


><pre>
> import _root_.hex.deeplearning.DeepLearning
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
> import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
> dlParams: hex.deeplearning.DeepLearningModel.DeepLearningParameters = hex.deeplearning.DeepLearningModel$DeepLearningParameters@599b8557
> dlParams._train: water.Key[water.fvec.Frame] = train.hex
> dlParams._response_column: String = IsDepDelayed
> dlParams._variable_importances: Boolean = true
> dlParams._valid: water.Key[water.fvec.Frame] = valid.hex
> dlParams._epochs: Double = 100.0
> dlParams._activation: hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation = RectifierWithDropout
> dlParams._hidden: Array[Int] = [I@66570fa9
> dlParams._reproducible: Boolean = true
> dlParams._force_load_b...
> </pre>



### Model fit

We fit the model.  Be careful that this step can take a bit of time if there is  a lot data.

You can follow the evolution of the training from the H2O monitoring UI (Check the IP adress and port of the UI in the cell where you created the H2O Context)
From the UI, navigate to Admin/Jobs and then click on your running job to see the past and ongoing steps.

```scala
// Create a job
val dl = new DeepLearning(dlParams, Key.make("dlModel.hex"))
val dlModel = dl.trainModel.get
```


><pre>
> dl: hex.deeplearning.DeepLearning = hex.deeplearning.DeepLearning@59415a0c
> dlModel: hex.deeplearning.DeepLearningModel =
> Model Metrics Type: Binomial
>  Description: Metrics reported on full training frame
>  model id: dlModel.hex
>  frame id: train.hex
>  MSE: 0.21008591
>  RMSE: 0.45835128
>  AUC: 0.6962528
>  logloss: 0.60788965
>  mean_per_class_error: 0.3679355
>  default threshold: 0.2811903655529022
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>            0     1   Error           Rate
>      0   993  1068  0.5182  1,068 / 2,061
>      1   266   956  0.2177    266 / 1,222
> Totals  1259  2024  0.4063  1,334 / 3,283
> Gains/Lift Table (Avg response rate: 37.22 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture R...
> </pre>



### Collect model metrics and evaluate model quality

```scala
import water.support.ModelMetricsSupport
import _root_.hex.ModelMetricsBinomial

val trainMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, train)
val validMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, valid)
println("Training AUC: " + trainMetrics.auc)
println("Validation AUC: " + validMetrics.auc)
```


><pre>
> Training AUC: 0.6962387365388387
> Validation AUC: 0.6533975387907972
> import water.support.ModelMetricsSupport
> import _root_.hex.ModelMetricsBinomial
> trainMetrics: hex.ModelMetricsBinomial =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: dlModel.hex
>  frame id: train.hex
>  MSE: 0.21008591
>  RMSE: 0.45835128
>  AUC: 0.69623876
>  logloss: 0.60788965
>  mean_per_class_error: 0.3684352
>  default threshold: 0.28198716044425964
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>            0     1   Error           Rate
>      0   996  1065  0.5167  1,065 / 2,061
>      1   269   953  0.2201    269 / 1,222
> Totals  1265  2018  0.4063  1,334 / 3,283
> Gains/Lift Table (Avg response rate: 37.22 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture Rate  Cumulative Capture Rate      ...
> </pre>



### Testing the model on the test dataset

We add a new column to the test H2OFrame that holds the model prediction
We then convert back the H2OFrame to a spark dataframe.

```scala
test.add("predictDL", dlModel.score(test)(Symbol("predict")).anyVec())
asDataFrame(test)(sqlContext).take(25)
```


><pre>
> res19: Array[org.apache.spark.sql.Row] = Array([2005,9,5,1233,1425,112,OO,6850,null,TUL,585,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1], [2005,9,5,1154,1750,236,AA,868,null,SEA,1721,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,1], [2005,9,5,1641,1930,169,OO,6832,null,AUS,978,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1], [2005,9,5,647,737,50,MQ,4402,null,GRB,174,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,0], [2005,9,5,1100,1228,148,UA,677,null,LGA,733,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,0], [2005,9,5,1340,1512,92,MQ,3997,null,BNA,409,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1], [2005,9,5,1855,2001,126,MQ,4325,null,CLT,599,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1], [2005,9,5,1335,1345,70,NW,464,null,DTW,235,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,0], [2005,9,5,1410,1459,109,MQ,4020,null,ROC,528,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1], [2005,9,5,...
> </pre>



### Gradient Boosting Machine Model configuration

We create a Gradient Boosting Machine model to predict the value of the "isDepDelayed" column based on the other features.

```scala
import _root_.hex.tree.gbm.GBM
import _root_.hex.tree.gbm.GBMModel.GBMParameters

val gbmParams = new GBMParameters()
gbmParams._train = train
gbmParams._response_column = "IsDepDelayed"
gbmParams._valid = valid
gbmParams._ntrees = 100
//gbmParams._learn_rate = 0.01f
```


><pre>
> import _root_.hex.tree.gbm.GBM
> import _root_.hex.tree.gbm.GBMModel.GBMParameters
> gbmParams: hex.tree.gbm.GBMModel.GBMParameters = hex.tree.gbm.GBMModel$GBMParameters@73570ff0
> gbmParams._train: water.Key[water.fvec.Frame] = train.hex
> gbmParams._response_column: String = IsDepDelayed
> gbmParams._valid: water.Key[water.fvec.Frame] = valid.hex
> gbmParams._ntrees: Int = 100
> </pre>



### Model fit

We fit the model. Be careful that this step can take a bit of time if there is  a lot data.
You can follow the evolution of the training from the H2O monitoring UI (Check the IP adress and port of the UI in the cell where you created the H2O Context) From the UI, navigate to Admin/Jobs and then click on your running job to see the past and ongoing steps.

```scala
val gbm = new GBM(gbmParams)
val gbmModel = gbm.trainModel.get
```


><pre>
> gbm: hex.tree.gbm.GBM = hex.tree.gbm.GBM@1a6b6481
> gbmModel: hex.tree.gbm.GBMModel =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: GBM_model_1483366859969_42
>  frame id: train.hex
>  MSE: 0.12950657
>  RMSE: 0.35987023
>  AUC: 0.9104859
>  logloss: 0.41783765
>  mean_per_class_error: 0.17190322
>  default threshold: 0.3841988742351532
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>            0     1   Error         Rate
>      0  1769   292  0.1417  292 / 2,061
>      1   247   975  0.2021  247 / 1,222
> Totals  2016  1267  0.1642  539 / 3,283
> Gains/Lift Table (Avg response rate: 37.22 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture Rate  Cumulative Capture Rate        Gain  Cumulative Gain
>       1...
> </pre>



### Collect model metrics and evaluate model quality

```scala
import water.support.ModelMetricsSupport
import _root_.hex.ModelMetricsBinomial

val trainMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel, train)
val validMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](gbmModel, valid)
println("Training AUC: " + trainMetrics.auc)
println("Validation AUC: " + validMetrics.auc)
```


><pre>
> Training AUC: 0.9104859081166802
> Validation AUC: 0.6744146347677648
> import water.support.ModelMetricsSupport
> import _root_.hex.ModelMetricsBinomial
> trainMetrics: hex.ModelMetricsBinomial =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: GBM_model_1483366859969_42
>  frame id: train.hex
>  MSE: 0.12950657
>  RMSE: 0.35987023
>  AUC: 0.9104859
>  logloss: 0.41783765
>  mean_per_class_error: 0.17190322
>  default threshold: 0.3841988742351532
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>            0     1   Error         Rate
>      0  1769   292  0.1417  292 / 2,061
>      1   247   975  0.2021  247 / 1,222
> Totals  2016  1267  0.1642  539 / 3,283
> Gains/Lift Table (Avg response rate: 37.22 %):
>   Group  Cumulative Data Fraction  Lower Threshold      Lift  Cumulative Lift  Response Rate  Cumulative Response Rate  Capture Rate  Cumulative Capture Rate...
> </pre>



### Testing the GBM model on the test dataset

We add a new column to the test H2OFrame that holds the model prediction
We then convert back the H2OFrame to a spark dataframe.

```scala
test.add("predictGBM", gbmModel.score(test)(Symbol("predict")).anyVec())
asDataFrame(test)(sqlContext).take(25)
```


><pre>
> res25: Array[org.apache.spark.sql.Row] = Array([2005,9,5,1233,1425,112,OO,6850,null,TUL,585,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1,1], [2005,9,5,1154,1750,236,AA,868,null,SEA,1721,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,1,0], [2005,9,5,1641,1930,169,OO,6832,null,AUS,978,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1,1], [2005,9,5,647,737,50,MQ,4402,null,GRB,174,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,0,0], [2005,9,5,1100,1228,148,UA,677,null,LGA,733,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,0,0], [2005,9,5,1340,1512,92,MQ,3997,null,BNA,409,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1,0], [2005,9,5,1855,2001,126,MQ,4325,null,CLT,599,87,58,72.5,0.0,0.0,7.5,0.0,22.5,0,1,1], [2005,9,5,1335,1345,70,NW,464,null,DTW,235,87,58,72.5,0.0,0.0,7.5,0.0,22.5,1,0,1], [2005,9,5,1410,1459,109,MQ,4020,null,ROC,528,87,58,72.5,0.0,0.0,7.5,0.0,22.5,...
> </pre>



### All done