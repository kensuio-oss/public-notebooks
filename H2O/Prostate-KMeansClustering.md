### Intro:

This example is the Prostate K-Means clustering demo [1] of H2O Sparkling-water that is integrated into the spark-notebook. In this notebook we will see:

- how to perform some text parsing
- how to build H2OFrame/DataFrame from a case class using a parser object
- how to get a subsample using spark SQL
- how to build a K-Means model
- how to merge back model predictions and features into a single table

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2]

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/src/main/scala/org/apache/spark/examples/h2o/ProstateDemo.scala


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
import water._

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
> import water._
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
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@2932c9ce
> import sqlContext.implicits._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_1327485813
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



### Prostate data

- We define a Prostate class that is used to store all the data
- We create an object to parse the data into Prostate objects
- We read the prostate data file from the internet
- We show the first 10 rows of the table

```scala
import scala.util.Try

case class Prostate(ID      :Option[Long]  ,
                    CAPSULE :Option[Int]  ,
                    AGE     :Option[Int]  ,
                    RACE    :Option[Int]  ,
                    DPROS   :Option[Int]  ,
                    DCAPS   :Option[Int]  ,
                    PSA     :Option[Float],
                    VOL     :Option[Float],
                    GLEASON :Option[Int]  ) {
  def isWrongRow():Boolean = (0 until productArity).map( idx => productElement(idx)).forall(e => e==None)
}

object ProstateParse extends Serializable {
  private def int(s: String): Option[Int] = Try(s.toInt).toOption
  private def long(s: String): Option[Long] = Try(s.toLong).toOption  
  private def float(s: String): Option[Float] = Try(s.toFloat).toOption    
  
  val EMPTY = Prostate(None, None, None, None, None, None, None, None, None)
  def apply(row: Array[String]): Prostate = {
    if (row.length < 9) EMPTY
    else Prostate(long(row(0)), int(row(1)), int(row(2)), int(row(3)), int(row(4)), int(row(5)), float(row(6)), float(row(7)), int(row(8)) )
  }
}
```


><pre>
> import scala.util.Try
> defined class Prostate
> defined object ProstateParse
> </pre>




```scala
val prostateURL  = " https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/prostate.csv" 
val prostateData = sc.parallelize(scala.io.Source.fromURL(prostateURL)("ISO-8859-1").getLines.toList).cache()
val prostateTable = prostateData.map(_.split(",")).map(row => ProstateParse(row)).filter(!_.isWrongRow())
prostateTable.toDF.take(10)
```


><pre>
> prostateURL: String = " https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/prostate.csv"
> prostateData: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[4] at parallelize at <console>:100
> prostateTable: org.apache.spark.rdd.RDD[Prostate] = MapPartitionsRDD[7] at filter at <console>:101
> res4: Array[org.apache.spark.sql.Row] = Array([1,0,65,1,2,1,1.4,0.0,6], [2,0,72,1,3,2,6.7,0.0,7], [3,0,70,1,1,2,4.9,0.0,6], [4,0,76,2,2,1,51.2,20.0,7], [5,0,69,1,1,1,12.3,55.9,6], [6,1,71,1,3,2,3.3,0.0,8], [7,0,68,2,4,2,31.9,0.0,7], [8,0,61,2,4,2,66.7,27.2,7], [9,0,69,1,1,1,3.9,24.0,7], [10,0,68,2,1,2,13.0,0.0,6])
> </pre>



### Subsampling

We register the table and use spark SQL to select all rows where "CAPSULE=1"

```scala
prostateTable.toDF.createOrReplaceTempView("prostate_table")

val query = "SELECT * FROM prostate_table WHERE CAPSULE=1"
val result = sqlContext.sql(query)
```


><pre>
> query: String = SELECT * FROM prostate_table WHERE CAPSULE=1
> result: org.apache.spark.sql.DataFrame = [ID: bigint, CAPSULE: int ... 7 more fields]
> </pre>



### K-Means Model

We create a K-Means model (with 3 clusters) from the result table.  We then display the model string and model JSON output.

```scala
import _root_.hex.kmeans.KMeansModel.KMeansParameters
import _root_.hex.kmeans.{KMeans, KMeansModel}

val trainDataFrame:H2OFrame = result
val params = new KMeansParameters
params._train = trainDataFrame._key
params._k = 3


val kmm = new KMeans(params)
val model = kmm.trainModel.get
```


><pre>
> import _root_.hex.kmeans.KMeansModel.KMeansParameters
> import _root_.hex.kmeans.{KMeans, KMeansModel}
> trainDataFrame: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_16
>    cols: 9
>    rows: 153
>  chunks: 8
>    size: 8313
> 
> params: hex.kmeans.KMeansModel.KMeansParameters = hex.kmeans.KMeansModel$KMeansParameters@4bb58e0e
> params._train: water.Key[water.fvec.Frame] = frame_rdd_16
> params._k: Int = 3
> kmm: hex.kmeans.KMeans = hex.kmeans.KMeans@3b971b4c
> model: hex.kmeans.KMeansModel =
> Model Metrics Type: Clustering
>  Description: N/A
>  model id: KMeans_model_1483507639451_1
>  frame id: null
>  MSE: NaN
>  RMSE: NaN
>  total sum of squares: 1216.0
>  total within sum of squares: 889.7066
>  total between sum of squares: 326.2934
>  per cluster sizes: [112, 27, 14]
>  per cluster within sum of squares: [586.8545...
> </pre>




```scala
println(model)
```


><pre>
> Model Metrics Type: Clustering
>  Description: N/A
>  model id: KMeans_model_1483507639451_1
>  frame id: null
>  MSE: NaN
>  RMSE: NaN
>  total sum of squares: 1216.0
>  total within sum of squares: 889.7066
>  total between sum of squares: 326.2934
>  per cluster sizes: [112, 27, 14]
>  per cluster within sum of squares: [586.8545334071273, 201.20419705616098, 101.64788422954715]
> Model Summary:
>  Number of Rows Number of Clusters Number of Categorical Columns Number of Iterations Within Cluster Sum of Squares Total Sum of Squares Between Cluster Sum of Squares
>             153                  3                             0                    5                     889.70661           1216.00000                      326.29339
> Scoring History:
>            Timestamp   Duration Iteration Number of Reassigned Observations Within Cluster Sum Of Squares
>  2017-01-04 06:27:28  0.056 sec         0                               NaN                           NaN
>  2017-01-04 06:27:28  0.138 sec         1                             153.0                    1831.79545
>  2017-01-04 06:27:28  0.169 sec         2                              14.0                     948.91123
>  2017-01-04 06:27:28  0.170 sec         3                               6.0                     895.86664
>  2017-01-04 06:27:28  0.172 sec         4                               2.0                     890.36312
>  2017-01-04 06:27:28  0.173 sec         5                               0.0                     889.70661
> </pre>




```scala
new String(model._output.writeJSON(new AutoBuffer()).buf())
```


><pre>
> res10: String = {"_names":["ID","AGE","RACE","DPROS","DCAPS","PSA","VOL","GLEASON"],"_origNames":null,"_domains":[null,null,null,null,null,null,null,null],"_origDomains":null,"_cross_validation_models":null,"_cross_validation_predictions":null,"_cross_validation_holdout_predictions_frame_id":null,"_cross_validation_fold_assignment_frame_id":null,"_start_time":1483507648880,"_end_time":1483507649009,"_run_time":129,"_model_metrics":[{"name":"modelmetrics_KMeans_model_1483507639451_1@1282103735820096412_on_null@-1925878475650039426","type":"Key"}],"_job":{"_key":{"name":"$0301c0a8000d32d4ffffffff$_97db070f77048a775046a9e485e045a3","type":"Key"},"_checksum":0,"_result":{"name":"KMeans_model_1483507639451_1","type":"Key"},"_typeid":46,"_description":"KMeans","_ready_for_view":true,"_warns":...
> </pre>



### Merging tables

We merge the model output (cluster index) with the training table to see the information side by side.

```scala
trainDataFrame.add("KMeanCluster", model.score(trainDataFrame)(Symbol("predict")).anyVec())
trainDataFrame.update()
asDataFrame(trainDataFrame)(sqlContext).take(25)
```


><pre>
> res12: Array[org.apache.spark.sql.Row] = Array([6,1,71,1,3,2,3.299999952316284,0.0,8,1], [11,1,68,2,4,2,4.0,0.0,7,2], [12,1,72,1,2,2,21.200000762939453,0.0,7,1], [13,1,72,1,4,2,22.700000762939453,0.0,9,1], [14,1,65,1,4,2,39.0,0.0,7,1], [20,1,67,2,3,2,8.600000381469727,25.5,7,2], [21,1,58,1,2,1,3.0999999046325684,0.0,7,0], [22,1,70,0,4,1,67.0999984741211,0.0,7,1], [23,1,74,1,3,1,12.699999809265137,27.5,7,0], [25,1,77,1,1,1,61.099998474121094,58.0,7,0], [31,1,54,1,3,1,8.399999618530273,18.299999237060547,6,0], [34,1,60,1,3,2,9.5,0.0,7,1], [38,1,78,1,1,2,27.200000762939453,0.0,8,1], [39,1,63,1,2,1,35.099998474121094,18.700000762939453,7,0], [40,1,73,1,3,1,4.5,26.399999618530273,7,0], [41,1,66,1,3,1,7.900000095367432,20.799999237060547,7,0], [43,1,71,1,2,1,7.5,0.0,6,0], [45,1,65,2,3,1,83.69...
> </pre>



### All done