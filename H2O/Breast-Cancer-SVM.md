### Intro:

This example is the Breast Cancer Spark SVM demo [1] of H2O Sparkling-water that is integrated into the spark-notebook. In this notebook we will see:

- how to load a CSV file directly into a H2OFrame
- how to define and use an SVM model
- how to merge model prediction table with the feature table to have all the data within a single table

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2]

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/src/main/scala/org/apache/spark/examples/h2o/SparkSVMDemo.scala

### Metadata

In order to work properly, this notebook requires some edition of the metadata, in particular you should add the custom dependencies (as bellow) in order to load H2O and avoid interferences with spark versions.  Note that we have dependencies on both sparkling-water-core and sparkling-water-ml.  You also need to pass custom spark config parameters inorder to disable H2O REPL and to specify the port of the H2O Flow UI.  

```
"customLocalRepo": "/tmp/spark-notebook",
"customDeps": [
  "ai.h2o % sparkling-water-core_2.11 % 2.0.2",
  "ai.h2o % sparkling-water-ml_2.11 % 2.0.2",      
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
import org.apache.spark.ml.spark.models.svm.{SVM, SVMParameters} //from Sparkling-water (not spark)
import water.Key
import water.fvec.H2OFrame
import water.support.SparkContextSupport.addFiles
import water.support.H2OFrameSupport._

import java.io.File
import scala.util.Try
```


><pre>
> import org.apache.spark.SparkFiles
> import org.apache.spark.h2o._
> import org.apache.spark.sql.{DataFrame, SQLContext, Row}
> import org.apache.spark.ml.spark.models.svm.{SVM, SVMParameters}
> import water.Key
> import water.fvec.H2OFrame
> import water.support.SparkContextSupport.addFiles
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
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@6b1787c
> import sqlContext.implicits._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_1004153571
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



### Breast Cancer data

- We download the cancer data from the URL to a temporary local file
- We create a H2OFrame from the local file using the super-fast advanced H2O CSV parser
- We show the first 10 rows of the table  (it's converted to a spark DF in order to have a nice rendering of the table in the notebook widget)

```scala
import org.apache.commons.io.FileUtils
val breastCancerURL = new java.net.URL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/bcwd.csv")
val breastCancerFile = new File("/tmp/spark-notebook/bcwd.csv");
FileUtils.copyURLToFile(breastCancerURL, breastCancerFile) //download the file from URL
val breastCancerData:H2OFrame = new H2OFrame(breastCancerFile)
asDataFrame(breastCancerData)(sqlContext).take(10)
```


><pre>
> import org.apache.commons.io.FileUtils
> breastCancerURL: java.net.URL = https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/bcwd.csv
> breastCancerFile: java.io.File = /tmp/spark-notebook/bcwd.csv
> breastCancerData: water.fvec.H2OFrame =
> Frame key: bcwd.hex
>    cols: 10
>    rows: 699
>  chunks: 1
>    size: 7061
> 
> res3: Array[org.apache.spark.sql.Row] = Array([5,1,1,1,2,1,3,1,1,B], [5,4,4,5,7,10,3,2,1,B], [3,1,1,1,2,2,3,1,1,B], [6,8,8,1,3,4,3,7,1,B], [4,1,1,3,2,1,3,1,1,B], [8,10,10,8,7,10,9,7,1,M], [1,1,1,1,2,10,3,1,1,B], [2,1,2,1,2,1,3,1,1,B], [2,1,1,1,2,1,1,1,5,B], [4,2,1,1,2,1,2,1,1,B])
> </pre>



### Categorical column

We tell H2O that the label (last) column is categorical

```scala
breastCancerData.replace(breastCancerData.numCols()-1, breastCancerData.lastVec().toCategoricalVec)
breastCancerData.update()
```


><pre>
> res5: water.fvec.Frame =
> Frame key: bcwd.hex
>    cols: 10
>    rows: 699
>  chunks: 1
>    size: 7061
> </pre>



### SVM Model

- We create a SVM model configuration to predict the label column from the other ones
- We create the SVM model itself
- We train the SVM model

```scala
// Configure Deep Learning algorithm
val parms = new SVMParameters
parms._train = breastCancerData.key
parms._response_column = "label"

val svm = new SVM(parms, h2oContext)
val svmModel = svm.trainModel.get
```


><pre>
> parms: org.apache.spark.ml.spark.models.svm.SVMParameters = org.apache.spark.ml.spark.models.svm.SVMParameters@2731a9c3
> parms._train: water.Key[water.fvec.Frame] = bcwd.hex
> parms._response_column: String = label
> svm: org.apache.spark.ml.spark.models.svm.SVM = org.apache.spark.ml.spark.models.svm.SVM@456dceba
> svmModel: org.apache.spark.ml.spark.models.svm.SVMModel =
> Model Metrics Type: Binomial
>  Description: N/A
>  model id: SVM_model_1483505593914_1
>  frame id: bcwd.hex
>  MSE: 4.057861
>  RMSE: 2.0144134
>  AUC: 0.88748664
>  logloss: NaN
>  mean_per_class_error: 0.14041747
>  default threshold: 0.5703628659248352
>  CM: Confusion Matrix (vertical: actual; across: predicted):
>           B    M   Error      Rate
>      B  432   26  0.0568  26 / 458
>      M   54  187  0.2241  54 / 241
> Totals  486  213  0.114...
> </pre>



### Model Predictions

- We use the model to score each rows of the table
- We display the first entries.

```scala
val predictionH2OFrame = svmModel.score(breastCancerData)
h2oContext.asDataFrame(predictionH2OFrame).take(10)
```


><pre>
> predictionH2OFrame: water.fvec.Frame =
> Frame key: _8a0c51faff3a9cfa91d48702d4c64460
>    cols: 3
>    rows: 699
>  chunks: 1
>    size: 11478
> 
> res12: Array[org.apache.spark.sql.Row] = Array([B,-1.1509096730692234,-2.1509096730692234], [M,-0.07034367987592904,0.929656320124071], [B,-0.38250343368714734,-1.3825034336871473], [M,4.437234562767544,5.437234562767544], [B,-0.8396775949800639,-1.839677594980064], [M,3.4288945842797744,4.4288945842797744], [M,1.0699717275197926,2.0699717275197926], [B,-0.5191292110606585,-1.5191292110606585], [B,-0.7746193963925296,-1.7746193963925296], [B,0.17214830634585576,-0.8278516936541442])
> </pre>



### Merge tables

We merge the prediction table with the cancer data table in order to see the label and prediction side by side.

```scala
breastCancerData.add(predictionH2OFrame)
breastCancerData.update()
h2oContext.asDataFrame(breastCancerData).take(10)
```


><pre>
> res10: Array[org.apache.spark.sql.Row] = Array([5,1,1,1,2,1,3,1,1,B,B,-1.1509096730692234,-2.1509096730692234], [5,4,4,5,7,10,3,2,1,B,M,-0.07034367987592904,0.929656320124071], [3,1,1,1,2,2,3,1,1,B,B,-0.38250343368714734,-1.3825034336871473], [6,8,8,1,3,4,3,7,1,B,M,4.437234562767544,5.437234562767544], [4,1,1,3,2,1,3,1,1,B,B,-0.8396775949800639,-1.839677594980064], [8,10,10,8,7,10,9,7,1,M,M,3.4288945842797744,4.4288945842797744], [1,1,1,1,2,10,3,1,1,B,M,1.0699717275197926,2.0699717275197926], [2,1,2,1,2,1,3,1,1,B,B,-0.5191292110606585,-1.5191292110606585], [2,1,1,1,2,1,1,1,5,B,B,-0.7746193963925296,-1.7746193963925296], [4,2,1,1,2,1,2,1,1,B,B,0.17214830634585576,-0.8278516936541442])
> </pre>



### All done