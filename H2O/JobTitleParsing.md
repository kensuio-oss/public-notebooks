### Intro:

This example is the craigslistJobTitles demo [1] of H2O Sparkling-water that is integrated into the spark-notebook.
In this notebook we will see:
- how to perform some text parsing
- how to build a Word2Vec model using spark
- how to create a Dataframe, convert it to a H2OFrame
- how to open the H2O flow UI to test ML algorithms running on the H2OFrame

Please note that this code was tested with Scala [2.11.8] and Spark [2.0.2] 

*[1] https://github.com/h2oai/sparkling-water/blob/master/examples/scripts/craigslistJobTitles.script.scala


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
> globalScope.jars: Array[String] = [Ljava.lang.String;@aed98d3
> res2: List[String] = List(/tmp/spark-notebook/org/apache/avro/avro/1.8.0/avro-1.8.0.jar, /tmp/spark-notebook/com/google/code/gson/gson/2.3.1/gson-2.3.1.jar, /tmp/spark-notebook/org/apache/spark/spark-sketch_2.11/2.0.0/spark-sketch_2.11-2.0.0.jar, /tmp/spark-notebook/org/eclipse/jetty/orbit/javax.transaction/1.1.1.v201105210645/javax.transaction-1.1.1.v201105210645.jar, /tmp/spark-notebook/org/eclipse/jetty/orbit/javax.activation/1.1.0.v201105071233/javax.activation-1.1.0.v201105071233.jar, /tmp/spark-notebook/ai/h2o/h2o-persist-hdfs/3.10.0.7/h2o-persist-hdfs-3.10.0.7.jar, /tmp/spark-notebook/org/apache/parquet/parquet-hadoop/1.7.0/parquet-hadoop-1.7.0.jar, /tmp/spark-notebook/com/google/guava/guava/16.0.1/guava-16.0.1.jar, /t...
> </pre>



### Imports and contexts:

Imports the Spark and H2O packages that we will need and create the contexts

```scala
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib

//handle to the spark Session
val spark = SparkSession.builder().getOrCreate()



// Start H2O services
import org.apache.spark.h2o._
implicit val h2oContext = H2OContext.getOrCreate(sc)         
import h2oContext._
import h2oContext.implicits._
import water.support.H2OFrameSupport
```


><pre>
> import org.apache.spark.mllib.feature.Word2Vec
> import org.apache.spark.mllib.feature.Word2VecModel
> import org.apache.spark.mllib.linalg._
> import org.apache.spark.sql.DataFrame
> import org.apache.spark.mllib
> spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@39a47495
> import org.apache.spark.h2o._
> h2oContext: org.apache.spark.h2o.H2OContext =
> 
> Sparkling Water Context:
>  * H2O name: sparkling-water-loicus_-111586916
>  * cluster size: 1
>  * list of used nodes:
>   (executorId, host, port)
>   ------------------------
>   (driver,localhost,54321)
>   ------------------------
> 
>   Open H2O Flow in browser: http://127.0.0.1:54321 (CMD + click in Mac OSX)
> 
> import h2oContext._
> import h2oContext.implicits._
> import water.support.H2OFrameSupport
> </pre>



### Load and Prepare the data

- Load the data to the spark context  (header line is filtered out)
- Make a dataset of job category and a dataset of job description
- Count how many jobs are in each category


```scala
val dataBuffer = scala.io.Source.fromURL("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/craigslistJobTitles.csv")("ISO-8859-1").getLines.toList 
val data = sc.parallelize(dataBuffer).cache.filter(x => !x.contains("category")).map(d => d.split(','))

val jobCategories = data.map(l => l(0))
val jobTitles = data.map(l => l(1))

val labelCounts = jobCategories.map(n => (n, 1)).reduceByKey(_+_).collect
labelCounts
```


><pre>
> dataBuffer: List[String] = List(category,jobtitle, education,After School Supervisor, education,"*****TUTORS NEEDED - FOR ALL SUBJECTS, ALL AGES*****", education,Bay Area Family Recruiter, education,Adult Day Programs/Community Access/Job Coaches, education,General Counselor - Non Tenure track, education,Part-Time Summer Math Teachers/Tutors, education,Preschool Teacher (temp-to-hire), education,"*****TUTORS NEEDED - FOR ALL SUBJECTS, ALL AGES*****", education,Private Teachers and Tutors Needed in the South Bay, education,Art Therapist at Esther B. Clark School, education,Tutoring Position Available!!! Immediate Hire (Hayward), education,Part-Time Art Instructor Needed!!, education,Science Day Camp Leader @ CuriOdyssey, education,After School K-5 Floater, education,Part-Time Behavior In...
> </pre>



### Define the list of stopwords

Additional stop words can be added here

```scala
val stopwords = Set("ax","i","you","edu","s","t","m","subject","can","lines","re","what"
  ,"there","all","we","one","the","a","an","of","or","in","for","by","on"
  ,"but", "is", "in","a","not","with", "as", "was", "if","they", "are", "this", "and", "it", "have"
, "from", "at", "my","be","by","not", "that", "to","from","com","org","like","likes","so")
stopwords.toList
```


><pre>
> stopwords: scala.collection.immutable.Set[String] = Set(for, s, this, in, have, are, is, subject, but, what, if, so, t, all, re, it, a, as, m, or, they, i, that, to, you, likes, was, there, edu, at, can, on, my, by, not, with, from, ax, an, be, org, lines, com, we, like, of, and, one, the)
> res11: List[String] = List(for, s, this, in, have, are, is, subject, but, what, if, so, t, all, re, it, a, as, m, or, they, i, that, to, you, likes, was, there, edu, at, can, on, my, by, not, with, from, ax, an, be, org, lines, com, we, like, of, and, one, the)
> </pre>



### Identify rare words

List all words that appears less than 2 times in the entire dataset : 
- a words dataset is construced (from Job descriptions split in words)
- words are converted to lowercase
- words made of numbers are removed
- distinct words are counted with a map-reduce
- words which appears >=2 are removed
- the lest is collected as a set

```scala
val rareWords = jobTitles.flatMap(t => t.split("""\W+""").map(_.toLowerCase)).filter(word => """[^0-9]*""".r.pattern.matcher(word).matches).
  map(w => (w, 1)).reduceByKey(_+_).
  filter { case (k, v) => v < 2 }.map { case (k, v) => k }.
  collect.toSet

rareWords.toList
```


><pre>
> rareWords: scala.collection.immutable.Set[String] = Set(ihss, compliment, lover, rocketspace, pres, chalkboard, michael, teas, stringer, superintendents, chao, practitioner, precita, kinetic, esther, sweet, mau, joining, lush, used, bayview, eye, trackwork, impressions, sully, workshop, synthetic, printer, marquis, brixton, masters, side, flex, parenting, pulling, panel, maid, clinicians, researcher, decals, healthsherpa, construccion, msc, application, please, interesting, schoolers, mathnasium, weekdays, capabilities, efficiency, butler, excavation, buildings, hewitt, jury, ss, varsity, romero, comm, burner, sci, intelligent, teachable, dot, examiner, geico, maybe, find, apparel, resale, wildlife, ases, invoices, cocineras, ascentis, dietary, weight, lecturers, spay, fashionable, char...
> </pre>



### Define a Tokenizer

We define a tokenize which :
- split job descriptions into a lit of words
- convert words to lowercase
- remove words made of numbers
- remove stopwords
- remove words made of a single character
- remove rare words


```scala
def token(line: String): Seq[String] = {
  line.split("""\W+""")
    .map(_.toLowerCase)
    .filter(word => """[^0-9]*""".r.pattern.matcher(word).matches)
    .filterNot(word => stopwords.contains(word))
    .filter(word => word.size >= 2)
    .filterNot(word => rareWords.contains(word))
}
```


><pre>
> token: (line: String)Seq[String]
> </pre>



### Tokenize the dataset

Job descriptions are pass through the tokenizer and datasets of job labels and descrition words are built


```scala
val XXXwords = data.map(d => (d(0), token(d(1)).toSeq)).filter(s => s._2.length > 0)
val words = XXXwords.map(v => v._2)
val XXXlabels = XXXwords.map(v => v._1)
```


><pre>
> XXXwords: org.apache.spark.rdd.RDD[(String, Seq[String])] = MapPartitionsRDD[19] at filter at <console>:101
> words: org.apache.spark.rdd.RDD[Seq[String]] = MapPartitionsRDD[20] at map at <console>:102
> XXXlabels: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[21] at map at <console>:103
> </pre>



### Word2Vec model

We fit a spark word2Vec model on the words dataset


```scala
val word2vec = new Word2Vec()
val model = word2vec.fit(words)

// Ask the model to find similar words  (Sanity Check)
model.findSynonyms("teacher", 5).toList
```


><pre>
> word2vec: org.apache.spark.mllib.feature.Word2Vec = org.apache.spark.mllib.feature.Word2Vec@68e91b67
> model: org.apache.spark.mllib.feature.Word2VecModel = org.apache.spark.mllib.feature.Word2VecModel@ecbacb4
> res17: List[(String, Double)] = List((cantonese,0.867021186538646), (teachers,0.8554130828243502), (children,0.8523501251410623), (paraprofessional,0.8434342286976014), (speaking,0.842797342371354))
> </pre>



### Helper functions

We define a few helper functions which we will use to convert a job description to a single "word vector" in the Word2Vec space


```scala
def wordToVector (w:String, m: Word2VecModel): Vector = {
  try {
    return m.transform(w)
  } catch {
    case e: Exception => return Vectors.zeros(100)
  }
}

def sumArray (m: Array[Double], n: Array[Double]): Array[Double] = {
  for (i <- 0 until m.length) {m(i) += n(i)}
  return m
}

def divArray (m: Array[Double], divisor: Double) : Array[Double] = {
  for (i <- 0 until m.length) {m(i) /= divisor}
  return m
}
```


><pre>
> wordToVector: (w: String, m: org.apache.spark.mllib.feature.Word2VecModel)org.apache.spark.mllib.linalg.Vector
> sumArray: (m: Array[Double], n: Array[Double])Array[Double]
> divArray: (m: Array[Double], divisor: Double)Array[Double]
> </pre>



### Job Description Word2Vec dataset

We build a single word2vec vector for each job description by
- computing word2vec for each words
- summing up all vectors of a job description  (using a reduceLeft)
- normalizing the resulting vector by the number of meaningful words in the description


```scala
val title_vectors = words.map(x => new DenseVector(divArray(x.map(m => wordToVector(m, model).toArray).reduceLeft(sumArray),x.length)).asInstanceOf[Vector])
```


><pre>
> title_vectors: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = MapPartitionsRDD[34] at map at <console>:111
> </pre>



### Make a DataFrame

We can now create a label/feature dataset using the case class CRAIGSLIST

```scala
case class CRAIGSLIST(targetString: String, a: org.apache.spark.mllib.linalg.Vector)

import org.apache.spark.sql.DataFrame

// Create SQL support
implicit val sqlContext = spark.sqlContext
import sqlContext.implicits._

val resultRDD:DataFrame = XXXlabels.zip(title_vectors).map(v => CRAIGSLIST(v._1, v._2)).toDF
```


><pre>
> defined class CRAIGSLIST
> import org.apache.spark.sql.DataFrame
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@7ad54186
> import sqlContext.implicits._
> resultRDD: org.apache.spark.sql.DataFrame = [targetString: string, a: vector]
> </pre>



### Add a numerical target column

We use the spark mlib StringIndexer to add a numerical target column

```scala

import org.apache.spark.ml.feature.StringIndexer
val indexer = new StringIndexer()
  .setInputCol("targetString")
  .setOutputCol("target")
val resultRDDIndexed = indexer.fit(resultRDD).transform(resultRDD)

resultRDDIndexed.show(5)
```


><pre>
> +------------+--------------------+------+
> |targetString|                   a|target|
> +------------+--------------------+------+
> |   education|[-0.1373687560359...|   3.0|
> |   education|[-0.0962503850460...|   3.0|
> |   education|[-0.0441435514949...|   3.0|
> |   education|[0.00224943831562...|   3.0|
> |   education|[0.01537213691820...|   3.0|
> +------------+--------------------+------+
> only showing top 5 rows
> 
> import org.apache.spark.ml.feature.StringIndexer
> indexer: org.apache.spark.ml.feature.StringIndexer = strIdx_a84cf4b4b0c5
> resultRDDIndexed: org.apache.spark.sql.DataFrame = [targetString: string, a: vector ... 1 more field]
> </pre>



### Make a H2OFrame

We convert the dataframe to two H2OFrames : one for training of the algorithm (80% of the rows) and one for validation (20% of the rows)

```scala
val table:H2OFrame = resultRDDIndexed

val keys = Array[String]("train.hex", "valid.hex")
val ratios = Array[Double](0.8)
val frs = H2OFrameSupport.split(table, keys, ratios)
val (train, valid) = (frs(0), frs(1))
table.delete()

```


><pre>
> table: org.apache.spark.h2o.H2OFrame =
> Frame key: frame_rdd_52
>    cols: 0
>    rows: 13819
>  chunks: 2
>    size: 0
> 
> keys: Array[String] = Array(train.hex, valid.hex)
> ratios: Array[Double] = Array(0.8)
> frs: Array[water.fvec.Frame] =
> Array(Frame key: train.hex
>    cols: 102
>    rows: 11055
>  chunks: 2
>    size: 9044090
> , Frame key: valid.hex
>    cols: 102
>    rows: 2764
>  chunks: 1
>    size: 2278499
> )
> train: water.fvec.Frame =
> Frame key: train.hex
>    cols: 102
>    rows: 11055
>  chunks: 2
>    size: 9044090
> 
> valid: water.fvec.Frame =
> Frame key: valid.hex
>    cols: 102
>    rows: 2764
>  chunks: 1
>    size: 2278499
> </pre>



### H2O Flow UI

We open the H2O Flow User Interface that we case use to test various machine learning algorithm on the created H2O frames

```scala
openFlow
```