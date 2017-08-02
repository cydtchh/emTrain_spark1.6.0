import emTrain.newSchema
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, customNaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}


/**
  * Created by cai on 28.07.17.
  */
object testNewNB {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val smoothing = 0.01
  val VectorType: DataType = new VectorUDT
  val feature = "features"
  val weight = "weight"
  val label = "label"
  val weightStructure = new StructType()
    .add(StructField("ind", DoubleType, nullable = false))
    .add(StructField(feature, VectorType, nullable = false))
    .add(StructField(weight, VectorType, nullable = false))
    .add(StructField(label, DoubleType, nullable = false))
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("emNaiveBayes"))
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val train = sqlContext.read.parquet("/home/cai/DM/TrainingSet")
    val test = sqlContext.read.parquet("/home/cai/DM/TestSet")
    val indexer = new StringIndexer().setInputCol("topic").setOutputCol(label)
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("keywords")
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol(feature)//.setNumFeatures(500)
      .setNumFeatures(40000)
    val preproPipe = new Pipeline()
      .setStages(Array(stopWordsRemover, hashingTF, indexer))
    val customNaiveBayes = new customNaiveBayes()
      .setFeaturesCol(hashingTF.getOutputCol)
      .setLabelCol(indexer.getOutputCol)
      .setSmoothing(smoothing)
    val NaiveBayes = new NaiveBayes()
      .setFeaturesCol(hashingTF.getOutputCol)
      .setLabelCol(indexer.getOutputCol)
      .setSmoothing(smoothing)

    val numTestItems = test.count().toDouble

    val processingModel = preproPipe.fit(train)
    val corpusTrain = processingModel.transform(train)
    val corpusTest = processingModel.transform(test)
//--------add weight-----------------------------------------------

    val numLabel = (corpusTrain.groupBy().max(label).first().getDouble(0) + 1.0).toInt
    val weightedCorpus = corpusTrain.map{
      x => {
        val weightVector = new DenseVector(new Array[Double](numLabel))
        val tmpInd = x.getAs[Double](label).toInt
        weightVector.values(tmpInd) = 1.0
        Row(x.getAs[Double]("ind"), x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
      }
    }
    val weightedCorpusTrain = sqlContext.createDataFrame(weightedCorpus, weightStructure)
//-----------------------------------------------------------------
    val minNumTopic = corpusTrain.groupBy(label).max("ind").toDF(newSchema: _*)
      .groupBy().min(newSchema(1)).first().getDouble(0) + 1
    val NumSupervisedItems = Array[Int](
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for(numSupervisedItems <- NumSupervisedItems) {
      var divideBound = minNumTopic - numSupervisedItems
      if(divideBound > 0){
        divideBound = ((divideBound+1)*math.random).toInt
      }
      else divideBound = 0
      val labeledTrain = weightedCorpusTrain.where($"ind" >= divideBound && $"ind" < (divideBound + numSupervisedItems))

      val givenModel = NaiveBayes.fit(labeledTrain)
      val givenPrediction = givenModel.transform(corpusTest)
      val givenAcc = AccuracyCal(givenPrediction.select(label, "prediction"), sc, numTestItems)

      val customModel = customNaiveBayes.fit(labeledTrain)
      val customPrediction = customModel.transform(corpusTest)
      val customAcc = AccuracyCal(customPrediction.select(label, "prediction"), sc, numTestItems)

      println(s"CustomNaiveBayes Accuracy: $customAcc")
      println(s"GivenNaiveBayes Accuracy: $givenAcc")
    }
  }

  def AccuracyCal(dataFrame: DataFrame, sc: SparkContext, numOfItem: Double): Double ={
    val accumulator = sc.accumulator(0.0)
    dataFrame.foreach(x => {
      if (x.getDouble(0) == x.getDouble(1)){
        accumulator.add(1)
      }
    })
    accumulator.value/numOfItem
  }
}
