import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel, customNaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions.lit

//import org.apache.spark.sql.Row
//import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.Breaks._
import org.apache.log4j.Logger
import org.apache.log4j.Level

//SparkSession
//import org.apache.spark.sql.expressions.Window
//import org.apache.spark.sql.functions.row_number
/**
  * Created by cai on 04.07.17.
  */

object emTrain {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val smoothing = 0.001D    //0.007D
  val NUM_REPLICATIONS: Int = 5
  val newSchema = Seq("label", "maxInd")
  val minImprovement = 0.00001//0.0013 0.0004 0.001
  val maxEpoch = 10
  val hashingSize = 10000
  val splittingBase = 1600000.0
  val trainPath = "/home/cai/DM/twitter/processedTrain"
  val testPath = "/home/cai/DM/twitter/processedTest"
//  val trainPath = "twitter/processedTrain"
//  val testPath = "twitter/processedTest"

//  val stopWordList = "twitter/lowFrequencyWord"

  val VectorType: DataType = new VectorUDT
  val feature = "features"
  val weight = "weight"
  val label = "label"
  val groundTruth = "groundTruth"
  val originalLabel = "polarity"
  val unlabeledWeight = 1.0D
  val weightStructure: StructType = new StructType()
    .add(StructField(feature, VectorType, nullable = false))
    .add(StructField(weight, VectorType, nullable = false))
    .add(StructField(label, DoubleType, nullable = false))
//  val originalLabel = "topic"
//  val weightStructure: StructType = new StructType()
//    .add(StructField("ind", DoubleType, nullable = false))
//    .add(StructField(feature, VectorType, nullable = false))
//    .add(StructField(weight, VectorType, nullable = false))
//    .add(StructField(label, DoubleType, nullable = false))

  def main(args: Array[String]){
    val sc = new SparkContext(new SparkConf().setAppName("emNaiveBayes"))
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

//Schema: ID, topic, words, ind
    val train = sqlContext.read.parquet(trainPath)
    //in the dataset of twitter, there missing example of class label 2
    val test = sqlContext.read.parquet(testPath).where($"polarity" !== "\"2")
    val numofTestItems = test.count().toDouble

    //-----------------------------pre-processing-----------------------------


    val EMCore = new emCore().setMaxEpoch(maxEpoch).setUnlabeledWeight(unlabeledWeight)
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label).setFeaturesCol(feature)

    /*  val model = preproPipe.fit(train)
    model.transform(train).show()
    val corpusTest = model.transform(test)
      .drop("words")
      .drop(stopWordsRemover.getOutputCol)
      .drop(myWordsRemover.getOutputCol)
    val tmpTrain = model.transform(train)
      .drop("words")
      .drop(stopWordsRemover.getOutputCol)
      .drop(myWordsRemover.getOutputCol) */
    val numLabel = (train.groupBy()
      .max(label).first().getDouble(0) + 1.0).toInt

    val weightedTrain = train.map{
      x => {
        val weightVector = new DenseVector(new Array[Double](numLabel))
        val tmpInd = x.getAs[Double](label).toInt
        weightVector.values(tmpInd) = 1.0
        Row(x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
//        Row(x.getAs[Double]("ind"), x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
      }
    }
    val corpusTrain = sqlContext.createDataFrame(weightedTrain, weightStructure)  //Schema: ind, features, weight, label
    val corpusTest = test

//    val allUsedModel = naiveBayes.fit(corpusTrain.withColumn(groundTruth, lit(1.0D))).transform(corpusTest)
//    val allUsedAcc = EMCore.AccuracyCal(allUsedModel.select(label, "prediction"), sc, numofTestItems)
//    println(s"accuracy while using the whole dataset: $allUsedAcc")

    //-----------------------------start EM-----------------------------
//    val minNumTopic = corpusTrain.groupBy(label).max("ind").toDF(newSchema: _*)
//      .groupBy().min(newSchema(1)).first().getDouble(0) + 1
//    val NumSupervisedItems = Array[Int](
//      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    val NumUnsupervisedItems = Array[Int](
      0) //1, 2, 4, 8, 10, 40, 160, 1000,
    val NumSupervisedItems = Array[Int](
      100, 400, 1600, 10000, 100000, 500000)

    val acc = new Array[Double](NUM_REPLICATIONS.toInt)
    val accSup = new Array[Double](NUM_REPLICATIONS.toInt)
    for(numSupervisedItems <- NumSupervisedItems) {
      val splittingWeightPrime = Array[Double](numSupervisedItems, splittingBase-numSupervisedItems)
      val corpusPrime = corpusTrain.randomSplit(splittingWeightPrime)
      val labeledTrain = corpusPrime(0)

      val numLab = labeledTrain.count()
      val numPos = labeledTrain.where($"label" === 1).count()
      println(s"number of labeled instances: $numLab")
      println(s"number of labeled instances in positive class: $numPos")

      for (numUnsupervisedItems <- NumUnsupervisedItems) {
        if(numUnsupervisedItems != 0) println(s"#UNSUPERVISED ITEMS=$numUnsupervisedItems")
        else println(s"Use whole dataset")

        breakable{
          for (trial <- 1 to NUM_REPLICATIONS) {
            println(s"TRIAL=$trial")
            //splitting dataset
//            val splittingWeight = Array[Double](numUnsupervisedItems, splittingBase-numUnsupervisedItems)
//            val splittedDataset = corpusPrime(1).randomSplit(splittingWeight)
              //        val labeledTrain = splittedDataset(0)
            val unlabeledTrain = corpusPrime(1).drop(label).drop(weight)

//            if (trial == 1) { //numSupervisedItems == 4 &&
//              println(labeledTrain.count())
//              println(labeledTrain.where($"label" === 1).count())
//            }
            //for 20 News_Groups
            //        val splittedDataset: Array[DataFrame] = splitDataset(minNumTopic, numSupervisedItems, corpusTrain, sqlContext)
            //        val labeledTrain = splittedDataset(0)
            //        val unlabeledTrain = splittedDataset(1)

            //prediction using EM & NaiveBayes
            val model = EMCore.emTrainNaiveBayes(labeledTrain, unlabeledTrain, numLabel, sc, smoothing, minImprovement
              , corpusTest, numofTestItems) //
            val prediction = model.transform(corpusTest)

            //evaluate
            acc(trial - 1) = EMCore.AccuracyCal(prediction.select(label, "prediction"), sc, numofTestItems)

            //prediction using supervised classification
            val supervisedModel = naiveBayes.fit(labeledTrain.withColumn(groundTruth, lit(1.0D)))
            val supervisedPrediction = supervisedModel.transform(corpusTest)
            accSup(trial - 1) = EMCore.AccuracyCal(supervisedPrediction.select(label, "prediction"), sc, numofTestItems)
            val accToShow = acc(trial - 1)
            val accSupToShow = accSup(trial - 1)
            println(s"Supervised Accuracy: $accSupToShow      EM Accuracy: $accToShow")
            println("-----------------------------------------------------------------------------------------------------")
//            if(numUnsupervisedItems == 1600000) break
          }
        }
        //show accuracy
        val meanAccSup = accSup.sum / NUM_REPLICATIONS
        val meanAcc = acc.sum / NUM_REPLICATIONS
        var sdAcc = 0.0D
        var sdAccSup = 0.0D
        for (k <- 0 until NUM_REPLICATIONS.toInt) {
          sdAcc += math.pow(acc(k) - meanAcc, 2.0)
          sdAccSup += math.pow(accSup(k) - meanAccSup, 2.0)
        }
        sdAcc = math.sqrt(sdAcc / NUM_REPLICATIONS)
        sdAccSup = math.sqrt(sdAccSup / NUM_REPLICATIONS)
        println(s"#Unsup=$numUnsupervisedItems     Supervised mean(acc)=$meanAccSup  sd(Sup)=$sdAccSup    EM mean(acc)=$meanAcc  sd(EM)=$sdAcc")
        println("=====================================================================================================")
        println()
      }
    }
  }

  def splitTwitterDataset(numSupervisedItems: Int, corpusTrain: DataFrame, sqlContext: SQLContext): Array[DataFrame] = {
    import sqlContext.implicits._
    val splittingWeight = Array[Double](numSupervisedItems, splittingBase-numSupervisedItems)
    var tmpArrayOfDF = corpusTrain.where($"label" === 0).randomSplit(splittingWeight)
    var labeledTrain = tmpArrayOfDF(0)
    var unlabeledTrain = tmpArrayOfDF(1)
    tmpArrayOfDF = corpusTrain.where($"label" === 1).randomSplit(splittingWeight)
    labeledTrain = labeledTrain.unionAll(tmpArrayOfDF(0))
    unlabeledTrain = unlabeledTrain.unionAll(tmpArrayOfDF(1))
    Array(labeledTrain,unlabeledTrain)
  }

  //randomly split the dataset, such that each class will have exact same number of labeled documents
  /*
    required column "ind", i.e. assign index to each instance, than the function will randomly choose
    a position to split the dataset
  */
  def splitDataset(minNumTopic: Double, numSupervisedItems: Int, corpusTrain: DataFrame, sqlContext: SQLContext): Array[DataFrame] ={
    import sqlContext.implicits._
    var divideBound = minNumTopic - numSupervisedItems
    if(divideBound > 0){
      divideBound = ((divideBound+1)*math.random).toInt
    }
    else divideBound = 0
    val labeledTrain = corpusTrain.where($"ind" >= divideBound && $"ind" < (divideBound + numSupervisedItems))
    val unlabeledTrain = corpusTrain.where($"ind" < divideBound || $"ind" >= (divideBound + numSupervisedItems))
      .drop(label).drop(weight)
    Array(labeledTrain, unlabeledTrain)
  }
}

