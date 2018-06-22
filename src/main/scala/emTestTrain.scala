import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.customNaiveBayes
import org.apache.spark.ml.feature.{HashingTF, StringIndexer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions.{lit, split}
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}

import scala.util.control.Breaks.breakable

/**
  * Created by cai on 07.02.18.
  */
object emTestTrain {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val smoothing = 0.1D //0.007D
  val NUM_REPLICATIONS: Int = 1
  val newSchema = Seq("label", "maxInd")
  val minImprovement = 0.0
  //0.0008
  //0.0013 0.0004 0.001
  val maxEpoch = 6
  val splittingBase = 1600000.0
  //  val trainPath = "/home/cai/DM/twitter/processedTrain"
  //  val testPath = "/home/cai/DM/twitter/processedTest"
  val labeledPath = "/user/iosifidis/TwitterDatasets/NoRetweetsNewDataset2015/NoRetweets2015GroundTruth_Bigrams/"
  val unlabeledPath = "/user/iosifidis/TwitterDatasets/NoRetweetsNewDataset2015/NoRetweets2015Neutral_Bigrams/"
//  val testPath = "differentUnlabeled/unlabeled100"
//  val labeledPath = "/home/cai/DM/twitter/differentUnlabeled/labeled10"
//  val testPath = "/home/cai/DM/twitter/differentUnlabeled/unlabeled100"


  //  val stopWordList = "twitter/lowFrequencyWord"

  val VectorType: DataType = new VectorUDT
  val word = "words"
  val feature = "features"
  val weight = "weight"
  val originalLabel = "polarity"
  val label = "label"
  val groundTruth = "groundTruth"
  val hashingSize = 1500000 //15000
  val unlabeledWeight = 1.0D
  val weightStructure: StructType = new StructType()
    .add(StructField(feature, VectorType, nullable = false))
    .add(StructField(weight, VectorType, nullable = false))
    .add(StructField(label, DoubleType, nullable = false))

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("emNaiveBayes"))
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

//    val unlabeledPath = args(0)
//    val labeledPath = args(1)

    //Schema: polarity, id, user, features, label
    //    val train = sqlContext.read.parquet(trainPath)
    val rddLabeled = sqlContext.read.text(labeledPath).withColumn("tmp", split($"value", ","))  //"\t" split according to tab
    //Schema: polarity, words (Array[String])
    val dfLabeled = rddLabeled.select(    //1, 3
      $"tmp".getItem(3).as("polarity"),
      $"tmp".getItem(5).as("string")
    ).withColumn("words", split($"string", " ")).drop("string")


    val rddUnlabeled = sqlContext.read.text(unlabeledPath).withColumn("tmp", split($"value", ","))
    val dfUnlabeled = rddUnlabeled.select(
      $"tmp".getItem(3).as("polarity"), //******************
      $"tmp".getItem(5).as("string")
    ).withColumn("words", split($"string", " ")).drop("string")
    rddUnlabeled.take(20).map(line => println(line))

    /*val rddTest = sqlContext.read.text(testPath).withColumn("tmp", split($"value", "\t"))
    val dfTest = rddTest.select(
      $"tmp".getItem(1).as("polarity"),
      $"tmp".getItem(3).as("string")
    ).withColumn("words", split($"string", " ")).drop("string") */

    //-----------------------------pre-processing-----------------------------
    val hashingTF = new HashingTF()
      .setInputCol(word)
      .setOutputCol(feature) //.setNumFeatures(500)
      .setNumFeatures(hashingSize)
    val indexer = new StringIndexer().setInputCol(originalLabel).setOutputCol(label)

//    val preproPipe = new Pipeline().setStages(Array(hashingTF, indexer))  //original label, word, feature, label
//    val model = preproPipe.fit(dfLabeled)
    val model = indexer.fit(dfLabeled)
    val trainLabeled = model.transform(hashingTF.transform(dfLabeled))
      .drop(originalLabel)

    val EMCore = new emCore().setMaxEpoch(maxEpoch).setUnlabeledWeight(unlabeledWeight)
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label).setFeaturesCol(feature)

    val numLabel = (trainLabeled.groupBy()
      .max(label).first().getDouble(0) + 1.0).toInt

    val weightedLabeled = trainLabeled.map {
      x => {
        val weightVector = new DenseVector(new Array[Double](numLabel))
        val tmpInd = x.getAs[Double](label).toInt
        weightVector.values(tmpInd) = 1.0
        Row(x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
        //        Row(x.getAs[Double]("ind"), x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
      }
    }

    val corpusAllLabeled = sqlContext.createDataFrame(weightedLabeled, weightStructure) //Schema: features, weight, label
//      .repartition(200)

    dfUnlabeled.show(false)

    val trainUnlabeled = hashingTF.transform(dfUnlabeled) //features
      .drop(originalLabel)
      .drop(word)
    trainUnlabeled.show(false)
//      .repartition(200)

    val splittingWeight = Array[Double](9, 1)
    val splitted = corpusAllLabeled.randomSplit(splittingWeight)
    val allLabeled = splitted(0)  //val corpusTrain
    val corpusTest = splitted(1)
    val numOfTest= corpusTest.count().toDouble
//    val corpusTest = model.transform(dfTest)
//      .repartition(200)

    //-----------------------------start EM-----------------------------
    //    val minNumTopic = corpusTrain.groupBy(label).max("ind").toDF(newSchema: _*)
    //      .groupBy().min(newSchema(1)).first().getDouble(0) + 1
    //    val NumSupervisedItems = Array[Int](
    //      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
//    val unlabeledSplit = Array[Double](0.4, 0.5)

    val weightUnlabeled = Array[Double](0.1, 0.9)
    val splittedUnlabled = trainUnlabeled.randomSplit(weightUnlabeled)
    val corpusUnlabeled = splittedUnlabled(0)

    val labeledSplit = Array[Double](0.7,0.8,0.9)
    val totalWeight = 1.0
    for (spl <- labeledSplit) {
      val acc = new Array[Double](NUM_REPLICATIONS.toInt)
      val accSup = new Array[Double](NUM_REPLICATIONS.toInt)

      val weightLabeled = Array[Double](spl, totalWeight-spl)
      val splittedLabeled = allLabeled.randomSplit(weightLabeled)
      val corpusLabeled = splittedLabeled(0)


      val showCurrentWeight = (spl * 100).toInt
      println(s"$showCurrentWeight% of unlabeled data used")

      val startTimeMillis = System.currentTimeMillis()

      for (trial <- 1 to NUM_REPLICATIONS) {

        println(s"TRIAL=$trial")
        //      val splittingWeightPrime = Array[Double](numSupervisedItems, splittingBase - numSupervisedItems)
        //      val corpusPrime = corpusTrain.randomSplit(splittingWeightPrime)
        //      val labeledTrain = corpusPrime(0)
        //      val unlabeledTrain = corpusPrime(1).drop(label).drop(weight)

        val numLab = corpusLabeled.count()
        val numZero = corpusLabeled.where($"label" === 0).count()
        val numUnlab = corpusUnlabeled.count()
        val posR = numZero.toDouble / numLab
        println(s"number of labeled instances: $numLab")
        println(s"number of labeled instances in positive class: $numZero")
        println(s"ratio of positive in all: $posR")
        println(s"number of unlabeled instances: $numUnlab")

        //prediction using EM & NaiveBayes
        val model = EMCore.emTrainNaiveBayes(corpusLabeled, corpusUnlabeled, numLabel, sc, smoothing, minImprovement
          , corpusTest, numOfTest) //
        val prediction = model.transform(corpusTest)

        //evaluate
        acc(trial - 1) = EMCore.AccuracyCal(prediction.select(label, "prediction"), sc, numOfTest)

        //prediction using supervised classification
        val supervisedModel = naiveBayes.fit(corpusLabeled.withColumn(groundTruth, lit(1.0D)))
        val supervisedPrediction = supervisedModel.transform(corpusTest)
        accSup(trial - 1) = EMCore.AccuracyCal(supervisedPrediction.select(label, "prediction"), sc, numOfTest)
        val accToShow = acc(trial - 1)
        val accSupToShow = accSup(trial - 1)
        println(s"Supervised Accuracy: $accSupToShow      EM Accuracy: $accToShow")
        println("-----------------------------------------------------------------------------------------------------")
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
      println(s"unlabeled=$showCurrentWeight    Supervised mean(acc)=$meanAccSup  sd(Sup)=$sdAccSup    EM mean(acc)=$meanAcc  sd(EM)=$sdAcc")

      val endTimeMillis = System.currentTimeMillis()
      val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
      println($"time cost: $durationSeconds seconds")
      println("=====================================================================================================")
      println()
    }
  }

  def splitTwitterDataset(numSupervisedItems: Int, corpusTrain: DataFrame, sqlContext: SQLContext): Array[DataFrame] = {
    import sqlContext.implicits._
    val splittingWeight = Array[Double](numSupervisedItems, splittingBase - numSupervisedItems)
    var tmpArrayOfDF = corpusTrain.where($"label" === 0).randomSplit(splittingWeight)
    var labeledTrain = tmpArrayOfDF(0)
    var unlabeledTrain = tmpArrayOfDF(1)
    tmpArrayOfDF = corpusTrain.where($"label" === 1).randomSplit(splittingWeight)
    labeledTrain = labeledTrain.unionAll(tmpArrayOfDF(0))
    unlabeledTrain = unlabeledTrain.unionAll(tmpArrayOfDF(1))
    Array(labeledTrain, unlabeledTrain)
  }

  //randomly split the dataset, such that each class will have exact same number of labeled documents
  /*
    required column "ind", i.e. assign index to each instance, than the function will randomly choose
    a position to split the dataset
  */
  def splitDataset(minNumTopic: Double, numSupervisedItems: Int, corpusTrain: DataFrame, sqlContext: SQLContext): Array[DataFrame] = {
    import sqlContext.implicits._
    var divideBound = minNumTopic - numSupervisedItems
    if (divideBound > 0) {
      divideBound = ((divideBound + 1) * math.random).toInt
    }
    else divideBound = 0
    val labeledTrain = corpusTrain.where($"ind" >= divideBound && $"ind" < (divideBound + numSupervisedItems))
    val unlabeledTrain = corpusTrain.where($"ind" < divideBound || $"ind" >= (divideBound + numSupervisedItems))
      .drop(label).drop(weight)
    Array(labeledTrain, unlabeledTrain)
  }
}
