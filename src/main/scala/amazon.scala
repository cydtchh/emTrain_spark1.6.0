import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.customNaiveBayes
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}

/**
  * Created by cai on 27.11.17.
  */
object amazon {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val smoothing = 0.001D    //0.007D
  val NUM_REPLICATIONS: Int = 10
  val newSchema = Seq("label", "maxInd")
  val minImprovement = 0.001//0.0013 0.0004
  val maxEpoch = 20
  val hashingSize = 10000
  val splittingBase = 1600000.0

//  val trainPath = "/home/cai/DM/amazon/amazonTrain"
//  val testPath = "/home/cai/DM/amazon/amazonTest"
//  val trainPath = "amazon/amazonTrain"
//  val testPath = "amazon/amazonTest"
  val trainPath = "/home/cai/DM/TWENTY_NEWSGROUPS/TrainingSet"    //Schema: ID, topic, words, ind, keywords, features, label
  val testPath = "/home/cai/DM/TWENTY_NEWSGROUPS/TestSet"

  val VectorType: DataType = new VectorUDT
  val feature = "features"
  val weight = "weight"
  val label = "label"
  val groundTruth = "groundTruth"
  val originalLabel = "score"
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
    val test = sqlContext.read.parquet(testPath)
    val numofTestItems = test.count().toDouble

    //-----------------------------pre-processing-----------------------------
    val indexer = new StringIndexer().setInputCol(originalLabel).setOutputCol(label)
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("keywords")
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol(feature)//.setNumFeatures(500)
      .setNumFeatures(hashingSize)
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label).setFeaturesCol(feature)
    val preproPipe = new Pipeline()
      .setStages(Array(stopWordsRemover, hashingTF, indexer))

    val EMCore = new emCore().setMaxEpoch(maxEpoch).setUnlabeledWeight(unlabeledWeight)

    val model = preproPipe.fit(train)
    val corpusTest = model.transform(test)
    val tmpTrain = model.transform(train)
    val numLabel = (tmpTrain.groupBy()
      .max(label).first().getDouble(0) + 1.0).toInt

    val weightedTrain = tmpTrain.map{
      x => {
        val weightVector = new DenseVector(new Array[Double](numLabel))
        val tmpInd = x.getAs[Double](label).toInt
        weightVector.values(tmpInd) = 1.0
        Row(x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
        //        Row(x.getAs[Double]("ind"), x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
      }
    }
    val corpusTrain = sqlContext.createDataFrame(weightedTrain, weightStructure)  //Schema: ind, features, weight, label

//    val allUsedModel = naiveBayes.fit(corpusTrain.withColumn(groundTruth, lit(1.0D))).transform(corpusTest)
//    val allUsedAcc = EMCore.AccuracyCal(allUsedModel.select(label, "prediction"), sc, numofTestItems)
//    println(s"accuracy while using the whole dataset: $allUsedAcc")

    //-----------------------------start EM-----------------------------
    val minNumTopic = corpusTrain.groupBy(label).max("ind").toDF(newSchema: _*)
      .groupBy().min(newSchema(1)).first().getDouble(0) + 1
    val NumSupervisedItems = Array[Int](
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
//    val NumSupervisedItems = Array[Int](
//      100, 400, 1600, 10000, 50000, 100000) //
    val acc = new Array[Double](NUM_REPLICATIONS.toInt)
    val accSup = new Array[Double](NUM_REPLICATIONS.toInt)
    for(numSupervisedItems <- NumSupervisedItems){
      println(s"#SUPERVISED ITEMS=$numSupervisedItems")

      for(trial <- 1 to NUM_REPLICATIONS) {
        println(s"TRIAL=$trial")
        //splitting dataset
//        val splittingWeight = Array[Double](numSupervisedItems, splittingBase-numSupervisedItems)
//        val splittedDataset = corpusTrain.randomSplit(splittingWeight)
//        val labeledTrain = splittedDataset(0)
//        val unlabeledTrain = splittedDataset(1).drop(label).drop(weight)
//        if(trial == 1) {  //numSupervisedItems == 4 &&
//          println(labeledTrain.count())
//          println(labeledTrain.where($"label" === 0).count())
//        }
        //for 20 News_Groups
        val splittedDataset: Array[DataFrame] = splitDataset(minNumTopic, numSupervisedItems, corpusTrain, sqlContext)
        val labeledTrain = splittedDataset(0)
        val unlabeledTrain = splittedDataset(1)

        //prediction using EM & NaiveBayes
        val model = EMCore.emTrainNaiveBayes(labeledTrain, unlabeledTrain, numLabel, sc, smoothing, minImprovement
          , corpusTest, numofTestItems) //
        val prediction = model.transform(corpusTest)

        //evaluate
        acc(trial-1) = EMCore.AccuracyCal(prediction.select(label, "prediction"), sc, numofTestItems)

        //prediction using supervised classification
        val supervisedModel = naiveBayes.fit(labeledTrain.withColumn(groundTruth, lit(1.0D)))
        val supervisedPrediction = supervisedModel.transform(corpusTest)
        accSup(trial-1) = EMCore.AccuracyCal(supervisedPrediction.select(label, "prediction"), sc, numofTestItems)
        val accToShow = acc(trial - 1)
        val accSupToShow = accSup(trial - 1)
        println(s"Supervised Accuracy: $accSupToShow      EM Accuracy: $accToShow")
        println("-----------------------------------------------------------------------------------------------------")

      }
      //show accuracy
      val meanAccSup = accSup.sum/NUM_REPLICATIONS
      val meanAcc = acc.sum/NUM_REPLICATIONS
      var sdAcc = 0.0D
      var sdAccSup = 0.0D
      for(k <- 0 until NUM_REPLICATIONS.toInt){
        sdAcc += math.pow(acc(k)-meanAcc, 2.0)
        sdAccSup += math.pow(accSup(k)-meanAccSup, 2.0)
      }
      sdAcc = math.sqrt(sdAcc/NUM_REPLICATIONS)
      sdAccSup = math.sqrt(sdAccSup/NUM_REPLICATIONS)
      println(s"#Sup=$numSupervisedItems     Supervised mean(acc)=$meanAccSup  sd(Sup)=$sdAccSup    EM mean(acc)=$meanAcc  sd(EM)=$sdAcc")
      println("=====================================================================================================")
      println()
    }

  }

  def splitTwitterDataset(numSupervisedItems: Int, corpusTrain: DataFrame, sqlContext: SQLContext): Array[DataFrame] = {
    import sqlContext.implicits._
    val splittingWeight = Array[Double](numSupervisedItems, splittingBase)
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
