import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel, customNaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import testNewNB.{feature, label, weightStructure}
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
  val NUM_REPLICATIONS: Int = 10
  val newSchema = Seq("label", "maxInd")
  val minImprovement = 0.0001
  val maxEpoch = 20
  val hashingSize = 40000

  val VectorType: DataType = new VectorUDT
  val feature = "features"
  val weight = "weight"
  val label = "label"
  val weightStructure = new StructType()
    .add(StructField("ind", DoubleType, nullable = false))
    .add(StructField(feature, VectorType, nullable = false))
    .add(StructField(weight, VectorType, nullable = false))
    .add(StructField(label, DoubleType, nullable = false))

  def main(args: Array[String]){
    //    val spark = SparkSession.builder().appName("naiveBayes").getOrCreate()
    //    import spark.implicits._
    val sc = new SparkContext(new SparkConf().setAppName("emNaiveBayes"))
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

//    val train = sqlContext.read.parquet("20NewsGroups/TrainingSet")
//    val test = sqlContext.read.parquet("20NewsGroups/TestSet")
//Schema: ID, topic, words, ind
    val train = sqlContext.read.parquet("/home/cai/DM/TrainingSet")
    val test = sqlContext.read.parquet("/home/cai/DM/TestSet")
    val numofTestItems = test.count().toDouble
      //.repartition(10)// label: [0.0, 19.0]

    //-----------------------------pre-processing-----------------------------
    val indexer = new StringIndexer().setInputCol("topic").setOutputCol(label)
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("keywords")
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol(feature)//.setNumFeatures(500)
      .setNumFeatures(hashingSize)
    val naiveBayes = new NaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label).setFeaturesCol(feature)
    val preproPipe = new Pipeline()
      .setStages(Array(stopWordsRemover, hashingTF, indexer))
//    val evaluator = new MulticlassClassificationEvaluator()
    val model = preproPipe.fit(train)
    val corpusTest = model.transform(test)  //Schema: ID, topic, words, ind, keywords, features, label
    val tmpTrain = model.transform(train)
    val numLabel = (tmpTrain.groupBy()
      .max(label).first().getDouble(0) + 1.0).toInt

    val weightedTrain = tmpTrain.map{
      x => {
        val weightVector = new DenseVector(new Array[Double](numLabel))
        val tmpInd = x.getAs[Double](label).toInt
        weightVector.values(tmpInd) = 1.0
        Row(x.getAs[Double]("ind"), x.getAs[Vector](feature), weightVector, x.getAs[Double](label))
      }
    }
    val corpusTrain = sqlContext.createDataFrame(weightedTrain, weightStructure)  //Schema: ind, features, weight, label

    //-----------------------------start EM-----------------------------
    val minNumTopic = corpusTrain.groupBy(label).max("ind").toDF(newSchema: _*)
      .groupBy().min(newSchema(1)).first().getDouble(0) + 1
    val NumSupervisedItems = Array[Int](
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for(numSupervisedItems <- NumSupervisedItems){
      println(s"#SUPERVISED ITEMS=$numSupervisedItems")
      var sumAccSup = 0.0D
      var sumAcc = 0.0D
      for(trial <- 1 to NUM_REPLICATIONS) {
        println(s"TRIAL=$trial")
        var divideBound = minNumTopic - numSupervisedItems
        if(divideBound > 0){
          divideBound = ((divideBound+1)*math.random).toInt
        }
        else divideBound = 0
        val labeledTrain = corpusTrain.where($"ind" >= divideBound && $"ind" < (divideBound + numSupervisedItems))
        //.repartition(10)
        val unlabeledTrain = corpusTrain.where($"ind" < divideBound || $"ind" >= (divideBound + numSupervisedItems))
          .drop(label).drop(weight)
        //.repartition(10)

        //prediction using EM & NaiveBayes
        val model = emTrainNaiveBayes(labeledTrain, unlabeledTrain, numLabel, sc, corpusTest, numofTestItems)
        val prediction = model.transform(corpusTest)

        //evaluate
        val acc = AccuracyCal(prediction.select(label, "prediction"), sc, numofTestItems)

        //prediction using supervised classification
        val supervisedModel = naiveBayes.fit(labeledTrain)
        val supervisedPrediction = supervisedModel.transform(corpusTest)
        val accSup = AccuracyCal(supervisedPrediction.select(label, "prediction"), sc, numofTestItems)
        println(s"Supervised Accuracy: $accSup      EM Accuracy: $acc")
        println("-----------------------------------------------------------------------------------------------------")
        sumAccSup += accSup
        sumAcc += acc
      }
      //show accuracy
      val meanAccSup = sumAccSup/NUM_REPLICATIONS
      val meanAcc = sumAcc/NUM_REPLICATIONS
      println(s"#Sup=$numSupervisedItems     Supervised mean(acc)=$meanAccSup     EM mean(acc)=$meanAcc")
      println("=====================================================================================================")
      println()
    }

  }

  def emTrainNaiveBayes(labeledTrain: DataFrame, unlabeledTrain: DataFrame, numCate: Int, sc: SparkContext, testData: DataFrame, numOfTest: Double) : NaiveBayesModel = {
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label)
      .setFeaturesCol(feature)
    var lastModel = naiveBayes.fit(labeledTrain)        //features, label ==> prediction
    var lastLogProb = 1.0D / 0.0

//    println("==========================EM start==========================")
    breakable {
      for (epoch <- 1 to maxEpoch) {
        val result = lastModel.transform(unlabeledTrain)
//        val pro = result.select(naiveBayes.getProbabilityCol)
        val testPre = lastModel.transform(testData)
        val accEM = AccuracyCal(testPre.select(label, "prediction"), sc, numOfTest)
        println(s"Accuracy of customNaiveBayes: $accEM")

        val prediction = result
          .withColumnRenamed(naiveBayes.getProbabilityCol, weight)
          .withColumnRenamed(naiveBayes.getPredictionCol, label)
          .drop(naiveBayes.getRawPredictionCol)

        val combinedTrainingSet = prediction.unionAll(labeledTrain)
        val model = naiveBayes.fit(combinedTrainingSet)
        //training finished

        //calculate log probability
        var modelLogProb = 0.0D
        var dataLogProb = 0.0D
        modelLogProb = modelLogCal(model)
        //        println(s"epoch = $epoch     modelLogProb = $modelLogProb")
        dataLogProb = dataLogCal(combinedTrainingSet.select(feature, label), model, sc)
        val logProb = modelLogProb + dataLogProb
        //calculate the improvement
        val relativeDiff = relativeDif(logProb, lastLogProb)
        println(s"epoch = $epoch     modelLogProb = $modelLogProb     dataLogProb = $dataLogProb" +
          s"     logProb: $logProb     improvement: $relativeDiff")
        lastModel = model                               //feature, label1, prediction
        if (relativeDiff < minImprovement) {
          println("Converged.")
          break
        }
        lastLogProb = logProb
      }
    }
    lastModel
    //return lastModel
  }

  def dataLogCal(data: DataFrame, model: NaiveBayesModel, sc: SparkContext): Double ={
    //val accumulator = sparkSession.sparkContext.doubleAccumulator("dataLog Accumulator")
    val accumulator = sc.accumulator(0.0) //double
    data.foreach(row => {
      val sparseVector = row.getAs[Vector](0)
      val category = row.getDouble(1).toInt
      sparseVector.foreachActive((ind, _) => {
        accumulator.add(model.theta(category,ind))   //calculate P(wss|cat)
      })
      accumulator.add(model.pi(category))   // log(P(wss|cat)*P(cat))
    })  // sum of all log(P(wss|cat)*P(cat))
    accumulator.value
  }

  def modelLogCal(model: NaiveBayesModel): Double = {
    var modelLogProb = 0.0D
    for(i <- 0 until model.theta.numRows){
      modelLogProb += model.pi(i) * (-1.0D)
    }
    for(i <- 0 until model.theta.numCols){    //number of tokens
      for(j <- 0 until model.theta.numRows){  //number of categories
        modelLogProb += model.theta(j,i) * (-1.0D)
      }
    }
    modelLogProb
    //return modelLogProb
  }

  def relativeDif(x: Double, y: Double): Double = {
    val absDif = math.abs(x - y)
    val absSum = math.abs(x) + math.abs(y)
    (2.0*absDif) / absSum
  }

  def log2(e:Double): Double ={
    //return math.log(e)/math.log(2)
    math.log(e)/math.log(2)
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

