import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{NaiveBayesModel, customNaiveBayes}
//import org.apache.spark.mllib.linalg.VectorUDT
//import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions.lit

import scala.util.control.Breaks.{break, breakable}

/**
  * Created by cai on 21.08.17.
  */
class emCore{
//  val smoothing = 0.001D
//  val minImprovement = 0.0001
  private var maxEpoch: Int = 20
  private var feature = "features"
  private var weight = "weight"
  private var label = "label"
  private var groundTruth = "groundTruth"
  private var unlabeledWeight: Double = 1.0

  def setMaxEpoch(maxEpoch: Int): emCore ={
    this.maxEpoch = maxEpoch
    this
  }

  def setUnlabeledWeight(unlabeledWeight: Double): emCore = {
    this.unlabeledWeight = unlabeledWeight
    this
  }

  def emTrainNaiveBayes(labeledTrain: DataFrame,
                        unlabeledTrain: DataFrame,
                        numCate: Int,
                        sc: SparkContext,
                        smoothing: Double,
                        minImprovement: Double, testData: DataFrame, numOfTest: Double
                       ) : NaiveBayesModel = {   //
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label)
      .setFeaturesCol(feature)
      .setWeight(weight)
      .setGroundTruth(groundTruth)

    val gLabeledTrain = labeledTrain.withColumn(groundTruth, lit(1.0: Double))  //Schema: features, weight, label, groundTruth
    var lastModel = naiveBayes.fit(gLabeledTrain)        //features, label ==> prediction
    var lastLogProb = 1.0D / 0.0

    val numOfTrain = gLabeledTrain.count()
    //  "==========================EM start=========================="
    breakable {
      for (epoch <- 1 to maxEpoch) {
        val result = lastModel.transform(unlabeledTrain)
//        result.show()
        val testPre = lastModel.transform(testData)
        val trainPre = lastModel.transform(gLabeledTrain)
        val accEM = AccuracyCal(testPre.select(label, "prediction"), sc, numOfTest)
        val accTrain = AccuracyCal(trainPre.select(label, "prediction"), sc, numOfTrain)
        println(s"Accuracy of epoch $epoch in test set:  $accEM")
        println(s"Accuracy of epoch $epoch in training set:  $accTrain")

        println()

        val prediction = result
          .withColumnRenamed(naiveBayes.getProbabilityCol, weight)
          .withColumnRenamed(naiveBayes.getPredictionCol, label)
          .drop(naiveBayes.getRawPredictionCol)
          .withColumn(groundTruth, lit(unlabeledWeight))

        val combinedTrainingSet = prediction.unionAll(gLabeledTrain)
        val model = naiveBayes.fit(combinedTrainingSet)
        //training finished
        val numOne = combinedTrainingSet.where($"label" === 1).count()
        val numZero = combinedTrainingSet.where($"label" === 0).count()
        val ratio = numZero.toDouble / (numZero+numOne)
        println($"number of instances with positive label: $numZero")
        println($"number of instances with negative label: $numOne")
        println($"ratio of positive in all: $ratio")

        //calculate log probability
        var modelLogProb = 0.0D
        var dataLogProb = 0.0D
        modelLogProb = improvementCal.modelLogCal(model)
        dataLogProb = improvementCal.dataLogCal(combinedTrainingSet.select(feature, label), model, sc)
        val logProb = modelLogProb + dataLogProb
        //calculate the improvement
        val relativeDiff = relativeDif(logProb, lastLogProb)
        println(s"epoch = $epoch     modelLogProb = $modelLogProb     dataLogProb = $dataLogProb" +
          s"     logProb: $logProb     improvement: $relativeDiff")
//        println()
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



  def relativeDif(x: Double, y: Double): Double = {
    val absDif = math.abs(x - y)
    val absSum = math.abs(x) + math.abs(y)
    (2.0*absDif) / absSum
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

  def setFeatureCol(feature: String): emCore ={
    this.feature = feature
    this
  }
  def setWeightCol(weight: String): emCore ={
    this.weight = weight
    this
  }
  def setLabelCol(label: String): emCore ={
    this.label = label
    this
  }
  def setGroundTruthCol(groundTruth: String): emCore ={
    this.groundTruth = groundTruth
    this
  }

}