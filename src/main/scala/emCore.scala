import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{NaiveBayesModel, customNaiveBayes}
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.util.control.Breaks.{break, breakable}

/**
  * Created by cai on 21.08.17.
  */
object emCore{
//  val smoothing = 0.001D
//  val minImprovement = 0.0001
  val maxEpoch = 20
  val feature = "features"
  val weight = "weight"
  val label = "label"

  def emTrainNaiveBayes(labeledTrain: DataFrame,
                        unlabeledTrain: DataFrame,
                        numCate: Int,
                        sc: SparkContext,
                        smoothing: Double,
                        minImprovement: Double, testData: DataFrame, numOfTest: Double
                       ) : NaiveBayesModel = {   //
    val naiveBayes = new customNaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol(label)
      .setFeaturesCol(feature)
    var lastModel = naiveBayes.fit(labeledTrain)        //features, label ==> prediction
    var lastLogProb = 1.0D / 0.0

    //  "==========================EM start=========================="
    breakable {
      for (epoch <- 1 to maxEpoch) {
        val result = lastModel.transform(unlabeledTrain)
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
        modelLogProb = improvementCal.modelLogCal(model)
        dataLogProb = improvementCal.dataLogCal(combinedTrainingSet.select(feature, label), model, sc)
        val logProb = modelLogProb + dataLogProb
        //calculate the improvement
        val relativeDiff = relativeDif(logProb, lastLogProb)
        println(s"epoch = $epoch     modelLogProb = $modelLogProb     dataLogProb = $dataLogProb" +
          s"     logProb: $logProb     improvement: $relativeDiff")
        println()
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

}
