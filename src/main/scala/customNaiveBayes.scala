package org.apache.spark.ml.classification

//import org.apache.spark.SparkException
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.{oldNaiveBayes, NaiveBayesModel => OldNaiveBayesModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row}


/**
  * Created by cai on 27.07.17.
  */
class customNaiveBayes extends NaiveBayes{
  val labelSmoothing = 0.5
  val wordSmoothing = 0.01
//  private var unlabeledWeight = 1.0
//  private var mySmoothing = 0.001D
  private var myLabelCol = "label"
  private var myFeaturesCol = "features"
//  private var myModelType = "multinomial"
  private var weight = "weight"
  private var groundTruth = "groundTruth"

  override def train(dataset: DataFrame): NaiveBayesModel = {
    val oldDataSet: RDD[customLabeledPoint] = myExtractLabeledPoints(dataset)
    val oldModel = oldNaiveBayes.train(oldDataSet, $(smoothing), $(modelType))
    modifyOld(oldModel, this) //transform old model to new one
  }

  def myExtractLabeledPoints(dataset: DataFrame): RDD[customLabeledPoint] = {
    dataset.select(myLabelCol, myFeaturesCol, weight, groundTruth)
      .map { case Row(label: Double, features: Vector, weight: Vector, groundTruth: Double) => customLabeledPoint(label, features, weight, groundTruth) }
  }

  def modifyOld(oldModel: OldNaiveBayesModel, parent: NaiveBayes): NaiveBayesModel ={
    val uid = if(parent != null) parent.uid else Identifiable.randomUID("nb")
    val labels = Vectors.dense(oldModel.labels)
    val pi = Vectors.dense(oldModel.pi)
    val theta = new DenseMatrix(oldModel.labels.length, oldModel.theta(0).length,
      oldModel.theta.flatten, true)
    new NaiveBayesModel(uid, pi, theta)
  }

  override def setLabelCol(value: String): customNaiveBayes = {
    this.myLabelCol = value
    this
  }

  override def setFeaturesCol(value: String): customNaiveBayes = {
    this.myFeaturesCol = value
    this
  }

  def setWeight(weight: String): customNaiveBayes = {
    this.weight = weight
    this
  }

  def setGroundTruth(groundTruth: String): customNaiveBayes = {
    this.groundTruth = groundTruth
    this
  }

//  def setUnlabeledWeight(unlabeledWeight: Double): customNaiveBayes = {
//    this.unlabeledWeight = unlabeledWeight
//    this
//  }
}

