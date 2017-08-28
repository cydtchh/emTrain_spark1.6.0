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

  override def train(dataset: DataFrame): NaiveBayesModel = {
    val oldDataSet: RDD[customLabeledPoint] = myExtractLabeledPoints(dataset)
    val oldModel = oldNaiveBayes.train(oldDataSet, $(smoothing), $(modelType))
    modifyOld(oldModel, this)
  }

  def myExtractLabeledPoints(dataset: DataFrame): RDD[customLabeledPoint] = {
    dataset.select($(labelCol), $(featuresCol), "weight")
      .map { case Row(label: Double, features: Vector, weight: Vector) => customLabeledPoint(label, features, weight) }
  }

  def modifyOld(oldModel: OldNaiveBayesModel, parent: NaiveBayes): NaiveBayesModel ={
    val uid = if(parent != null) parent.uid else Identifiable.randomUID("nb")
    val labels = Vectors.dense(oldModel.labels)
    val pi = Vectors.dense(oldModel.pi)
    val theta = new DenseMatrix(oldModel.labels.length, oldModel.theta(0).length,
      oldModel.theta.flatten, true)
    new NaiveBayesModel(uid, pi, theta)
  }

}

