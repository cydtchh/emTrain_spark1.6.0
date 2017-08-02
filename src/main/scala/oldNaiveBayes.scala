/**
  * Created by cai on 28.07.17.
  */
package org.apache.spark.mllib.classification

import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.classification.customLabeledPoint
import org.apache.spark.mllib.linalg.{BLAS, DenseVector, SparseVector, Vector}
import scala.util.control.Breaks._

class oldNaiveBayes(private var lambda: Double,private var modelType: String)
  extends NaiveBayes{
  import oldNaiveBayes.{Bernoulli, Multinomial}

  val docLengthNorm = 9.0D

  def this(lambda: Double) = this(lambda, NaiveBayes.Multinomial)

  def this() = this(1.0, NaiveBayes.Multinomial)

  def myRun(data: RDD[customLabeledPoint]): NaiveBayesModel = {
    val requireNonnegativeValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }
      if (!values.forall(_ >= 0.0)) {
        throw new SparkException(s"Naive Bayes requires nonnegative feature values but found $v.")
      }
    }

    val requireZeroOneBernoulliValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }
      if (!values.forall(v => v == 0.0 || v == 1.0)) {
        throw new SparkException(
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
      }
    }

    // Aggregates term frequencies per label.
    // TODO: Calling combineByKey and collect creates two stages, we can implement something
    // TODO: similar to reduceByKeyLocally to save one stage.
//    val numLabels = (data.max()(Ordering.by[customLabeledPoint, Double](_.label)).label + 1).toLong
    val numLabels = data.first().weight.size
    val numFeatures = data.first().features.size
    //initialization, create the array to save all frequency
    val aggregated = data.map(p => (1.0D, Array(p.features,p.weight))).combineByKey[(DenseVector,Array[DenseVector])](
      //map each single entry to all possible classes
      createCombiner = (v: Array[Vector]) => {
        if (modelType == Bernoulli) {
          requireZeroOneBernoulliValues(v(1))
        } else {
          requireNonnegativeValues(v(1))
        }
        val buffArray = new Array[DenseVector](numLabels)
        var weightCount = 0.0D
        breakable {
          for (i <- 0 until numLabels) {
            if (v(1)(i) >= 0.01D){
              buffArray(i) = v(0).copy.toDense
              val lengthNormCatMultiplier = v(1)(i) / lengthMultiplier(v(0))
              BLAS.scal(lengthNormCatMultiplier, buffArray(i))
              weightCount += v(1)(i)
              if (weightCount >= 0.99D) break
            }
          }
        }
        (v(1).copy.toDense, buffArray)
      },
      mergeValue = (c: (DenseVector, Array[DenseVector]), v: Array[Vector]) => {
        requireNonnegativeValues(v(1))
        var weightCount = 0.0D
        breakable {
          for (i <- 0 until numLabels) {
            if (v(1)(i) >= 0.01D) {
              val lengthNormCatMultiplier = v(1)(i) / lengthMultiplier(v(0))
              if (c._2(i) == null) {
                c._2(i) = v(0).copy.toDense
                BLAS.scal(lengthNormCatMultiplier, c._2(i))
              } else {
                BLAS.axpy(lengthNormCatMultiplier, v(0), c._2(i))
              }
              weightCount += v(1)(i)
              if (weightCount >= 0.99D) break
            }
          }
        }
        BLAS.axpy(1.0D, v(1), c._1)
        (c._1, c._2)
      },
      mergeCombiners = (c1: (DenseVector, Array[DenseVector]), c2: (DenseVector, Array[DenseVector])) => {
        for(i <- 0 until numLabels){
          if(c1._2(i) != null){
            if(c2._2(i) != null) BLAS.axpy(1.0, c1._2(i), c2._2(i))
            else c2._2(i) = c1._2(i)
          } else{
            if(c2._2(i) == null) c2._2(i) = new DenseVector(new Array[Double](numFeatures))
          }
        }
        BLAS.axpy(1.0, c1._1, c2._1)
        (c2._1, c2._2)
      }
    ).collect()
    //    val numLabels = aggregated.length
    //    val numFeatures = aggregated.head match { case (_, (_, v)) => v.size }
    val numDocuments = aggregated(0)._2._1.toArray.sum

    val labels = new Array[Double](numLabels)
    val pi = new Array[Double](numLabels)   // a-priori
    val theta = Array.fill(numLabels)(new Array[Double](numFeatures)) //conditional probability
//    val lambdaOfPi = lambda * 5
    val piLogDenom = math.log(numDocuments + lambda * numLabels)

    aggregated.foreach { case (_, (n, sumTermFreqs)) => //only one element contain information of all classes
      for(i <- 0 until numLabels){
        pi(i) = math.log(n(i) + lambda) - piLogDenom
        val thetaLogDenom = modelType match {
          case Multinomial => math.log(sumTermFreqs(i).values.sum + lambda * numFeatures)
          case Bernoulli => math.log(n(i) + 2.0 * lambda)
          case _ =>
            // This should never happen.
            throw new UnknownError(s"Invalid modelType: $modelType.")
        }
        var j = 0
        while (j < numFeatures) {
          theta(i)(j) = math.log(sumTermFreqs(i)(j) + lambda) - thetaLogDenom
          j += 1
        }
      }
    }

    new NaiveBayesModel(labels, pi, theta, modelType)
  }

  def lengthMultiplier(v: Vector): Double ={
    val docLength = v.toArray.sum
    if(docLength <= 0) 1.0
    else docLength
  }
}

object oldNaiveBayes {
  val Multinomial: String = "multinomial"
  val Bernoulli: String = "bernoulli"
  val supportedModelTypes = Set(Multinomial, Bernoulli)

  def train(input: RDD[customLabeledPoint]): NaiveBayesModel = {
    new oldNaiveBayes().myRun(input)
  }

  def train(input: RDD[customLabeledPoint], lambda: Double): NaiveBayesModel = {
    new oldNaiveBayes(lambda, Multinomial).myRun(input)
  }

  def train(input: RDD[customLabeledPoint], lambda: Double, modelType: String): NaiveBayesModel = {
    require(supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType: $modelType.")
    new oldNaiveBayes(lambda, modelType).myRun(input)
  }

}
