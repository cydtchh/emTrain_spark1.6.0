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
  val docLengthNorm = 1.0D
  val minCertainty = 0.05D
  val maxCertainty = 0.8D
  def this(lambda: Double) = this(lambda, NaiveBayes.Multinomial)
  def this() = this(1.0, NaiveBayes.Multinomial)
  def myRun(data: RDD[customLabeledPoint]): NaiveBayesModel = {
    //customLabeledPoint: label, features, weight, groundTruth
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
    val aggregated = data.map(p => (p.label, p.features)).combineByKey[(Long, DenseVector)](
      createCombiner = (v: Vector) => {
        if (modelType == Bernoulli) {
          requireZeroOneBernoulliValues(v)
        } else {
          requireNonnegativeValues(v)
        }
        val lenMultiplier = lengthMultiplier(v)
        val tmpValue = v.copy.toDense
        BLAS.scal(lenMultiplier, tmpValue)
        (1L, tmpValue.copy.toDense)
      },
      mergeValue = (c: (Long, DenseVector), v: Vector) => {
        requireNonnegativeValues(v)
        val lenMultiplier = lengthMultiplier(v)
        BLAS.axpy(lenMultiplier, v, c._2)
        (c._1 + 1L, c._2)
      },
      mergeCombiners = (c1: (Long, DenseVector), c2:(Long, DenseVector)) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      }
    ).collect().sortBy(_._1)
    val numLabels = aggregated.length
    var numDocuments = 0L
    aggregated.foreach { case (_, (n, _)) =>
      numDocuments += n
    }
    val numFeatures = aggregated.head match { case (_, (_, v)) => v.size }
    val labels = new Array[Double](numLabels)
    val pi = new Array[Double](numLabels)
    val theta = Array.fill(numLabels)(new Array[Double](numFeatures))
    val piLogDenom = math.log(numDocuments + numLabels * lambda)
    var i = 0
    aggregated.foreach { case (label, (n, sumTermFreqs)) =>
      labels(i) = label
      pi(i) = math.log(n + lambda) - piLogDenom
      val thetaLogDenom = modelType match {
        case Multinomial => math.log(sumTermFreqs.values.sum + numFeatures * lambda)
        case Bernoulli => math.log(n + 2.0 * lambda)
        case _ =>
          // This should never happen.
          throw new UnknownError(s"Invalid modelType: $modelType.")
      }
      var j = 0
      while (j < numFeatures) {
        theta(i)(j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
        j += 1
      }
      i += 1
    }
    new NaiveBayesModel(labels, pi, theta, modelType)
  }
  def lengthMultiplier(v: Vector): Double ={
    val docLength = v.toArray.sum
    if(docLength <= 0) 1.0
    else docLengthNorm/docLength
  }
  def certaintyCal(v: Vector, minCertainty: Double, maxCertainty: Double): Double = {
    val sortedArr = v.toArray.sortBy(x => x > x)
    val diff = sortedArr(0) - sortedArr(1)
    val domain = maxCertainty - minCertainty
    var certainty = 0.0D
    if(diff > minCertainty){
      if(diff > maxCertainty) certainty = 1.0D
      else certainty = (diff - minCertainty) / domain
    }
    certainty //* certainty
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
//line 64
//line 95
