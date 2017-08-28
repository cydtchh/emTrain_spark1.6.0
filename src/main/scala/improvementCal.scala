import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.sql.DataFrame

/**
  * Created by cai on 08.08.17.
  */
object improvementCal {
  val PI = 3.1415926D
  val LANCZOS_COEFFS:Array[Double] = Array[Double](0.9999999999998099D, 676.5203681218851D, -1259.1392167224028D, 771.3234287776531D, -176.6150291621406D, 12.507343278686905D, -0.13857109526572012D, 9.984369578019572E-6D, 1.5056327351493116E-7D)

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
  /*public double log2CaseProb(CharSequence input) {
    JointClassification c = this.classify(input);
    double maxJointLog2P = -1.0D / 0.0;

    for(int rank = 0; rank < c.size(); ++rank) {
      double jointLog2P = c.jointLog2Probability(rank);
      if(jointLog2P > maxJointLog2P) {
        maxJointLog2P = jointLog2P;
      }
    }

    double sum = 0.0D;

    for(int rank = 0; rank < c.size(); ++rank) {
      sum += java.lang.Math.pow(2.0D, c.jointLog2Probability(rank) - maxJointLog2P);
    }

    return maxJointLog2P + Math.log2(sum);
  }*/

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
    modelLogProb/math.log(2)
  }
//  def modelLogCal(model: NaiveBayesModel, categoryPrior: Double, tokenInCatPrior: Double): Double = {
//    val catProbs = model.pi.toArray
//    var sum = dirichletLog2Prob(categoryPrior, catProbs)
//    for (catIndex <- catProbs.indices){
//      val wordProbs = new Array[Double](model.theta.numCols)
//      for(tokenIndex <- wordProbs.indices){
//        wordProbs(tokenIndex) = model.theta(catIndex, tokenIndex)
//      }
//      sum += dirichletLog2Prob(tokenInCatPrior, wordProbs)
//    }
//    sum
//  }

  def dirichletLog2Prob(alpha: Double, xs: Array[Double]): Double ={
    verifyAlpha(alpha)
    verifyDistro(xs)
    val k = xs.length
    var result = log2Gamma(k * alpha) - k * log2Gamma(alpha)
    val alphaMinus1 = alpha - 1.0D
    for(i <- xs.indices){
      result += alphaMinus1 * xs(i) / math.log(2)
    }
    result
  }

  private def verifyAlpha(alpha: Double){
    if (alpha.isNaN || alpha.isInfinite || alpha <= 0.0D) throw new SparkException(s"Concentration parameter must be positive and finite. Found alpha=$alpha.")
  }

  private def verifyDistro(xs: Array[Double]){
    for (i <- xs.indices){
      val tmpXs = math.exp(xs(i))
      if(tmpXs < 0.0D || tmpXs > 1.0D || tmpXs.isNaN || tmpXs.isInfinite) {
        throw new SparkException(s"All xs must be between 0.0 and 1.0 inclusive. Found xs[$i]=$tmpXs")
      }
    }
  }

  private def log2Gamma(e:Double): Double ={
    if(e < 0.5D){
      (math.log(PI) - math.log(Math.sin(PI * e)) - math.log(1.0D - e))/math.log(2)
    }
    else{
      var sum = 0.0D
      var z = e
      while(z > 1.5D){
        sum += math.log(z - 1.0D)
        z = z - 1.0
      }
      sum + math.log(lanczosGamma(z)) / math.log(2)
    }
  }

  private def lanczosGamma(z: Double): Double ={
    val zMinus1 = z - 1.0D
    var x = LANCZOS_COEFFS(0)
    for(i <- 1 until(LANCZOS_COEFFS.length-2)){
      x += LANCZOS_COEFFS(i) / (zMinus1 + i.toDouble)
    }
    val t = zMinus1 + (LANCZOS_COEFFS.length - 2).toDouble + 0.5D
    math.sqrt(2*PI) * math.pow(t, zMinus1 + 0.5D) * math.exp(-t) * x
  }


}

