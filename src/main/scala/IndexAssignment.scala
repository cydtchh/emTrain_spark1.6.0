import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by cai on 12.07.17.
  */
object IndexAssignment {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  def main(args: Array[String]){
    val sc = new SparkContext(new SparkConf().setAppName("IndexAssignment"))
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val test = sqlContext.read.parquet("/home/cai/DM/output-test")
    val train = sqlContext.read.parquet("/home/cai/DM/output-train")

    val categorySet = train.select("topic").distinct().cache()
    val numCate = categorySet.count().toInt
    val arrayOfCat = categorySet.take(numCate)

    val indexerAssign = new StringIndexer().setInputCol("ID").setOutputCol("ind")
//    val indexer = new StringIndexer().setInputCol("topic").setOutputCol("label")

//    val step1ForTest = indexer.fit(test).transform(test)
//    val step1ForTrain = indexer.fit(train).transform(train)

    var TestSet = indexerAssign.fit(test.where($"topic" === arrayOfCat(0).getAs[String](0)))
      .transform(test.where($"topic" === arrayOfCat(0).getAs[String](0)))
    var TrainingSet = indexerAssign.fit(train.where($"topic" === arrayOfCat(0).getAs[String](0)))
      .transform(train.where($"topic" === arrayOfCat(0).getAs[String](0)))
    println(TrainingSet.first().get(1))
    TrainingSet.groupBy().max(indexerAssign.getOutputCol).show()
    println(TrainingSet.count())
    println()
    var flag = false
    for(category <- arrayOfCat){
      if(flag){
        val tmpTest = indexerAssign
          .fit(test.where($"topic"===category.getAs[String](0)))
          .transform(test.where($"topic"===category.getAs[String](0)))
        TestSet = TestSet.unionAll(tmpTest)
        val tmpTrain = indexerAssign
          .fit(train.where($"topic"===category.getAs[String](0)))
          .transform(train.where($"topic"===category.getAs[String](0)))
        TrainingSet = TrainingSet.unionAll(tmpTrain)
        println(tmpTrain.first().get(1))
        tmpTrain.groupBy().max(indexerAssign.getOutputCol).show()
        println(tmpTrain.count())
        println
      }
      else flag = true
    }
    println(TestSet.count())
    println(TrainingSet.count())

    TrainingSet.repartition(200).write.parquet("/home/cai/DM/TrainingSet") //.repartition(10).
    TestSet.repartition(10).write.parquet("/home/cai/DM/TestSet")
  }

}
