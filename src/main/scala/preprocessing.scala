/**
  * Created by cai on 12.07.17.
  */
/**
  * Created by cai on 04.07.17.
  */
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object preprocessing{
  //  val labeledNewsgroup = newsGroups.withColumn("label", newsGroups("topic").like("sci%").cast("double")).cache()
  //  val Array(training, test) = labeledNewsgroup.randomSplit(Array(0.9, 0.1), seed = 12345)
  //  val training = preprocess(newsGroups)

  def main(args: Array[String]): Unit ={
    val sc = new SparkContext(new SparkConf().setAppName("preprocessing"))
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val flatten = udf((xs: Seq[Seq[String]]) => xs.flatten) //.distinct
    val regexTokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("words").setPattern("[\\W+_0-9]")
      .setMinTokenLength(2)
    //val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("keywords")
    //val hashingTF = new HashingTF().setNumFeatures(5000).setInputCol("keywords").setOutputCol("features")
    //val preprocess = new Pipeline()
    //there are some files share the same ID

    val df = regexTokenizer.transform(
      sqlContext.read.text("/home/cai/DM/TWENTY_NEWSGROUPS/20news-bydate-train/*")
        .withColumn("filename",input_file_name())
    ).drop('value)
    //.select(input_file_name,$"value")
    val testdf = regexTokenizer.transform(
      sqlContext.read.text("/home/cai/DM/TWENTY_NEWSGROUPS/20news-bydate-test/*")
        .withColumn("filename", input_file_name())
    ).drop('value)


    val newsGroups = df.groupBy('filename).agg(flatten(collect_list('words)).as('words))
      .withColumn("tmp", split($"filename", "/")).select(
      $"tmp".getItem(9).as("ID"),
      $"tmp".getItem(8).as("topic"),
      $"words".as("words")
    )   //text saves whole text of file
    val testnewsGroups = testdf.groupBy('filename).agg(flatten(collect_list('words)).as('words))
      .withColumn("tmp", split($"filename", "/")).select(
      $"tmp".getItem(9).as("ID"),
      $"tmp".getItem(8).as("topic"),
      $"words".as("words")
    )


    //    println(newsGroups.first())
    //    System.in.read()
    newsGroups.write.parquet("/home/cai/DM/output-train")
    testnewsGroups.write.parquet("/home/cai/DM/output-test")
    //    newsGroups.filter(newsGroups("topic").like("sci%")).show()
    System.in.read()
  }


}
