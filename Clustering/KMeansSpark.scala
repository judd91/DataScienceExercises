import java.io.{BufferedWriter, File, FileWriter, PrintWriter}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

object KMeansScala {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("KMeansScala").setMaster("local"))

    // Load data
    val data = sc.textFile("mnist_test.csv")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    val numClusters = 10
    val numIterations = 40
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    val centers = clusters.clusterCenters
    printf("Cluster Centers: ")

    //export to a CSV
    val clustersStr = for (e <- clusters.clusterCenters) yield e.toString.dropRight(1).drop(1)

    val csvwriter = new PrintWriter(new File("output.csv"))
    for (x <- clustersStr) yield csvwriter.write(x + '\n')
    csvwriter.close

//    val file = new File("out.csv")
//    val bw = new BufferedWriter(new FileWriter(file))
//    for (x <- clustersStr) yield bw.write(x + '\n')
//    bw.close()

  }
}
