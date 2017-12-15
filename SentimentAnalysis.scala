package edu.uta.cse6331

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import scala.io.Source
import scala.util.parsing.json.JSON
import scala.collection.immutable.Map
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import java.util.Properties
import org.apache.spark.sql.functions.explode

object SentimentAnalysis {
   def main(args: Array[ String ]) {
    val conf = new SparkConf().setAppName("tweets")
    val sc = new SparkContext(conf)
    val sqlc = new org.apache.spark.sql.SQLContext(sc)
    
    val setting = {
       val p = new Properties()
       p.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment")
       p
     }
    
    def countryTweets(line: String, req: Array[String]): Boolean = {
     val words = line.split(",")
     var b = false
     var parsedWords = new ArrayBuffer[String]()
     for ( w <- words) {
       parsedWords +=  w.trim()
     }
     parsedWords.exists(req.contains)
    }
    
    def wordTweets(line: String, grp: Array[String]): Boolean = {
     val words = line.toLowerCase().split(" ")
     words.exists(grp.contains)
    }
    
    def sentimentValue(tweet: String): String = {
       val pipe = new StanfordCoreNLP(setting)
       val annotation = pipe.process(tweet)
       var value: ListBuffer[Double] = ListBuffer()
       var size: ListBuffer[Int] = ListBuffer()
       
       var l = 0
       var main = 0
       
       for (t <- annotation.get(classOf[CoreAnnotations.SentencesAnnotation])) {
        val tree = t.get(classOf[SentimentCoreAnnotations.SentimentAnnotatedTree])
        val res = RNNCoreAnnotations.getPredictedClass(tree)
        val message = t.toString

        if (message.length() > l) {
          main = res
          l = message.length()
        }

        value += res.toDouble
        size += message.length
       }
       val weightedTotal = (value, size).zipped.map((x, y) => x * y)
       var finalValue = weightedTotal.sum / (size.fold(0)(_ + _))
       
       if (finalValue <= 0.0)
        "NOT_UNDERSTOOD"
       else if (finalValue < 1.6)
        "NEGATIVE"
       else if (finalValue <= 2.0)
        "NEUTRAL"
       else if (finalValue < 5.0)
        "POSITIVE"
       else "NOT_UNDERSTOOD"       
     } 
      
    val data = sqlc.jsonFile(args(0))
    //println(data.collect().length)
    data.registerTempTable("tweets")
    //data.printSchema()
    val info = sqlc.sql("SELECT text,user FROM tweets")
    val allTweets = info.rdd.map(x => {
                              (x(0).toString, x(1).toString())
                               })
                               
    val reqWordsI = Array("India", "New Delhi", "Pune", "Mumbai", "IN", "Gurgaon ncr delhi", "MADHYA PRADE")
    val in = allTweets.filter(x => countryTweets(x._2,reqWordsI)).collect()
    
    val reqWordsL = Array("London", "UK", "United Kingdom", "Ireland", "England", "GB", "Romford", "Leicester", "Cannock", "Scotland")
    val gb = allTweets.filter(x => countryTweets(x._2,reqWordsL)).collect()
    
    val reqWordsA = Array("New York", "Brooklyn", "United States", "US", "phl", "Philadelphia PA", "NY", "Pennsylvania", "Eastern Time (US & Canada)", "Central Time (US & Canada)")
    val us = allTweets.filter(x => countryTweets(x._2,reqWordsA)).collect()
    
    val reqWordsK = Array("Republic of korea", "Korea", "Seoul", "Shenzhen China", "WA", "Beijing", "KR")
    val kr = allTweets.filter(x => countryTweets(x._2,reqWordsK)).collect()
    
    val dataGroup1 = Array("iphone", "ios", "apple", "@apple", "#iphone")
    val dataGroup2 = Array("health", "food", "coffee", "healthy", "#food")
    val dataGroup3 = Array("gym", "cardio", "jogging")
    
    var result = new ArrayBuffer[String]()
     
    var dataGroup = new ArrayBuffer[Array[String]]()
    dataGroup += dataGroup1
    dataGroup += dataGroup2
    dataGroup += dataGroup3
     
    var countryGroup = new ArrayBuffer[Array[(String,String)]]()
    countryGroup += in
    countryGroup += gb
    countryGroup += us
    countryGroup += kr
     
    var c1 = 0
     
    for (c <- countryGroup)
     {
       c1 = c1+1
       for (g <- dataGroup)
       {
         var grp = new ArrayBuffer[String]()
         for (word <- g) {
           grp += word
         }
         val subGroup = sc.parallelize(c).filter(x => wordTweets(x._1,g))
         
         val senti = subGroup.map(x => {
                            val result = sentimentValue(x._1)
                            (result,x._1)            
                            })
         val sCounts = senti.map(x => (x._1,x._2)).countByKey().toSeq.sortBy(_._2).toMap
         if (c1 == 1) result += "India,"+grp.toList+","+sCounts
         if (c1 == 2) result += "United Kingdom,"+grp.toList+","+sCounts
         if (c1 == 3) result += "United States,"+grp.toList+","+sCounts
         if (c1 == 4) result += "Korea,"+grp.toList+","+sCounts
       }
     }
     
    result.foreach(println)
    sc.parallelize(result).coalesce(1).saveAsTextFile(args(1))
    sc.stop()
   }
}
