package edu.uta.cse6331

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import scala.io.Source
import scala.util.parsing.json.JSON
import scala.collection.immutable.Map
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import java.util.Properties

object testing {
  def main(args: Array[ String ]) {
     val conf = new SparkConf().setAppName("tweets3")
     conf.setMaster("local[2]")
     val sc = new SparkContext(conf)
     val sqlc = new org.apache.spark.sql.SQLContext(sc)
     val spark = SparkSession
      .builder()
      .appName("Twitter Analysis")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
      
      val dict = sc.textFile(args(1)).map( line => { val a = line.split("\t")
                                                (a(0).toString,a(1).toInt) } )
      val positiveDict = sc.textFile(args(2)).map( x => (x,1) )
      val negativeDict = sc.textFile(args(3)).map( x => (x,-1) )
    
      val target = sc.textFile(args(0)).collect()
      
      //Sentiment Analysis using AFINN lib
      val sentiAFINN = target.map(x => {
                            var words = x.toLowerCase().split("\\W")
                            val l = words.length
                            for (i <- 0 to l-2) {
                              val ex = words(i)+" "+words(i+1)
                              if (dict.lookup(ex).length > 0) {
                                words.drop(i)
                                words.drop(i+1)
                                words :+ ex
                              }
                            }
                            val wSenti = words.map(w => {
                                         var value = 0
                                         if (dict.lookup(w).length > 0) {
                                           value = dict.lookup(w)(0)
                                         }
                                         value
                                         }) 
                            val result = wSenti.sum
                            (result,x)            
                            })
      
      //Sentiment Analysis using Bing liu opinion lexicon
      val sentiLexicon = target.map(x => {
                            val words = x.toLowerCase().split(" ")
                            val wSenti = words.map(w => {
                                         var value = 0
                                         if (positiveDict.lookup(w).length > 0) {
                                           value = positiveDict.lookup(w)(0)
                                         }
                                         if (negativeDict.lookup(w).length > 0) {
                                           value = negativeDict.lookup(w)(0)
                                         }
                                         value
                                         })
                            val result = wSenti.sum
                            (result,x)            
                            })
     
     //Sentiment Analysis using Stanford NLP
     val setting = {
       val p = new Properties()
       p.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment")
       p
     }
     
     def sentimentValue(tweet: String): String = {
       val pipe = new StanfordCoreNLP(setting)
       val annotation = pipe.process(tweet)
       var value: ListBuffer[Double] = ListBuffer()
       var size: ListBuffer[Int] = ListBuffer()
       
       var l = 0
       var main = 0
       
       for (t <- annotation.get(classOf[CoreAnnotations.SentencesAnnotation])) {
        val tree = t.get(classOf[SentimentCoreAnnotations.AnnotatedTree])
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
     
     val senti = target.map(x => {
                            val result = sentimentValue(x)
                            (result,x)            
                            })
     
     println("Sentiment Analysis using AFINN lib")
     sentiAFINN.foreach(println)
     println("Sentiment Analysis using Bing liu opinion lexicon")
     sentiLexicon.foreach(println)
     println("Sentiment Analysis using Stanford NLP")
     senti.foreach(println)    
  }  
}