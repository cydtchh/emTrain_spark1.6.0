package org.apache.spark.ml.classification

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.beans.BeanInfo

/**
  * Created by cai on 27.07.17.
  */
@BeanInfo
case class customLabeledPoint (label: Double, features: Vector, weight: Vector, groundTruth: Double) {
  override def toString: String = {
    s"($label, $features, $weight, $groundTruth)"
  }
}