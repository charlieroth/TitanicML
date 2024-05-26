# TitanicML

A Swift Playground for training a CoreML model to predict survival of Titanic passengers based on a set of features

## Motivation

Learn how to use the [CreateML](https://developer.apple.com/documentation/createml) library to do on-device training of a ML model in Swift

## Description

One of Kaggle's recommended starting competitions is the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). I took this competition as an opportunity to learn how to use the `CreatML` framework and eventually integrate the resulting `CoreML` model into a Swift application. This application can be found at [https://github.com/charlieroth/TitanicSurvival](https://github.com/charlieroth/TitanicSurvival)

Building a simple prediction model for survival outcomes can be done with a traditional methods such as a [Random Forest](https://en.wikipedia.org/wiki/Random_forest); an ensemble learning method for classifcation. Typically this is done with Python, `scikit-learn`, and the `RandomForestClassifier` class but Apple's `CreateML` framework provides a `MLRandomForestClassifier` struct to do the same learning. Using `MLRandomForestClassifier`, I was able to train a model to predict the correct survival outcome 77% of the time (based on submission results to Kaggle's platform)