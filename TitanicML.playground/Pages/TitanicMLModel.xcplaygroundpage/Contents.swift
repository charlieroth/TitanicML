import CreateML
import TabularData
import Foundation

// ~~~~ Data Preparation ~~~~
// Load training data
let trainFile = Bundle.main.url(forResource: "train", withExtension: "csv")!
let testFile = Bundle.main.url(forResource: "test", withExtension: "csv")!

let trainTable = try DataFrame(contentsOfCSVFile: trainFile)
var testTable = try DataFrame(contentsOfCSVFile: testFile)
print("Training Data")
print(trainTable)

let (classifierEvaluationTable, classifierTrainingTable) = trainTable.randomSplit(by: 0.2, seed: 5)

let targetColumn = "Survived"
let featureColumns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

let classifier = try MLRandomForestClassifier(
    trainingData: classifierTrainingTable.base,
    targetColumn: targetColumn,
    featureColumns: featureColumns
)

// Classifier training accuracy as a percentage
let trainingError = classifier.trainingMetrics.classificationError
let trainingAccuracy = (1.0 - trainingError) * 100
print("Training accuracy: \(trainingAccuracy)")

// Classifier validation accuracy as a percentage
let validationError = classifier.validationMetrics.classificationError
let validationAccuracy = (1.0 - validationError) * 100
print("Validation accuracy: \(trainingAccuracy)")

let trainMetrics = classifier.evaluation(on: classifierEvaluationTable.base)
let classificationError = trainMetrics.classificationError
let classificationAccuracy = (1.0 - classificationError) * 100
print("Classification Accuracy: ", classificationAccuracy)

let predictionMetrics = try classifier.predictions(from: testTable.base)

let desktopPath = URL(filePath: "/Users/charlie/Desktop")

// Write CSV submission file to Desktop
testTable.append(column: predictionMetrics)
let submissionTable = testTable[["PassengerId", "Survived"]]
try submissionTable.writeCSV(
    to: desktopPath.appendingPathComponent("swift_random_forest_classifier_submission.csv")
)

// Save model to Desktop
let classifierMetadata = MLModelMetadata(
    author: "Charlie Roth",
    shortDescription: "Predicts whether a passenger of the Titanic, given some information about them, will survive or not",
    version: "1.0"
)


/// Save the trained classifier model to the Desktop.
try classifier.write(
    to: desktopPath.appendingPathComponent("TitanicSurvivalClassifier.mlmodel"),
    metadata: classifierMetadata
)
