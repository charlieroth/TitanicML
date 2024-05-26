//: [Previous](@previous)

import Foundation
import TabularData

let trainFile = Bundle.main.url(forResource: "train", withExtension: "csv")!

let trainTable = try DataFrame(contentsOfCSVFile: trainFile)
print(trainTable)


//: [Next](@next)
