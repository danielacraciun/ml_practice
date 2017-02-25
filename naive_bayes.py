import csv
import random
import math

from collections import defaultdict

def load_csv(filename):
    with open(filename) as ifile:
        read = csv.reader(ifile)
        return [[float(x) for x in line] for line in read]

def split_data(dataset, ratio=0.67):
	size = int(len(dataset) * ratio);print(size)
	train = []
	copy = list(dataset)
	while len(train) < size:
		index = random.randrange(len(copy))
		train.append(copy.pop(index))
	return [train, copy]

def separate_by_class(dataset):
    separated = defaultdict(list)
    for index, item in enumerate(dataset):
        separated[item[-1]].append(item)
    return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x - avg, 2) for x in numbers])/float(len(numbers) - 1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries[:-1]

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    return {values: summarize(instances) for value, instances in separated.iteritems()}

def prob(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean, 2)/(2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def class_prob(summaries, vect):
	probabilities = defaultdict(1)
	for class_value, class_summaries in summaries.iteritems():
		for index, item in enumerate(class_summaries):
			probabilities[class_value] *= prob(vect[i], item[0], item[1])
	return probabilities

def predict(summaries, vect):
	probabilities = calculateClassProbabilities(summaries, vect)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb, bestLabel = probability, classValue
	return bestLabel

def get_predictions(summaries, testSet):
	return [predict(summaries, item) for item in testSet]

data = load_csv('pima-indians-diabetes.csv')
print(summarize(data))
