# Exported notebook code; may contain IPython commands.
# %%
from math import sqrt
from math import exp
from math import pi
from sklearn.datasets import load_iris 

# %%
# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# %%
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

# %%
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(setdata):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*setdata)]
    del(summaries[-1])
    return summaries

# %%
# Split dataset by class then calculate statistics for each row
def summarize_by_class(separated):
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# %%
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# %%
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# %%
# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# %%
# Make a prediction with Naive Bayes on Iris Dataset
setiris = load_iris()
setdata = dict()
for (i, item) in enumerate(list(setiris.target_names), start = 0):
    print('[%s] => %d' % (item, i))
    setdata[i] = setiris.data[setiris.target == i]

# %%
# fit model
model = summarize_by_class(setdata)
# define a new record
row = [5.7, 2.9, 4.2, 1.3]
# predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))

# %%
# Calculate accuracy of our custom Naive Bayes on the entire Iris dataset
correct = 0
total = len(setiris.data)

for i in range(total):
    row = setiris.data[i]
    actual = setiris.target[i]
    predicted = predict(model, row)
    if predicted == actual:
        correct += 1

accuracy_custom = correct / total * 100
print(f'Custom Naive Bayes Accuracy: {accuracy_custom:.2f}%')
print(f'Correct: {correct} / {total}')

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data
X = setiris.data
y = setiris.target

# Create and fit the GaussianNB model
gnb = GaussianNB()
gnb.fit(X, y)

# Predict on the same dataset
y_pred = gnb.predict(X)

# Calculate accuracy
accuracy_sklearn = accuracy_score(y, y_pred) * 100
print(f'Sklearn GaussianNB Accuracy: {accuracy_sklearn:.2f}%')

# %%
print(f'Custom Naive Bayes Accuracy:  {accuracy_custom:.2f}%')
print(f'Sklearn GaussianNB Accuracy:  {accuracy_sklearn:.2f}%')
