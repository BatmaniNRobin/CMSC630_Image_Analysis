import os
import numpy as np
import re

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from decimal import Decimal
from random import randrange
from operator import eq

from main import (
    calc_histogram,
    read_yaml,
    read_image,
    convert_image_to_single_channel
    )

from part2 import (
    hist_threshold,
    balance_hist,
    erosion,
    dilation
)

def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

def deserialize_label(value, labels):
    return labels[value]

def serialize_label(label, labels):
    return labels.index(label)

# convert floats to be between 0-1
def normalize(dataset):
    minmax = dataset.copy()

    without_labels = dataset[:, :-1]
    
    # transpose dataset
    for idx, column in enumerate(without_labels.T):
        smallest = np.min(column)
        largest = np.max(column)

        rng = largest - smallest

        if rng == 0:
            continue

        minmax[:, idx] = (minmax[:, idx] - smallest) / rng

    return minmax

# evaluation metrics
# https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/
def calc_accuracy(actual, predicted):
    
    correct = 0
    
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            
    accuracy = correct / float(len(actual)) * 100.0
    
    return accuracy

def confusion_matrix(actual, predicted):
    
    classes = np.unique(actual)
    
    confmat = np.zeroes((len(classes), len(classes)))
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            confmat[i, j] = np.sum((actual == classes[i]) and (predicted == classes[j]))
            
    return confmat

def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def roc_from_scratch(probabilities, y_test, partitions=100):
    roc = np.array([])
    for i in range(partitions + 1):
        
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)

def mean_abs_error(actual, predicted):
    sum_error = 0.0
    
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    
    mase = sum_error / float(len(actual))
    
    return mase

def rmse(actual, predicted):
    sum_error = 0.0
    
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    
    mean_error = sum_error / float(len(actual))
    
    rmse = np.sqrt(mean_error)
    
    return rmse


# feature extraction 
def shrinking_and_growing(hist_threshold):
    
    # my code works backwards on binary images
    # so gotta call dilation in order to erode
    eroded = dilation(hist_threshold)
    sng = erosion(eroded)
    
    return sng

# https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/
# https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated
def entropy(img, hist):
    
    # marg calculates marginal dist, then filter out probs == 0
    marg = hist / img.size
    marg = np.array(list(filter(lambda p: p > 0, marg)))
    
    # then sum whats left
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))

    return entropy

def area(sng):
    unique, counts = np.unique(sng, return_counts=True)
    counter = dict(zip(unique, counts))

    # black pixels are the cells
    black_pixel_count = counter[0]

    return black_pixel_count

# TODO 4th feature needed
def extract_features(img, hist, sng, labels, filename):
    
    search_obj = re.search(r"(\D+)(\d+).*", filename, re.M | re.I)
    label = search_obj.group(1)

    try:
        y = serialize_label(label, labels)
    except KeyError:
        y = None
    
    # calculate the randomness of the pixel intensities
    x1 = entropy(img, hist)
    
    # calculate mean of img histogram
    x2 = np.mean(hist)
    
    # calculate area
    x3 = area(sng)
    
    return {
        "features": [x1, x2, x3, y]
    }



# knn
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def p_root(value, root):
    root_value = 1 / float(root)
    return round (Decimal(value) **
             Decimal(root_value), 3)

def distance_calc(p1, p2, dist_type):

    if(dist_type == 'manhattan'):
        dist =  sum(abs(val1-val2) for val1, val2 in zip(p1,p2))
    elif(dist_type == 'minkowski'):
        p_value = 3
        return p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(p1, p2)), p_value)
    elif(dist_type == 'mahalanobis'):
        return "nothing"
    elif(dist_type == "chebychev"):
        return 'cheby'
    else: # euclidean
        dist = np.sqrt(np.sum((p1-p2)**2))
        
    return dist

# could use np arrays instead of lists to make it faster
def get_neighbors(train, test_row, k):
    
    distances = [(train_row, distance_calc(test_row, train_row, 'euclidean')) for train_row in train]
    distances.sort(key=lambda tup:tup[1])
    
    neighbors = np.array([distances[i][0] for i in range(k)])
        
    return neighbors        

def predict(train, test_row, k):
    
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    
    return prediction

def knn(train, test, k):
    knn = np.array([predict(train, row, k) for row in test])
    
    return knn



# evaluation
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def cross_validation(dataset, n_folds):
    
    dataset_split = []
    dataset_copy = dataset.copy()
    fold_size = len(dataset) // n_folds
    
    for i in range(n_folds):
        fold = []
        
        while(len(fold) < fold_size):
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy[index])
            dataset_copy = np.delete(dataset_copy, index, axis=0)
        
        dataset_split.append(fold)
        
    cv = np.array(dataset_split)
    
    return cv

def evaluate(dataset, n_folds, k):
    folds = cross_validation(dataset, n_folds)
    acc_scores = []
    
    for idx, fold in enumerate(folds):
        train_set = np.delete(folds, idx, axis=0)
        train_set = np.concatenate(train_set, axis=0)
        test_set = []
        
        for row in fold:
            row_copy = row.copy()
            test_set.append(row_copy)
            row_copy[-1] = None
        
        test_set = np.array(test_set)
        
        predicted = knn(train_set, test_set, k)
        actual = [row[-1] for row in fold]
        accuracy = calc_accuracy(actual, predicted)
        
        acc_scores.append(accuracy)
        
    return acc_scores


def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    Path(safe_conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    
    dataset_out_file = Path(safe_conf["DATASET_OUTPUT"])
    
    features = []
    
    labels = ["cyl", "inter", "let", "mod", "para", "super", "svar"]
    
    for i in range(len(files)):
        filenames[i] = os.path.basename(files[i])
        
        if (".BMP" in filenames[i]):
            filenames[i] = os.path.splitext(filenames[i])[0]
        
        
        color_image = read_image(files[i])
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        
        img_hist = calc_histogram(img)
        
        histogram_threshold = hist_threshold(img, img_hist)
        save_image(histogram_threshold, filenames[i], "_hist_threshold")
        
        sng = shrinking_and_growing(histogram_threshold)
        save_image(sng, filenames[i], "_sng")
        
        img_features = extract_features(img, img_hist, sng, labels, filenames[i])
            
        for x in img_features:
            features.append(img_features[x])
    
    
    norm_dataset = normalize(np.array(features))
    np.savetxt(dataset_out_file, norm_dataset, delimiter=',')
    # dataset = np.loadtxt(dataset_out_file, delimiter=',')

    total_avg = 0

    for k in range(1, 6):
        scores = evaluate(norm_dataset, n_folds=10, k=k)
        average = sum(scores) / float(len(scores))
        
        total_avg += average
        

        print("K=", k)
        print("scores:", scores)
        print("avg: ", average)
        
    total_avg /= int(5)
    print("\ntotal avg: ", total_avg)
        
        
    
if __name__ == "__main__":
    main()