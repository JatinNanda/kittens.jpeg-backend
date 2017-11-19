import numpy as np
import math

def regression_to_classification(year, num_classes):
    data = np.genfromtxt(year + '-dataset.csv', delimiter=',', names=True)
    data_sorted = sorted(data, key=lambda tup: tup[196])

    bins = {}
    # num_classes = 10
    for i in xrange(1, num_classes+1):
        lower = int((1.0/num_classes)*(i-1)*len(data_sorted))
        upper = int((1.0/num_classes)*(i)*len(data_sorted))
        print lower, ',', upper
        bins[i] = data_sorted[lower:upper]

    with open('magic_dataset_' + year + '.csv', 'wb') as out:
        out.write(','.join([str(i) for i in xrange(197)]) + '\n')
        for i in bins:
            examples = bins[i]
            for row in examples:
                new_row = [str(j) for j in row][:-1]
                out.write(','.join(new_row + [str(i)]) + '\n')
