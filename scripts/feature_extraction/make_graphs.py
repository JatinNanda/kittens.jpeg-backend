import matplotlib.pyplot as plt


"""
# correlations of all features
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance Score")
plt.title("Feature Importance as Reported by Random Forest")
plt.bar(np.arange(len(clf.feature_importances_)), clf.feature_importances_, 5) #, width, color='r', yerr=men_std)
plt.show()
"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(testing_labels, predicted)
four_class_title = 'Four Class Confusion Matrix'
two_class_title = 'Two Class Confusion Matrix'
four_classes = ['Not Popular', 'Less Popular', 'Popular', 'Most Popular']
two_classes = ['Not Popular', 'Popular']
plot_confusion_matrix(cnf_matrix, classes=four_classes, title=four_class_title, normalize=True)
plt.show()
