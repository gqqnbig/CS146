"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None #prob of y=0
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        freqDict = Counter(y).most_common(2)
        length = freqDict[0][1] + freqDict[1][1]
        self.probabilities_ = (freqDict[0][1] if freqDict[0][0] == 0 else freqDict[1][1])/length
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        #np.random.seed(seed)
        y = np.random.choice(2, X.shape[0], p=[self.probabilities_, 1-self.probabilities_])
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_err = 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        test_err = 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        train_error += train_err
        test_error += test_err
    train_error /= ntrials
    test_error /= ntrials

    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    randClf = RandomClassifier()
    randClf.fit(X, y)
    randy_pred = randClf.predict(X)
    randTrain_error = 1 - metrics.accuracy_score(y, randy_pred, normalize=True)
    print('\t-- training error: %.3f' % randTrain_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    dTree = DecisionTreeClassifier(criterion='entropy')
    dTree.fit(X, y)
    dty_pred = dTree.predict(X)
    dtTrain_error = 1 - metrics.accuracy_score(y, dty_pred, normalize=True)
    print('\t-- training error: %.3f' % dtTrain_error)
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    # save the classifier -- requires GraphViz and pydot
    
    # import io, pydotplus
    # from sklearn import tree
    # dot_data = io.StringIO()
    # tree.export_graphviz(dTree, out_file=dot_data,
    #                      feature_names=Xnames)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("dtree.pdf") 




    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    knn3 = KNeighborsClassifier(n_neighbors=3, p=2)
    knn3.fit(X, y)
    knn3y_pred = knn3.predict(X)
    knn3Train_error = 1 - metrics.accuracy_score(y, knn3y_pred, normalize=True)
    print('\t-- training error: %.3f when k = 3' % knn3Train_error)
    knn5 = KNeighborsClassifier(n_neighbors=5, p=2)
    knn5.fit(X, y)
    knn5y_pred = knn5.predict(X)
    knn5Train_error = 1 - metrics.accuracy_score(y, knn5y_pred, normalize=True)
    print('\t-- training error: %.3f when k = 5' % knn5Train_error)
    knn7 = KNeighborsClassifier(n_neighbors=7, p=2)
    knn7.fit(X, y)
    knn7y_pred = knn7.predict(X)
    knn7Train_error = 1 - metrics.accuracy_score(y, knn7y_pred, normalize=True)
    print('\t-- training error: %.3f when k = 7' % knn7Train_error)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    majorClf = MajorityVoteClassifier()
    randomClf = RandomClassifier()
    dtClf = DecisionTreeClassifier(criterion='entropy')
    knn5Clf = KNeighborsClassifier(n_neighbors=5, p=2)
    for classifier in [majorClf, randomClf, dtClf, knn5Clf]:
        train_error, test_error = error(classifier, X, y)
        print("Average results of {}:\n\t-- training error: {:.3f}, test error: {:.3f}".format(classifier.__class__.__name__, train_error, test_error))
    
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    kLst = []
    errLst = []
    for k in range(1, 50, 2):
        errors = cross_val_score(KNeighborsClassifier(n_neighbors=k, p=2), X, y, cv=10)
        kLst.append(k)
        errLst.append(1-np.mean(errors))
    plt.plot(kLst, errLst)
    plt.xlabel('k, # of neighbors')
    plt.ylabel('error rate')
    plt.savefig("4fGraph.pdf")

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    kLst = []
    trainErrs = []
    testErrs = []
    for k in range(1, 21):
        train_error, test_error = error(DecisionTreeClassifier(criterion='entropy', max_depth=k), X, y)
        kLst.append(k)
        trainErrs.append(train_error)
        testErrs.append(test_error)
    plt.clf()
    red_patch = mpl.patches.Patch(color='red', label='training error')
    green_patch = mpl.patches.Patch(color='green', label='test error')
    plt.plot(kLst, testErrs, 'go-', kLst, trainErrs, 'r^-')
    plt.xlabel('decision tree depth limit')
    plt.ylabel('error rate')
    plt.legend(handles=[red_patch, green_patch])
    plt.savefig('4gGraph.pdf')


    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    index = []
    dtTrainErrs = []
    dtTestErrs = []
    knnTrainErrs = []
    knnTestErrs = []
    for i in range(1, 11):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        index.append(i*0.1)
        for k in range(100):
            X_train2, y_train2 = X_train, y_train
            if i != 10:
                X_train2, _, y_train2, _ = train_test_split(X_train, y_train, test_size=(1-i*0.1), random_state=k)
            dtClf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            knnClf = KNeighborsClassifier(n_neighbors=7, p=2)

            dtClf.fit(X_train2, y_train2)
            dt_y_train_pred = dtClf.predict(X_train2)
            train_error1 = 1 - metrics.accuracy_score(y_train2, dt_y_train_pred, normalize=True)
            sum1 += train_error1
            
            dt_y_test_pred = dtClf.predict(X_test)
            test_error1 = 1 - metrics.accuracy_score(y_test, dt_y_test_pred, normalize=True)
            sum2 += test_error1


            knnClf.fit(X_train2, y_train2)
            knn_y_train_pred = knnClf.predict(X_train2)
            train_error2 = 1 - metrics.accuracy_score(y_train2, knn_y_train_pred, normalize=True)
            sum3 += train_error2
            
            knn_y_test_pred = knnClf.predict(X_test)
            test_error2 = 1 - metrics.accuracy_score(y_test, knn_y_test_pred, normalize=True)
            sum4 += test_error2
        dtTrainErrs.append(sum1/100)
        dtTestErrs.append(sum2/100)
        knnTrainErrs.append(sum3/100)
        knnTestErrs.append(sum4/100)

    plt.clf()
    plt.plot(index, dtTrainErrs, 'ro-', \
            index, dtTestErrs, 'go-', \
            index, knnTrainErrs, 'r^-', \
            index, knnTestErrs, 'g^-')
    red_circle = mpl.lines.Line2D([], [], color='r', marker='o', label='DT training error')
    green_circle = mpl.lines.Line2D([], [], color='g', marker='o', label='DT test error')
    red_tri = mpl.lines.Line2D([], [], color='r', marker='^', label='KNN training error')
    green_tri = mpl.lines.Line2D([], [], color='g', marker='^', label='KNN test error')
    plt.legend(handles=[red_circle, green_circle, red_tri, green_tri])
    plt.xlabel('portion of 90% training set')
    plt.ylabel('error rate')
    plt.savefig('4hGraph.pdf')


    
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
