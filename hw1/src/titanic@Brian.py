"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter
import random
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
        self.probabilities_ = None
    
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
        c = Counter(y)

        # This will get the total number of items (the values combined)
        total_elements = float(sum(c.values()))
        d = dict(c)
        for key in d:
            d[key] = (d[key] / total_elements)

        self.probabilities_ = d
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
        
        # Grab shape of X
        n, _ = X.shape

        label = np.array(list(self.probabilities_.keys()))
        prob = np.array(list(self.probabilities_.values()))
        y = np.random.choice(label, n, p=prob)
        ### ========== TODO : END ========== ###
        
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
    l1 = []
    l2 = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)                  # fit training data using the classifier
        y_train_pred = clf.predict(X_train)        # take the classifier and run it on the training data
        train_error = 1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True)
        y_test_pred = clf.predict(X_test)        # take the classifier and run it on the training data
        test_error = 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)
        l1.append(train_error)
        l2.append(test_error)

    ### ========== TODO : END ========== ###
    
    return sum(l1)/float(len(l1)), sum(l2)/float(len(l2))


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
    #for i in range(d) :
    #    plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
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
    clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion = "entropy") # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    '''print('Classifying using k-Nearest Neighbors...')
                clf = KNeighborsClassifier(n_neighbors=3) # create MajorityVote classifier, which includes all model parameters
                clf.fit(X, y)                  # fit training data using the classifier
                y_pred = clf.predict(X)        # take the classifier and run it on the training data
                train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
                print('\t-3- training error: %.3f' % train_error)
                clf = KNeighborsClassifier(n_neighbors=5) # create MajorityVote classifier, which includes all model parameters
                clf.fit(X, y)                  # fit training data using the classifier
                y_pred = clf.predict(X)        # take the classifier and run it on the training data
                train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
                print('\t-5- training error: %.3f' % train_error)
                clf = KNeighborsClassifier(n_neighbors=7) # create MajorityVote classifier, which includes all model parameters
                clf.fit(X, y)                  # fit training data using the classifier
                y_pred = clf.predict(X)        # take the classifier and run it on the training data
                train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
                print('\t-7- training error: %.3f' % train_error)
                ### ========== TODO : END ========== ###
                
                
                
                ### ========== TODO : START ========== ###
                # part e: use cross-validation to compute average training and test error of classifiers
                print('Investigating various classifiers...')
                majority_clf = MajorityVoteClassifier()
                random_clf = RandomClassifier()
                decision_tree_clf = DecisionTreeClassifier(criterion = "entropy")
                knn_clf = KNeighborsClassifier(n_neighbors=5)
                print("Majority:", error(majority_clf, X, y))
                print("Random:",error(random_clf, X, y))
                print("Decision Tree:",error(decision_tree_clf, X, y))
                print("KNN:",error(knn_clf, X, y))
                ### ========== TODO : END ========== ###
            
            
            
                ### ========== TODO : START ========== ###
                # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
                print('Finding the best k for KNeighbors classifier...')
                
                a = np.arange(1, 51, 2)
            
                score = []
                for num in a: 
                    clf = KNeighborsClassifier(n_neighbors=num)
                    s = 1- sum(cross_val_score(clf, X, y, cv = 10))/float(10)
                    score.append(s)
                print(score)
                print(np.array(score).argmax())
                plt.plot(a, score)
                plt.xlabel('number of neighbors')
                plt.ylabel('validation error')
                plt.legend()
                plt.show()
                
                ### ========== TODO : END ========== ###
                
                
                
                ### ========== TODO : START ========== ###
                # part g: investigate decision tree classifier with various depths
                print('Investigating depths...')
                
                a = np.arange(1, 21)
                train_error = []
                test_error = []
                cv_error = []
                for num in a: 
                    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = num)
                    s = 1- sum(cross_val_score(clf, X, y, cv = 10))/10
                    cv_error.append(s)
                    (tr, te) = error(clf, X, y)
                    train_error.append(tr)
                    test_error.append(te)
                plt.plot(a, train_error,color = 'b', label = "train error")
                #TODO
                plt.plot(a, cv_error,color = 'g', label = "cross-validation error")
                plt.plot(a, test_error,color = 'r', label = "test error")
                plt.xlabel('depth limit')
                plt.ylabel('error')
                plt.legend()
                plt.show()'''
    ### ========== TODO : END ========== ###

    ### ========== TODO : END ========== ###
    
    best_depth = 3
    best_k = 7
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')    
    knn_train_errs= []
    knn_test_errs = []
    tree_train_errs = []
    tree_test_errs = []
    random_train_errs = []
    random_test_errs = []
    majority_train_errs = []
    majority_test_errs = []
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,train_size=0.9,random_state=100)
    
    for i in np.arange(0.1,1.1,0.1):
        clf_tree = DecisionTreeClassifier("entropy",max_depth=3)
        clf_knn  = KNeighborsClassifier(7)
        clf_majority = MajorityVoteClassifier()
        clf_random = RandomClassifier()
        #used to calculate the train/test err of a single split
        tree_train_err = 0
        tree_test_err = 0
        knn_train_err = 0
        knn_test_err = 0
        random_train_err = 0
        random_test_err = 0
        majority_train_err = 0
        majority_test_err = 0
        train_size = 0.999 if i == 1 else i
        for j in range(100):
            X_real_train, Junk, Y_real_train, Junk1 = train_test_split(X_train,Y_train,train_size=train_size,random_state=j)
           
            # Tree error 
            clf_tree.fit(X_real_train,Y_real_train)
            y_pred_train_tree = clf_tree.predict(X_real_train)
            y_pred_test_tree = clf_tree.predict(X_test)
            tree_train_err += 1 - metrics.accuracy_score(Y_real_train,y_pred_train_tree,normalize=True)
            tree_test_err += 1 - metrics.accuracy_score(Y_test,y_pred_test_tree,normalize=True)

            # KNN error 
            clf_knn.fit(X_real_train,Y_real_train)
            y_pred_train_knn = clf_knn.predict(X_real_train)
            y_pred_test_knn = clf_knn.predict(X_test)
            knn_train_err += 1 - metrics.accuracy_score(Y_real_train,y_pred_train_knn,normalize=True)
            knn_test_err +=  1 - metrics.accuracy_score(Y_test,y_pred_test_knn,normalize=True)

            clf_random.fit(X_real_train,Y_real_train)
            y_pred_train_random = clf_random.predict(X_real_train)
            y_pred_test_random = clf_random.predict(X_test)
            random_train_err += 1 - metrics.accuracy_score(Y_real_train,y_pred_train_random,normalize=True)
            random_test_err +=  1 - metrics.accuracy_score(Y_test,y_pred_test_random,normalize=True)

            clf_majority.fit(X_real_train,Y_real_train)
            y_pred_train_majority = clf_majority.predict(X_real_train)
            y_pred_test_majority = clf_majority.predict(X_test)
            majority_train_err += 1 - metrics.accuracy_score(Y_real_train,y_pred_train_majority,normalize=True)
            majority_test_err +=  1 - metrics.accuracy_score(Y_test,y_pred_test_majority,normalize=True)

        tree_train_errs.append(tree_train_err/100) 
        tree_test_errs.append(tree_test_err/100)
        knn_train_errs.append(knn_train_err/100)
        knn_test_errs.append(knn_test_err/100)
        random_train_errs.append((random_train_err)/100) 
        random_test_errs.append((random_test_err)/100)
        majority_train_errs.append(majority_train_err/100)
        majority_test_errs.append(majority_test_err/100)

    '''    plt.plot(np.arange(0.1,1.1,0.1),tree_train_errs,'g',label="DecisionTree_train")
    plt.plot(np.arange(0.1,1.1,0.1),tree_test_errs,'b',label="DecisionTree_test")
    plt.plot(np.arange(0.1,1.1,0.1),knn_train_errs,'r',label="KNN_train")
    plt.plot(np.arange(0.1,1.1,0.1),knn_test_errs,'y',label="KNN_test")
    plt.plot(np.arange(0.1,1.1,0.1),random_train_errs,'r',label="random_train")
    plt.plot(np.arange(0.1,1.1,0.1),random_test_errs,'y',label="random_test")
    plt.plot(np.arange(0.1,1.1,0.1),majority_train_errs,'r',label="majority_train")
    plt.plot(np.arange(0.1,1.1,0.1),majority_test_errs,'y',label="majority_test")
    '''
    #decision tree 
    plt.plot(np.arange(0.1,1.1,0.1),tree_train_errs,'g',label="DecisionTree_train")
    plt.plot(np.arange(0.1,1.1,0.1),tree_test_errs,'b',label="DecisionTree_test")
    plt.plot(np.arange(0.1,1.1,0.1),random_train_errs,'r',label="random_train")
    plt.plot(np.arange(0.1,1.1,0.1),random_test_errs,'c',label="random_test")
    plt.plot(np.arange(0.1,1.1,0.1),majority_train_errs,'m',label="majority_train")
    plt.plot(np.arange(0.1,1.1,0.1),majority_test_errs,'k',label="majority_test")
    plt.legend(loc=4, fontsize = 6)
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.show()

    #knn
    plt.plot(np.arange(0.1,1.1,0.1),knn_train_errs,'g',label="KNN_train")
    plt.plot(np.arange(0.1,1.1,0.1),knn_test_errs,'b',label="KNN_test")
    plt.plot(np.arange(0.1,1.1,0.1),random_train_errs,'r',label="random_train")
    plt.plot(np.arange(0.1,1.1,0.1),random_test_errs,'c',label="random_test")
    plt.plot(np.arange(0.1,1.1,0.1),majority_train_errs,'m',label="majority_train")
    plt.plot(np.arange(0.1,1.1,0.1),majority_test_errs,'k',label="majority_test")
    plt.legend(loc=4, fontsize = 6)
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.show()

    #both 
    plt.plot(np.arange(0.1,1.1,0.1),knn_train_errs,'g',label="KNN_train")
    plt.plot(np.arange(0.1,1.1,0.1),knn_test_errs,'b',label="KNN_test")
    plt.plot(np.arange(0.1,1.1,0.1),tree_train_errs,'r',label="DecisionTree_train")
    plt.plot(np.arange(0.1,1.1,0.1),tree_test_errs,'c',label="DecisionTree_test")
    plt.legend(loc=4, fontsize = 6)
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
