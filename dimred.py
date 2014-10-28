import arff
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
import csv
import timeit

tree = True
pca = True
ica = True
rp = True

explainedVarianceTable = []
runtimeTable = []

def appendClassAttribute(reducedData, classAttribute):
    outtable = []
    for i in range(len(reducedData)):
        outrow = []
        for col in reducedData[i]:
            outrow.append(col)
        outrow.append(classAttribute[i])
        outtable.append(outrow)
    return outtable

def writeArff(filename, data, relation, classes):
    outfile = open(filename, 'w', newline='')
    outfile.write('@relation ' + relation + '\n')
    for attr in range(len(data[0]) - 1):
        outfile.write('@attribute \'attr' + str(attr) + '\' real\n')
    outfile.write('@attribute \'class\' ' + classes + '\n')
    outfile.write('@data\n')
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(data)

def logTime(start, stop, datasetName, function, n_components):
    duration = stop - start
    runtimeTable.append([duration, datasetName, function, n_components])

def doDimRed(datasetName, classEnum):
    global X, classes, diabetesArff, row, XArr, n, pca, pca_transformed, outtable, ica, ica_transformed
    X = []
    classes = []
    arffData = arff.load('../' + datasetName + '/data/' + datasetName + '.arff')
    for row in arffData:
        X.append(list(row)[:-1])
        classes.append(list(row)[-1])
    XArr = np.array(X)
    print(XArr.shape)

    if tree:
        # Tree-based feature selection
        print(datasetName, 'Tree-based feature selection')
        extraTreesClassifier = ExtraTreesClassifier()
        start = timeit.default_timer()
        X_new = extraTreesClassifier.fit(X, classes).transform(X)
        stop = timeit.default_timer()
        logTime(start, stop, datasetName, "tree", X_new.shape[1])
        print(X_new.shape)
        print(extraTreesClassifier.feature_importances_)
        outtable = appendClassAttribute(X_new, classes)
        writeArff(
            '../' + datasetName + '/data/' + datasetName + '_tree.arff',
            outtable,
            datasetName + '_tree_classifier',
            classEnum)


    for n in range(1, XArr.shape[1] + 1):
        # Randomized Projection
        if rp:
            print(datasetName, 'GaussianRandomProjection', 'n_components = ' + str(n))
            grp = random_projection.GaussianRandomProjection(n_components=n)
            start = timeit.default_timer()
            model = grp.fit(XArr)
            grp_xform = model.transform(XArr)
            stop = timeit.default_timer()
            logTime(start, stop, datasetName, "rp", n)
            outtable = appendClassAttribute(grp_xform, classes)
            writeArff(
                '../' + datasetName + '/data/' + datasetName + '_rp_' + str(n) + '.arff',
                outtable,
                datasetName + '_RP_with_' + str(n) + '_components',
            classEnum)

        # PCA
        if pca:
            print(datasetName, 'PCA', 'n_components = ' + str(n))
            pca = PCA(n_components=n)
            start = timeit.default_timer()
            pca_transformed = pca.fit_transform(XArr)
            stop = timeit.default_timer()
            logTime(start, stop, datasetName, "pca", n)
            print("Explained Variance Ratio", sum(pca.explained_variance_ratio_))
            explainedVarianceTable.append([datasetName, n, sum(pca.explained_variance_ratio_)])
            outtable = appendClassAttribute(pca_transformed, classes)
            writeArff(
                '../' + datasetName + '/data/' + datasetName + '_pca_' + str(n) + '.arff',
                outtable,
                datasetName + '_PCA_with_' + str(n) + '_components',
                classEnum)

        # ICA
        if ica:
            print(datasetName, 'FastICA', 'n_components = ' + str(n))
            ica = FastICA(n_components=n, max_iter=100000)
            start = timeit.default_timer()
            ica_transformed = ica.fit_transform(XArr)
            stop = timeit.default_timer()
            logTime(start, stop, datasetName, "ica", n)
            outtable = appendClassAttribute(ica_transformed, classes)
            writeArff(
                '../' + datasetName + '/data/' + datasetName + '_ica_' + str(n) + '.arff',
                outtable,
                datasetName + '_ICA_with_' + str(n) + '_components',
                classEnum)




doDimRed('pendigits', '{0,1,2,3,4,5,6,7,8,9}')
doDimRed('diabetes', '{tested_negative,tested_positive}')

with open('explained_variance_ratio.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    w.writerows(explainedVarianceTable)

with open('dimred_runtime.csv', 'w', newline='') as csvfile2:
    w = csv.writer(csvfile2)
    w.writerows(runtimeTable)