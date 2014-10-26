import arff
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
import csv

def writeArff(filename, data, relation, classes):
    outfile = open(filename, 'w', newline='')
    outfile.write('@relation ' + relation + '\n')
    for attr in range(len(data[0]) - 1):
        outfile.write('@attribute \'attr' + str(attr) + '\' real\n')
    outfile.write('@attribute \'class\' ' + classes + '\n')
    outfile.write('@data\n')
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(data)

def appendClassAttribute(reducedData, classAttribute):
    outtable = []
    for i in range(len(reducedData)):
        outrow = []
        for col in reducedData[i]:
            outrow.append(col)
        outrow.append(classAttribute[i])
        outtable.append(outrow)
    return outtable

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



    for n in range(1, XArr.shape[1] + 1):
        # Randomized Projection
        print(datasetName, 'GaussianRandomProjection')
        grp = random_projection.GaussianRandomProjection(n_components=n)
        model = grp.fit(XArr)
        grp_xform = model.transform(XArr)
        outtable = appendClassAttribute(grp_xform, classes)
        writeArff(
            '../' + datasetName + '/data/' + datasetName + '_rp_' + str(n) + '.arff',
            outtable,
            datasetName + ' - RP with ' + str(n) + ' component(s)',
            classEnum)

        # PCA
        print(datasetName, 'PCA', 'n_components = ' + str(n))
        pca = PCA(n_components=n)
        pca_transformed = pca.fit_transform(XArr)
        outtable = appendClassAttribute(pca_transformed, classes)
        writeArff(
            '../' + datasetName + '/data/' + datasetName + '_pca_' + str(n) + '.arff',
            outtable,
            datasetName + ' - PCA with ' + str(n) + ' component(s)',
            classEnum)

        # ICA
        print(datasetName, 'FastICA', 'n_components = ' + str(n))
        ica = FastICA(n_components=n)
        ica_transformed = ica.fit_transform(XArr)
        outtable = appendClassAttribute(ica_transformed, classes)
        writeArff(
            '../' + datasetName + '/data/' + datasetName + '_ica_' + str(n) + '.arff',
            outtable,
            datasetName + ' - ICA with ' + str(n) + ' component(s)',
            classEnum)




doDimRed('pendigits', '{0,1,2,3,4,5,6,7,8,9}')
doDimRed('diabetes', '{tested_negative,tested_positive}')
