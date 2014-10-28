import os
import re

WEKA_CLASSPATH = 'C:/Program Files/Weka-3-6/weka.jar;C:/Users/Dave/Downloads/libsvm.jar'
CONFIGS = [
    'weka.clusterers.EM -I 100 -M 1.0E-6 -S 100',
    'weka.clusterers.SimpleKMeans -A "weka.core.EuclideanDistance -R first-last" -I 500 -S 10'
    ]

def no_fail_remove_empty_file(filename):
    try:
        if os.path.isfile(filename):
            if 0 == os.path.getsize(filename):
                os.remove(filename)
    except:
        pass

def runClusterers(dataset, numClusters):
    for filename in os.listdir('../' + dataset + '/data/'):
        print("Clustering " + filename)
        relPath = '../' + dataset + '/data/' + filename

        try:
            classIndex = str(int(re.findall(r'\d+', filename)[0]) + 1)
        except:
            continue

        for config in CONFIGS:
            sanitized_config = re.sub('[ \"\\\\]', '_', config)
            output_file = "../" + dataset + "/results/" + filename + sanitized_config + ".txt"
            if not os.path.isfile(output_file):
                error_file = "../" + dataset + "/errors/" + filename + sanitized_config + ".txt"
                standard_args = ' -t ' + relPath + ' -c ' + classIndex + ' -N ' + str(numClusters)
                insert_pos = config.index(' ')
                config_with_args = config[:insert_pos] + standard_args + config[insert_pos:]
                cmd = 'java -cp "' + WEKA_CLASSPATH + '" ' + config_with_args + ' > ' + output_file + ' 2> ' + error_file
                os.system(cmd)
                no_fail_remove_empty_file(error_file)
            else:
                print(filename + " aleady run, skipping")

runClusterers('diabetes', 2)
runClusterers('pendigits', 10)
