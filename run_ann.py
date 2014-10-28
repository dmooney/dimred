import os
import re

WEKA_CLASSPATH = 'C:/Program Files/Weka-3-6/weka.jar;C:/Users/Dave/Downloads/libsvm.jar'
CONFIGS = [
    'weka.classifiers.functions.MultilayerPerceptron -L 1.0 -M 0.0 -N 100 -V 0 -S 0 -E 20 -H a -R'
    ]


def no_fail_remove_empty_file(filename):
    try:
        if os.path.isfile(filename):
            if 0 == os.path.getsize(filename):
                os.remove(filename)
    except:
        pass

def runANNs(dataset):
    for filename in os.listdir('../' + dataset + '/data/'):
        print("Training on " + filename)
        relPath = '../' + dataset + '/data/' + filename

        try:
            classIndex = str(int(re.findall(r'\d+', filename)[0]) + 1)
        except:
            continue

        for config in CONFIGS:
            sanitized_config = re.sub('[ \"\\\\]', '_', config)
            output_file = "../" + dataset + "/ann_results/" + filename + sanitized_config + ".txt"
            if not os.path.isfile(output_file):
                error_file = "../" + dataset + "/errors/" + filename + sanitized_config + ".txt"
                standard_args = ' -t ' + relPath + ' -split-percentage 30 -i -k '
                insert_pos = config.index(' ')
                config_with_args = config[:insert_pos] + standard_args + config[insert_pos:]
                cmd = 'java -cp "' + WEKA_CLASSPATH + '" ' + config_with_args + ' > ' + output_file + ' 2> ' + error_file
                os.system(cmd)
                no_fail_remove_empty_file(error_file)
                no_fail_remove_empty_file(output_file)
            else:
                print(filename + " aleady run, skipping")

runANNs('diabetes')
runANNs('pendigits')
