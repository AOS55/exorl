import subprocess
import os

def write_data_to_s3(root_dir='data/datasets'):
    bashCommand = f"aws s3 sync ./data/datasets/ s3://urlsuite-data/"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

def write_models_to_s3(root_dir='data/models'):
    bashCommand = f"aws s3 sync ./data/models s3://urlsuite-models/"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

if __name__=='__main__':
    write_data_to_s3()
    write_models_to_s3()
