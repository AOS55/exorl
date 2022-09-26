import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import urllib3
from io import BytesIO
import numpy as np
import torch


def download_data_from_s3(obs_type='states', env='SimplePointBot', agent='diayn', priors=10, pretrain_steps=50000, root_dir='data/datasets'):

    if agent=='diayn':
        elems = [obs_type, env, agent, str(priors), str(pretrain_steps)]
        dir_path = os.path.join(*elems)
    else:
        elems = [obs_type, env, agent, str(pretrain_steps)]
        dir_path = os.path.join(*elems)

    bucket_name = 'urlsuite-data'
    region_name = 'eu-west-2'
    resource_type = 's3'
    s3 = boto3.client(resource_type, region_name=region_name, config=Config(signature_version=UNSIGNED))
    write_dir = os.path.join(root_dir, dir_path)

    if os.path.exists(write_dir):
        print(f'file path already exists: {write_dir}')
        return None
    else:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=dir_path)
        print(dir_path)
        files = response.get("Contents")
        http = urllib3.PoolManager()
        os.mkdir(write_dir)
        for file in files:
            key = file['Key']
            url = f'https://{bucket_name}.{resource_type}.{region_name}.amazonaws.com/{key}'
            r = http.request('GET', url)
            data = np.load(BytesIO(r.data))
            print(f'key: {key}')
            print(f'data: {dir_path}')
            np.savez(os.path.join(root_dir, key), dict(data))
    return None


def download_model_from_s3(obs_type='states', env='SimplePointBot', agent='diayn', priors=20, seed=1, root_dir='data/models'):

    if agent=='diayn':
        elems = [obs_type, env, agent, str(priors), str(seed)]
        dir_path = os.path.join(*elems)
    else:
        elems = [obs_type, env, agent, seed]
        dir_path = os.path.join(*elems)

    bucket_name = 'urlsuite-models'
    region_name = 'eu-west-2'
    resource_type = 's3'
    s3 = boto3.client(resource_type, region_name=region_name, config=Config(signature_version=UNSIGNED))
    write_dir = os.path.join(root_dir, dir_path)

    if os.path.exists(write_dir):
        print(f'file path already exists: {write_dir}')
        return None
    else:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=dir_path)
        files = response.get("Contents")
        http = urllib3.PoolManager()
        os.mkdir(write_dir)
        for file in files:
            key = file['Key']
            url = f'https://{bucket_name}.{resource_type}.{region_name}.amazonaws.com/{key}'
            r = http.request('GET', url)
            model = torch.load(BytesIO(r.data))
            torch.save(model, os.path.join(root_dir, key))

def download_all_models_from_s3(root_dir='data/models'):
    bucket_name = 'urlsuite-models'
    region_name = 'eu-west-2'
    resource_type = 's3'
    s3 = boto3.client(resource_type, region_name=region_name, config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='')
    files = response.get("Contents")
    http = urllib3.PoolManager()
    os.mkdir(root_dir)
    for file in files:
            key = file['Key']
            url = f'https://{bucket_name}.{resource_type}.{region_name}.amazonaws.com/{key}'
            r = http.request('GET', url)
            model = torch.load(BytesIO(r.data))
            torch.save(model, os.path.join(root_dir, key))


if __name__=='__main__':
    download_model_from_s3()