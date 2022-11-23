import os
import sys
import yaml
import re

def main():
    dir = f'exp_local/{sys.argv[1]}'
    print(f'dir is: {dir}')
    dir_list = os.listdir(dir)
    dir_list = [os.path.join(dir, x) for x in dir_list]
    for directory in dir_list:
        src = directory
        dst = directory.split('/')[:-1]
        dst.append(get_name(directory))
        dst = os.path.join(*dst)
        print(f'src: {src}, dst: {dst}')
        os.rename(src, dst)
    return None

def get_name(dir):
    config = os.path.join(dir, '.hydra/config.yaml')
    with open(config, 'r') as stream:
        try:
            contents = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f'yaml.YAMLError is exc: {exc}')
    skill_dim = contents['skill_dim']
    snapshot_ts = contents['snapshot_ts']
    name = f"prioritized_sampling_{skill_dim}_{snapshot_ts}"
    return name
if __name__=='__main__':
    main()
