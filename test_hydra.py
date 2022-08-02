import hydra

@hydra.main(config_path='configs', config_name='test')
def main(cfg):
    print(f'cfg: {cfg}')
    

if __name__=='__main__':
    main()
