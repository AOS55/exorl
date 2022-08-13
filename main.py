import os
import hydra


@hydra.main(config_path='configs/.', config_name='main')
def main(cfg):
    snapshot = f"./../../../data/models/{cfg.obs_type}/{cfg.domain}/{cfg.unsupervised_agent}/{cfg.seed}/snapshot_{cfg.snapshot_ts}.pt"
    if cfg.mode == "online":
        if not os.path.exists(snapshot):
            os.system(f"python ./../../../pretrain.py agent={cfg.unsupervised_agent} domain={cfg.domain}")
        os.system(f"python ./../../../finetune.py agent={cfg.unsupervised_agent} task={cfg.task} snapshot_ts={cfg.snapshot_ts} obs_type={cfg.obs_type}")
        
    elif cfg.mode == "offline":
        if not os.path.exists(snapshot):
            os.system(f"python ./../../../pretrain.py agent={cfg.unsupervised_agent} domain={cfg.domain}")
        # TODO: add way to move data into new directory
        os.system(f"python ./../../../sampling.py agent={cfg.unsupervised_agent} task={cfg.task}")
        os.system(f"python ./../../../train_offline.py agent={cfg.offline_agent} task={cfg.task}")
    else:
        print(f'Mode: {cfg.mode} is unrecognized')

if __name__=='__main__':
    main()
