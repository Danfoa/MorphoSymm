import os
from pathlib import Path


if __name__ == '__main__':


    experiments_path = 'experiments/com_sample_eff_Solo-c2'
    experiments_path = Path(experiments_path)

    assert experiments_path.exists(), experiments_path

    group = 'C2'
    for path in list(experiments_path.glob('model=*')):
        if f'G={group}' in str(path): continue
        path.rename(path.parent / (f'G={group}_' + str(path.name)))
        print(f'{path.name} \t --> \t {(f"G={group}_"+ str(path.name))}')