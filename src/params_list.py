import yaml
from src.utils import get_dataset_params

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

kobe20 = get_dataset_params(
    config,"kobe20",period=4.29112,offset=None,drop_indices=[3,7],
    system='johnson',observatory='k20',binning=False, mag_offset=0.08
    )

kobe21 = get_dataset_params(
    config,"kobe21",period=4.29112,offset=None,drop_indices=None,
    system='johnson',observatory='k21',binning=False, mag_offset=1.266
    )

pmo = get_dataset_params(
    config,"pmo",period=4.29112,offset=None,drop_indices=[0,2,20],
    system='sdss',observatory='pmo',binning=False, mag_offset=0
    )

params_list = [pmo, kobe20, kobe21]