from load_from_config import dataset_from_config, model_from_config
import os
# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

(ds, bs, total_num), _ = dataset_from_config(regime='test')

print(bs, total_num)
print(ds.as_numpy_iterator().next())