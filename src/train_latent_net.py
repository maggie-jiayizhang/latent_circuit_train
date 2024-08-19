import os
import sys
from util_train import *
from training_loop import *

if __name__ == "__main__":
    # batch training routine
    m = int(sys.argv[1])
    result_dir = sys.argv[2]
    job_id = int(sys.argv[3])

    config_fn = os.path.join(result_dir, "config.json")
    print(config_fn)
    training_loop(m, config_fn, job_id)