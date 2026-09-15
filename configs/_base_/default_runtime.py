weight = None
resume = False
seed = 1024
save_path = "exp/test"

num_worker = 4
batch_size_test = 1
empty_cache = False
enable_amp = False

test = dict(type="SemSegTester", verbose=True)
