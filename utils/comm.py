def get_world_size() -> int:
    return 1


def get_rank() -> int:
    return 0


def get_local_rank() -> int:
    return 0


def get_local_size() -> int:
    return 1


def is_main_process() -> bool:
    return True


def synchronize():
    return


def gather(data, dst=0, group=None):
    return [data]
