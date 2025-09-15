import logging, os, time

def get_logger(output_dir: str, to_file: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(output_dir)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if to_file:
        fn = time.strftime("log_%Y%m%d_%H%M%S.txt")
        fh = logging.FileHandler(os.path.join(output_dir, fn))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
