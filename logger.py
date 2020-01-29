import logging


def get_logger(level=logging.DEBUG, filename="./sim.log"):

    assert filename is not None

    logger = logging.getLogger(__name__)

    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('[%(asctime)s] %(message)s')
    #stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=filename, mode='a')
    file_handler.setLevel(logging.INFO)
    #file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
