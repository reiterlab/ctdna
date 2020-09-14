import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# create logger
logger = logging.getLogger('ctdna')

# create file handler
fh = logging.FileHandler('ctdna.log')
fh.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
# logger.propagate = False
logger.setLevel(logging.INFO)
