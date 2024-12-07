import six
import os
import sys
import torch
import logging

log = logging.getLogger(__name__)

def prepare_logger(logger, debug=False, save_to_file=None):
    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    # formatter = logging.Formatter(fmt='[%(levelname)s] [%(filename)12s:%(lineno)5d]:\t%(message)s')
    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    logger.addHandler(console_hdl)
    if save_to_file is not None and not os.path.exists(save_to_file):
        file_hdl = logging.FileHandler(save_to_file)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
         logger.setLevel(logging.INFO)
    logger.propagate = False


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    if v.lower() in ("true", "t", "1", "false", "f", "0"):
        return v.lower() in ("true", "t", "1")
    else:
        raise Exception(f"Invalide boolean flag {v}. Please use one of {('true', 't', '1', 'false', 'f', '0')}")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, positional_arg=False, **kwargs):
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(
            prefix + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')


def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because cuda is not available in your machine.\n \
    Please: 1. Install cuda to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
                                                                                                                     ):
    try:
        if use_cuda == True and torch.cuda.is_available() == False:
            log.error(err)
            sys.exit(1)
    except Exception as e:
        pass
