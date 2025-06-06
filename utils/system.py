import os
import time
import torch
import wandb
import random
import logging
import numpy as np


def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)


def setup_directory(args):
    # log directory
    args.log_dir = os.path.join(args.log_dir, '{}_{}_{}'.format(args.prefix, args.dataset, time.strftime('%Y%m%d_%H%M%S')))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # checkpoint directory
    if (args.mode == 'train') and (args.checkpoint_dir == None):
        checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = checkpoint_dir

    # label saved directory
    if args.save_labels:
        label_saved_dir = args.checkpoint_dir[:-1] if args.checkpoint_dir[-1] == '/' else args.checkpoint_dir
        label_saved_dir = '/'.join(label_saved_dir.split('/')[:-1])
        a_label_saved_dir = os.path.join(label_saved_dir, args.dataset+'_Dataset', 'a_segmentwise_pseudo_labels')
        v_label_saved_dir = os.path.join(label_saved_dir, args.dataset+'_Dataset', 'v_segmentwise_pseudo_labels')
        if not os.path.exists(a_label_saved_dir):
            os.makedirs(a_label_saved_dir, exist_ok=True)
        if not os.path.exists(v_label_saved_dir):
            os.makedirs(v_label_saved_dir, exist_ok=True)

        args.a_label_saved_dir = a_label_saved_dir
        args.v_label_saved_dir = v_label_saved_dir

    return args


def setup_wandb(project_name, run_name, args):
    wandb.init(project=project_name)
    if run_name != None:
        wandb.run.name = run_name
    wandb.config.update(args)


def save_args(args, save_args_name='args.txt'):
    args_dict = args.__dict__
    with open(os.path.join(args.log_dir, save_args_name), 'w') as f:
        f.writelines('-------------------------start-------------------------\n')
        for key, value in args_dict.items():
            f.writelines(key + ': ' + str(value) + '\n')
        f.writelines('--------------------------end--------------------------\n')


def show_args(logger, args):

    logger.info(f'{args.mode.capitalize()} on {args.dataset} dataset')

    logger.info(f'Model Arch.: {args.model}')
    if args.load_checkpoint is not None:
        logger.info(f'\tinitialize model weights from {args.load_checkpoint}')
    if args.use_clip_clap_feat:
        logger.info('\tuse CILP & CLAP features')

    logger.info(f'Training Loss:')
    if args.cal_video_loss:
        logger.info('\tvideo loss')
    if args.cal_segment_loss:
        logger.info(f'\tsegment loss (loss weight: {args.segment_loss_weight})')
        if args.apply_uncertainty:
            logger.info('\t\tapply uncertainty')
        if args.apply_reweighting:
            logger.info(f'\t\tapply reweighting (type: {args.reweight_type}, pos weight: {args.pos_weight}, neg weight: {args.neg_weight})')
    if args.cal_mixup_loss:
        logger.info(f'\tmixup loss (alpha: {args.alpha}, loss weight: {args.mixup_loss_weight})')
        if args.apply_uncertainty_mixup:
            logger.info('\t\tapply uncertainty mixup')


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
