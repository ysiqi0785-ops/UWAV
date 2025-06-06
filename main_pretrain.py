import os
import wandb
import logging
import argparse
import numpy as np
from einops import repeat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader import build_dataset, build_dataloader
from model import build_model
from utils.system import setup_logging, setup_directory, setup_wandb, save_args, seed_everything
from utils.train_utils import calculate_grad_norm, AverageMeter, WarmUpCosineAnnealingLR, BaseScheduler
from utils.eval_utils import AVVPEvaluator, show_result, calculate_classwise_thresholds
from utils.eval_metrics import segment_level, event_level
    

def train(args, model, train_loader, optimizer, epoch, device):
    
    model.train()
    train_loss_avg_meter = AverageMeter()

    for batch_idx, batch_data in enumerate(train_loader):
        clip_feats, clap_feats = batch_data['clip_feat'].to(device), batch_data['clap_feat'].to(device)
        weak_labels = batch_data['weak_label'].float().to(device)    # (B, C)
        valid_masks = batch_data['valid_mask'].to(device)       # (B, T)
        attn_masks = batch_data['attn_mask'].to(device)         # (B, T, T)
        labels = batch_data['gt_label'].float().to(device)  # (B, T, C)
        batch_size, T = clip_feats.size()[:2]

        optimizer.zero_grad()
        
        outputs = model(clap_feats, clip_feats, valid_mask=valid_masks, attn_mask=attn_masks)
        loss, loss_dict = model.calculate_loss(args, outputs, labels, valid_masks)
        train_loss_avg_meter.update(loss_dict, batch_size)

        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        total_grad_norm = calculate_grad_norm(model)
        optimizer.step()

        if args.use_wandb:
            wandb.log({
                'grad_norm': total_grad_norm,
                'iters': (epoch - 1) * len(train_loader) + batch_idx,
                'epoch': epoch
            })
            for loss_name, loss_val in loss_dict.items():
                wandb.log({f'train {loss_name}': loss_val})

    train_loss_dict = train_loss_avg_meter.average()

    return train_loss_dict


@torch.no_grad()
def eval(args, model, data_loader, device):

    model.eval()

    if isinstance(args.a_thres, np.ndarray):
        args.a_thres = torch.from_numpy(args.a_thres).to(device)
    if isinstance(args.v_thres, np.ndarray):
        args.v_thres = torch.from_numpy(args.v_thres).to(device)

    if args.save_logits:
        logit_saved_dir = args.checkpoint_dir[:-1] if args.checkpoint_dir[-1] == '/' else args.checkpoint_dir
        logit_saved_dir = '/'.join(logit_saved_dir.split('/')[:-1])
        a_logit_saved_dir = os.path.join(logit_saved_dir, args.dataset+'_Dataset', 'a_segmentwise_logits')
        v_logit_saved_dir = os.path.join(logit_saved_dir, args.dataset+'_Dataset', 'v_segmentwise_logits')
        if not os.path.exists(a_logit_saved_dir):
            os.makedirs(a_logit_saved_dir, exist_ok=True)
        if not os.path.exists(v_logit_saved_dir):
            os.makedirs(v_logit_saved_dir, exist_ok=True)


    loss_avg_meter = AverageMeter()
    avvp_evaluator = AVVPEvaluator()
    for batch_idx, batch_data in enumerate(data_loader):
        video_name = batch_data['video_name'][0]
        clip_feats, clap_feats = batch_data['clip_feat'].to(device), batch_data['clap_feat'].to(device)
        valid_masks = batch_data['valid_mask'].to(device)
        attn_masks = batch_data['attn_mask'].to(device)
        weak_labels = batch_data['weak_label'].float().to(device)
        duration = batch_data['duration'][0].item()
        batch_size, T = clip_feats.size()[:2]


        outputs = model(clap_feats, clip_feats, valid_mask=valid_masks, attn_mask=attn_masks)

        if args.dataset == 'UnAV':
            labels = batch_data['gt_label'].float().to(device)  # (B, T, C)
            _, loss_dict = model.calculate_loss(args, outputs, labels, valid_masks)
            loss_avg_meter.update(loss_dict, batch_size)

        a_logits, v_logits = outputs
        a_logits, v_logits = a_logits.squeeze(0), v_logits.squeeze(0)   # (T, C)
        a_probs, v_probs = a_logits.sigmoid(), v_logits.sigmoid()       # (T, C)
        av_probs = a_probs * v_probs

        if args.dataset == 'UnAV':
            pred_a = torch.zeros_like(av_probs)
            pred_v = torch.zeros_like(av_probs)
            pred_av = (av_probs >= 0.5).to(torch.int32)
        else:
            pred_a = (a_logits > args.a_thres)   # (T, C)
            pred_v = (v_logits > args.v_thres)   # (T, C)
            pred_av = torch.logical_and(pred_a, pred_v).to(torch.float32)


        if args.label_filtering:
            T, C = pred_a.shape
            weak_labels = repeat(weak_labels.squeeze(0), 'c -> t c', t=T).to(torch.float32)[:, :C]  # remove background
            pred_a = torch.logical_and(pred_a, weak_labels).to(torch.float32)
            pred_v = torch.logical_and(pred_v, weak_labels).to(torch.float32)
            pred_av = torch.logical_and(pred_av, weak_labels).to(torch.float32)


        # remove padded tokens
        pred_a = pred_a.permute(1, 0).cpu().detach().numpy()    # ndarray, (C, T)
        pred_a = pred_a[:, :duration]     # remove the padded segments (frames)
        pred_v = pred_v.permute(1, 0).cpu().detach().numpy()    # ndarray, (C, T)
        pred_v = pred_v[:, :duration]     # remove the padded segments (frames)
        pred_av = pred_av.permute(1, 0).cpu().detach().numpy()  # ndarray, (C, T)
        pred_av = pred_av[:, :duration]     # remove the padded segments (frames)


        if args.dataset == 'UnAV':
            GT_av = batch_data['gt_label'].float().squeeze(0).permute(1, 0).numpy()    # (C, T)
            GT_av = GT_av[:, :duration]     # remove the padded segments (frames)
            GT_a = np.zeros(GT_av.shape, dtype=float)
            GT_v = np.zeros(GT_av.shape, dtype=float)
        elif args.dataset == 'LLP':
            GT_a = batch_data['gt_a_label'].float().cpu().squeeze(0).permute(1, 0).numpy()  # (C, T)
            GT_v = batch_data['gt_v_label'].float().cpu().squeeze(0).permute(1, 0).numpy()  # (C, T)
            GT_av = GT_a * GT_v


        # AVVP evaluation
        f_a, f_v, f, f_av = segment_level(pred_a, pred_v, pred_av, GT_a, GT_v, GT_av)
        avvp_evaluator.update('segment', f_a, f_v, f, f_av)
        f_a, f_v, f, f_av = event_level(pred_a, pred_v, pred_av, GT_a, GT_v, GT_av)
        avvp_evaluator.update('event', f_a, f_v, f, f_av)


        # save labels
        if args.save_labels:
            a_label_save_path = os.path.join(args.a_label_saved_dir, video_name + '.npy')
            v_label_save_path = os.path.join(args.v_label_saved_dir, video_name + '.npy')
            np.save(a_label_save_path, np.transpose(pred_a))    # (T, C)
            np.save(v_label_save_path, np.transpose(pred_v))    # (T, C)

        # save logits
        if args.save_logits:
            a_logit_save_path = os.path.join(a_logit_saved_dir, video_name + '.npy')
            v_logit_save_path = os.path.join(v_logit_saved_dir, video_name + '.npy')
            np.save(a_logit_save_path, a_logits.squeeze(0).cpu().detach().numpy())    # (T, C)
            np.save(v_logit_save_path, v_logits.squeeze(0).cpu().detach().numpy())    # (T, C)

    eval_loss_dict = loss_avg_meter.average()

    return avvp_evaluator, eval_loss_dict


def main():
    parser = argparse.ArgumentParser()

    # system configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)

    # basic dataset configs
    parser.add_argument("--dataset", type=str, choices=['LLP', 'UnAV'], required=True)
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'pseudo_label_generation', 'test'])
    parser.add_argument('--num_classes', type=int, default=100)

    # feature configs
    parser.add_argument("--audio_dir", type=str, help="audio features dir")
    parser.add_argument("--video_dir", type=str, help="2D visual features dir")
    parser.add_argument("--st_dir", type=str, help="3D visual features dir")
    
    parser.add_argument("--clip_feat_dir", type=str, help="dir where segment features from CLIP are saved")
    parser.add_argument("--clap_feat_dir", type=str, help="dir where segment features from CLAP are saved")
    parser.add_argument("--clip_event_feat_path", type=str)
    parser.add_argument("--clap_event_feat_path", type=str)

    # annotation configs
    parser.add_argument("--label_all", type=str, help="weak label csv file")
    parser.add_argument("--label_train", type=str, help="weak train csv file")
    parser.add_argument("--label_val", type=str, help="weak val csv file")
    parser.add_argument("--label_test", type=str, help="weak test csv file")
    parser.add_argument("--gt_audio_csv", type=str, help="ground-truth audio event annotations")
    parser.add_argument("--gt_visual_csv", type=str, help="ground-truth visual event annotations")

    parser.add_argument("--v_logit_dir", type=str, help="visual segment-level logit dir")
    parser.add_argument("--a_logit_dir", type=str, help="audio segment-level logit dir")
    parser.add_argument("--v_threshold_path", type=str, help="")
    parser.add_argument("--a_threshold_path", type=str, help="")
    parser.add_argument("--v_pseudo_data_dir", type=str, help="visual segment-level pseudo labels dir")
    parser.add_argument("--a_pseudo_data_dir", type=str, help="audio segment-level pseudo labels dir")

    # basic training hyper-parameters
    parser.add_argument('--loss_type', type=str, default='weak', choices=['valor', 'weak'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='the value for gradient clipping (0 means no gradient clipping)')
    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument('--pos_weight', type=float, default=5, help='weight for positive classes')
    parser.add_argument('--neg_weight', type=float, default=1, help='weight for negative classes')

    # optimizer hyper-parameters
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # scheduler hyper-parameters
    parser.add_argument('--scheduler', type=str, default='warm_up_cos_anneal', help='which scheduler to use')
    parser.add_argument('--step_size', type=int, default=10, help='step size for learning scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for learning scheduler')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='the number of epochs for warm up')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='the minimum lr for lr decay')

    # model hyper-parameters
    parser.add_argument("--model", type=str, default='SimpleNet', help="which model to use")
    parser.add_argument("--input_v_2d_dim", type=int, default=2048)
    parser.add_argument("--input_v_3d_dim", type=int, default=512)
    parser.add_argument("--input_a_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pre_norm", action="store_true")

    # eval hyper-parameters
    parser.add_argument('--a_thres', type=float, default=-1)
    parser.add_argument('--v_thres', type=float, default=13)
    parser.add_argument('--av_thres', type=float, default=13)
    parser.add_argument("--threshold_type", type=str, choices=['classwise', 'class_agnostic'])
    parser.add_argument("--label_filtering", action="store_true")
    parser.add_argument("--num_thresholds", type=int, default=100, help="number of thresholds for grid search (used for AVE only)")
    parser.add_argument("--save_labels", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_classwise_thresholds", action="store_true")
    
    # log configs
    parser.add_argument("--prefix", type=str, default='PREFIX')
    parser.add_argument("--log_dir", type=str, default='train_logs/')
    parser.add_argument("--checkpoint_dir", type=str, help='where model weights are saved')
    parser.add_argument("--checkpoint_model", type=str, default='checkpoint_best.pt', help='which model checkpoint will be used for evaluation')
    parser.add_argument("--save_interval", type=int, default=10, help='how many epochs to save one checkpoint')

    # wandb configurations
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='Baseline')
    parser.add_argument("--wandb_run_name", type=str)

    args = parser.parse_args()
    
    if (args.threshold_type is not None) and (args.dataset == 'UnAV'):
        raise ValueError("NOT calculating thresholds for the UnAV dataset!")
    

    args = setup_directory(args)
    setup_logging(filename=os.path.join(args.log_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    save_args(args)
    if args.use_wandb:
        setup_wandb(args.wandb_project_name, args.wandb_run_name, args)


    # Set random seed and device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    clip_event_feats = torch.from_numpy(np.load(args.clip_event_feat_path)).to(device)
    clap_event_feats = torch.from_numpy(np.load(args.clap_event_feat_path)).to(device)
    model = build_model(args, args.model, clip_event_feats, clap_event_feats)
    model = model.to(device)


    if args.mode == 'train':
        train_dataset = build_dataset(args, 'train')
        val_dataset   = build_dataset(args, 'val')
        train_loader  = build_dataloader(args, train_dataset, 'train')
        val_loader    = build_dataloader(args, val_dataset, 'val')
        assert args.num_classes == len(train_dataset.categories), 'args.num_classes is not the same as len(dataset.categories)'

        # Create optimizer, scheduler
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

        if args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
        elif args.scheduler == 'warm_up_cos_anneal':
            scheduler = WarmUpCosineAnnealingLR(optimizer, args.warm_up_epoch, args.epochs, args.lr_min, args.lr)
        else:
            scheduler = BaseScheduler(optimizer)


        best_F = {'Seg-a': 0.0, 'Seg-v': 0.0, 'Seg-av': 0.0, 'Seg-type': 0.0, 'Event-av': 0.0}
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):

            cur_lr = optimizer.param_groups[0]['lr']
            train_loss_dict = train(args, model, train_loader, optimizer, epoch, device)

            scheduler.step()

            avvp_evaluator, val_loss_dict = eval(args, model, val_loader, device)
            avvp_F_scores_dict = avvp_evaluator.output_result()


            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_epoch_{}.pt".format(epoch)))

            if avvp_F_scores_dict['Seg-av'] > best_F['Seg-av']:
                best_F = avvp_F_scores_dict
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
            
            if args.use_wandb:
                wandb.log({
                    'Segment-level audio F score': avvp_F_scores_dict['Seg-a'],
                    'Segment-level visual F score': avvp_F_scores_dict['Seg-v'],
                    'Segment-level type F score': avvp_F_scores_dict['Seg-type'],
                    'lr': cur_lr,
                    'epoch': epoch
                })
                for loss_name, loss_val in val_loss_dict.items():
                    wandb.log({f'val {loss_name}': loss_val})

            train_log = 'Epoch[%2d/%2d](lr:%.6f) Train Loss: %.3f  Val Loss: %.3f  Val Seg-av: %.3f'%(
                        epoch, args.epochs, cur_lr, train_loss_dict['loss_all'], val_loss_dict['loss_all'], avvp_F_scores_dict['Seg-av'])
            logger.info(train_log)

        logger.info('-'*70)
        logger.info(f'Best AVVP F-scores:  Seg-av: {best_F["Seg-av"]:.3f}\t Event-av: {best_F["Event-av"]:.3f} (at epoch {best_epoch})')

    elif args.mode == 'pseudo_label_generation':
        val_dataset  = build_dataset(args, 'val')
        val_loader   = build_dataloader(args, val_dataset, 'val')
        
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_model)))

        # calculate classwise thresholds
        a_thres, v_thres = calculate_classwise_thresholds(args, model, val_loader, device)
        args.a_thres = a_thres   # (C,)
        args.v_thres = v_thres   # (C,)
        logger.info(f'a_thres: {args.a_thres.cpu().numpy()}, v_thres: {args.v_thres.cpu().numpy()}')


        if args.save_classwise_thresholds:
            thres_saved_dir = args.checkpoint_dir[:-1] if args.checkpoint_dir[-1] == '/' else args.checkpoint_dir
            thres_saved_dir = '/'.join(thres_saved_dir.split('/')[:-1])
            thres_saved_dir = os.path.join(thres_saved_dir, args.dataset+'_Dataset')
            if not os.path.exists(thres_saved_dir):
                os.makedirs(thres_saved_dir, exist_ok=True)
            np.save(os.path.join(thres_saved_dir, 'a_classwise_threhsolds.npy'), a_thres.cpu().numpy())
            np.save(os.path.join(thres_saved_dir, 'v_classwise_threhsolds.npy'), v_thres.cpu().numpy())


        val_dataset  = build_dataset(args, 'val')
        val_loader   = build_dataloader(args, val_dataset, 'val')
        avvp_evaluator, _ = eval(args, model, val_loader, device=device)
        avvp_F_scores_dict = avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Val', avvp_F_scores_dict)

        test_dataset  = build_dataset(args, 'test')
        test_loader   = build_dataloader(args, test_dataset, 'test')
        avvp_evaluator, _ = eval(args, model, test_loader, device=device)
        avvp_F_scores_dict = avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Test', avvp_F_scores_dict)

        del val_dataset, test_dataset, val_loader, test_loader

        # Generate logits and binary pseudo labels for the training split
        train_dataset  = build_dataset(args, 'train')
        train_loader   = build_dataloader(args, train_dataset, 'val') # batch_size = 1 and no shuffle
        _, _ = eval(args, model, train_loader, device=device)

        del train_dataset, train_loader

    elif args.mode == 'test':
        val_dataset  = build_dataset(args, 'val')
        test_dataset = build_dataset(args, 'test')
        val_loader   = build_dataloader(args, val_dataset, 'val')
        test_loader  = build_dataloader(args, test_dataset, 'test')
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_model)))

        val_avvp_evaluator, _ = eval(args, model, val_loader, device=device)
        val_avvp_F_scores_dict = val_avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Val', val_avvp_F_scores_dict)

        test_avvp_evaluator, _ = eval(args, model, test_loader, device=device)
        test_avvp_F_scores_dict = test_avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Test', test_avvp_F_scores_dict)

    else:
        logger.info('Please specify args.mode!')
        

if __name__ == '__main__':
    main()