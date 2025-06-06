import os
import wandb
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import build_dataset, build_dataloader
from model import build_model
from utils.system import setup_logging, setup_directory, setup_wandb, save_args, seed_everything, show_args
from utils.train_utils import calculate_grad_norm, AverageMeter, WarmUpCosineAnnealingLR, BaseScheduler
from utils.eval_utils import AVVPEvaluator, show_result
from utils.eval_metrics import segment_level, event_level


def train(args, model, data_loader, optimizer, epoch, device):

    model.train()
    train_loss_avg_meter = AverageMeter()

    # get class frequencies and classwise thresholds for class-balanced re-weighting & uncertainty-aware training
    a_class_freq = torch.from_numpy(data_loader.dataset.a_class_freq).to(device)   # (C,)
    v_class_freq = torch.from_numpy(data_loader.dataset.v_class_freq).to(device)   # (C,)
    if args.a_threshold_path is not None:
        a_classwise_thresholds = torch.from_numpy(np.load(args.a_threshold_path)).to(device)    # (C,)
    else:
        a_classwise_thresholds = torch.zeros(len(data_loader.dataset.categories), dtype=float, device=device)
    
    if args.v_threshold_path is not None:
        v_classwise_thresholds = torch.from_numpy(np.load(args.v_threshold_path)).to(device)    # (C,)
    else:
        v_classwise_thresholds = torch.zeros(len(data_loader.dataset.categories), dtype=float, device=device)

    for batch_idx, batch_data in enumerate(data_loader):
        video_res_feats, video_3d_feats, audio_feats = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device)
        # (B, 8*T, 2048), (B, T, 512), (B, T, 128)
        clip_feats, clap_feats = batch_data['clip_feat'].to(device), batch_data['clap_feat'].to(device)
        # (B, T, 768), (B, T, 512)
        weak_labels = batch_data['weak_label'].float().to(device)           # (B, C)
        valid_masks = batch_data['valid_mask'].to(device)                   # (B, T), torch.float32
        attn_masks = batch_data['attn_mask'].to(device)                     # (B, T, T), torch.bool
        batch_size, T = audio_feats.size()[:2]

        optimizer.zero_grad()

        if args.use_clip_clap_feat:
            outputs = model(clap_feats, clip_feats, attn_mask=attn_masks)
        else:
            outputs = model(audio_feats, video_res_feats, vis_3d_feat=video_3d_feats, attn_mask=attn_masks)


        a_pseudo_labels = batch_data['a_pseudo_label'].float().to(device)    # (B, T, C)
        v_pseudo_labels = batch_data['v_pseudo_label'].float().to(device)    # (B, T, C)
        labels = (weak_labels, a_pseudo_labels, v_pseudo_labels)

        a_logits = batch_data['a_logit'].float().to(device)    # (B, T, C)
        v_logits = batch_data['v_logit'].float().to(device)    # (B, T, C)

        loss, loss_dict = model.calculate_loss(args, outputs, labels, valid_masks,
                                               a_logits = a_logits, v_logits = v_logits,
                                               a_thresholds = a_classwise_thresholds, v_thresholds = v_classwise_thresholds,
                                               a_class_freq = a_class_freq, v_class_freq = v_class_freq)
        train_loss_avg_meter.update(loss_dict, batch_size)

        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        total_grad_norm = calculate_grad_norm(model)
        optimizer.step()

        if args.use_wandb:
            wandb.log({
                'grad_norm': total_grad_norm,
                'iters': (epoch - 1) * len(data_loader) + batch_idx,
                'epoch': epoch,
            })
            for loss_name, loss_val in loss_dict.items():
                wandb.log({f'train {loss_name}': loss_val})

    train_loss_dict = train_loss_avg_meter.average()

    return train_loss_dict


@torch.no_grad()
def eval(args, model, data_loader, device):

    model.eval()

    # get class frequencies and classwise thresholds for class-balanced re-weighting & uncertainty-aware training
    a_class_freq = torch.from_numpy(data_loader.dataset.a_class_freq).to(device)   # (C,)
    v_class_freq = torch.from_numpy(data_loader.dataset.v_class_freq).to(device)   # (C,)
    if args.a_threshold_path is not None:
        a_classwise_thresholds = torch.from_numpy(np.load(args.a_threshold_path)).to(device)    # (C,)
    else:
        a_classwise_thresholds = torch.zeros(len(data_loader.dataset.categories), dtype=float, device=device)
    
    if args.v_threshold_path is not None:
        v_classwise_thresholds = torch.from_numpy(np.load(args.v_threshold_path)).to(device)    # (C,)
    else:
        v_classwise_thresholds = torch.zeros(len(data_loader.dataset.categories), dtype=float, device=device)


    loss_avg_meter = AverageMeter()
    avvp_evaluator = AVVPEvaluator()

    for batch_idx, batch_data in enumerate(data_loader):
        video_name = batch_data['video_name'][0]
        video_res_feats, video_3d_feats, audio_feats = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device)
        clip_feats, clap_feats = batch_data['clip_feat'].to(device), batch_data['clap_feat'].to(device)
        weak_labels = batch_data['weak_label'].float().to(device)           # (B, C)
        valid_masks = batch_data['valid_mask'].to(device)
        attn_masks = batch_data['attn_mask'].to(device)
        batch_size, T = audio_feats.size()[:2]

        if args.use_clip_clap_feat:
            outputs = model(clap_feats, clip_feats, attn_mask=attn_masks)
        else:
            outputs = model(audio_feats, video_res_feats, vis_3d_feat=video_3d_feats, attn_mask=attn_masks)


        a_pseudo_labels = batch_data['a_pseudo_label'].float().to(device)    # (B, T, C)
        v_pseudo_labels = batch_data['v_pseudo_label'].float().to(device)    # (B, T, C)
        labels = (weak_labels, a_pseudo_labels, v_pseudo_labels)

        a_logits = batch_data['a_logit'].float().to(device)    # (B, T, C)
        v_logits = batch_data['v_logit'].float().to(device)    # (B, T, C)

        _, loss_dict = model.calculate_loss(args, outputs, labels, valid_masks,
                                            a_logits = a_logits, v_logits = v_logits,
                                            a_thresholds = a_classwise_thresholds, v_thresholds = v_classwise_thresholds,
                                            a_class_freq = a_class_freq, v_class_freq = v_class_freq)
        loss_avg_meter.update(loss_dict, batch_size)


        pred_a, pred_v, pred_av = model.get_pred(outputs)

        # ground-truth matrices
        GT_a = batch_data['gt_a_label'].float().cpu().squeeze(0).permute(1, 0).numpy()  # (C, T)
        GT_v = batch_data['gt_v_label'].float().cpu().squeeze(0).permute(1, 0).numpy()  # (C, T)
        GT_av = GT_a * GT_v

        # AVVP evaluation
        f_a, f_v, f, f_av = segment_level(pred_a, pred_v, pred_av, GT_a, GT_v, GT_av)
        avvp_evaluator.update('segment', f_a, f_v, f, f_av)
        f_a, f_v, f, f_av = event_level(pred_a, pred_v, pred_av, GT_a, GT_v, GT_av)
        avvp_evaluator.update('event', f_a, f_v, f, f_av)

    eval_loss_dict = loss_avg_meter.average()

    return avvp_evaluator, eval_loss_dict


def main():
    parser = argparse.ArgumentParser()

    # system configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)

    # basic dataset configs
    parser.add_argument("--dataset", type=str, choices=['LLP'], required=True)
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num_classes', type=int, default=25)

    # feature configs
    parser.add_argument("--audio_dir", type=str, default='./data/feats/vggish', help="audio features dir")
    parser.add_argument("--video_dir", type=str, default='./data/feats/res152', help="2D visual features dir")
    parser.add_argument("--st_dir", type=str, default='./data/feats/r2plus1d_18', help="3D visual features dir")
    
    parser.add_argument("--use_clip_clap_feat", action="store_true")
    parser.add_argument("--clip_feat_dir", type=str, help="dir where segment features from CLIP are saved")
    parser.add_argument("--clap_feat_dir", type=str, help="dir where segment features from CLAP are saved")

    # annotation configs
    parser.add_argument("--label_all", type=str, default="./data/AVVP_dataset_full.csv",
                        help="weak label csv file")
    parser.add_argument("--label_train", type=str, default="./data/AVVP_train.csv",
                        help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="./data/AVVP_val_pd.csv",
                        help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="./data/AVVP_test_pd.csv",
                        help="weak test csv file")
    parser.add_argument("--gt_audio_csv", type=str, default="./data/AVVP_eval_audio.csv",
                        help="ground-truth audio event annotations")
    parser.add_argument("--gt_visual_csv", type=str, default="./data/AVVP_eval_visual.csv",
                        help="ground-truth visual event annotations")
    
    parser.add_argument("--v_pseudo_data_dir", type=str, help="visual segment-level pseudo labels dir")
    parser.add_argument("--a_pseudo_data_dir", type=str, help="audio segment-level pseudo labels dir")
    parser.add_argument("--v_logit_dir", type=str, help="visual segment-level logit dir")
    parser.add_argument("--a_logit_dir", type=str, help="audio segment-level logit dir")
    parser.add_argument("--v_threshold_path", type=str, help="")
    parser.add_argument("--a_threshold_path", type=str, help="")

    # basic training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='the value for gradient clipping (0 means no gradient clipping)')
    parser.add_argument('--load_checkpoint', type=str)

    # model hyper-parameters
    parser.add_argument("--model", type=str, default='HAN', help="which model to use")
    parser.add_argument("--input_v_2d_dim", type=int, default=2048)
    parser.add_argument("--input_v_3d_dim", type=int, default=512)
    parser.add_argument("--input_a_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_han_layers", type=int, default=1)
    parser.add_argument("--num_proj_layers", type=int, default=2)
    parser.add_argument("--num_MMIL_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pre_norm", action="store_true")

    # loss configs & hyper-parameters
    parser.add_argument('--cal_video_loss', action="store_true")

    parser.add_argument('--cal_segment_loss', action="store_true")
    parser.add_argument('--segment_loss_weight', type=float, default=1.0, help='weight for segment loss')
    parser.add_argument("--apply_uncertainty", action="store_true", help='use probabilities as labels when calculating segment loss')
    parser.add_argument("--apply_reweighting", action="store_true", help='assign larger loss weight for positive events when calculating segment loss')
    parser.add_argument("--reweight_type", type=str, default='fixed', choices=['fixed', 'inverse_freq'])
    parser.add_argument('--pos_weight', type=float, default=7, help='weight for positive classes')
    parser.add_argument('--neg_weight', type=float, default=1, help='weight for negative classes')
    
    parser.add_argument('--cal_mixup_loss', action="store_true", help='calculate feature mixup loss')
    parser.add_argument('--alpha', type=float, default=2, help='hyper-parameter for the beta distribution')
    parser.add_argument("--apply_uncertainty_mixup", action="store_true", help='use probabilities as labels when calculating mixup loss')
    parser.add_argument('--mixup_loss_weight', type=float, default=1.0, help='weight for mixup loss')

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
    
    # log configs
    parser.add_argument("--prefix", type=str, default='PREFIX')
    parser.add_argument("--log_dir", type=str, default='train_logs/')
    parser.add_argument("--checkpoint_dir", type=str, help='where model weights are saved')
    parser.add_argument("--checkpoint_model", type=str, default='checkpoint_best.pt', help='which model checkpoint will be used for evaluation')
    parser.add_argument("--save_interval", type=int, default=5, help='how many epochs to save one checkpoint')

    # wandb configurations
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='Baseline')
    parser.add_argument("--wandb_run_name", type=str)

    # useless configs (to avoid error)
    parser.add_argument("--save_labels", action="store_true")

    args = parser.parse_args()


    # Initial setup
    args = setup_directory(args)
    setup_logging(filename=os.path.join(args.log_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    save_args(args)
    if args.use_wandb:
        setup_wandb(args.wandb_project_name, args.wandb_run_name, args)

    show_args(logger, args)


    # Set random seed and device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = build_model(args, args.model)
    model = model.to(device)

    if args.mode == 'train':
        train_dataset = build_dataset(args, 'train')
        val_dataset   = build_dataset(args, 'val')
        train_loader  = build_dataloader(args, train_dataset, 'train')
        val_loader    = build_dataloader(args, val_dataset, 'val')
        assert args.num_classes == len(train_dataset.categories), 'args.num_classes is not the same as len(dataset.categories)'


        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

        if args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'warm_up_cos_anneal':
            scheduler = WarmUpCosineAnnealingLR(optimizer, args.warm_up_epoch, args.epochs, args.lr_min, args.lr)
        else:
            scheduler = BaseScheduler(optimizer)


        best_metrics = {'Seg-a': 0.0, 'Seg-v': 0.0, 'Seg-av': 0.0, 'Seg-type': 0.0, 'Event-av': 0.0, 'Acc': 0.0}
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            
            cur_lr = optimizer.param_groups[0]['lr']
            train_loss_dict = train(args, model, train_loader, optimizer, epoch, device)
            
            scheduler.step()


            avvp_evaluator, val_loss_dict = eval(args, model, val_loader, device)
            avvp_F_scores_dict = avvp_evaluator.output_result()

            
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_epoch_{}.pt".format(epoch)))
            if (avvp_F_scores_dict['Seg-type'] > best_metrics['Seg-type']):
                best_metrics = avvp_F_scores_dict
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

            logger.info(f'Epoch[{epoch:3d}/{args.epochs:3d}](lr:{cur_lr:.6f}) '
                        f'Train Loss: {train_loss_dict["loss_all"]:.3f}  Val Loss: {val_loss_dict["loss_all"]:.3f}  '
                        f'Seg-a: {avvp_F_scores_dict["Seg-a"]:.3f}  Seg-v: {avvp_F_scores_dict["Seg-v"]:.3f}  '
                        f'Seg-av: {avvp_F_scores_dict["Seg-av"]:.3f}  Seg-type: {avvp_F_scores_dict["Seg-type"]:.3f}')
        
        logger.info('-'*70)
        logger.info(f'Best AVVP F-scores:  Seg-a: {best_metrics["Seg-a"]:.3f}  Seg-v: {best_metrics["Seg-v"]:.3f}  '
                    f'Seg-av: {best_metrics["Seg-av"]:.3f}  Seg-type: {best_metrics["Seg-type"]:.3f}  '
                    f'Seg-event: {best_metrics["Seg-event"]:.3f} (at epoch {best_epoch})')

    elif args.mode == 'val':
        val_dataset = build_dataset(args, 'val')
        val_loader  = build_dataloader(args, val_dataset, 'val')
        assert args.num_classes == len(val_dataset.categories), 'args.num_classes is not the same as len(dataset.categories)'

        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_model)))

        # Evaluation
        val_dataset  = build_dataset(args, 'val')
        val_loader   = build_dataloader(args, val_dataset, 'val')

        avvp_evaluator, _ = eval(args, model, val_loader, device=device)
        avvp_F_scores_dict = avvp_evaluator.output_result()
        show_result(logger, args.dataset, "Val", avvp_F_scores_dict)

    elif args.mode == 'test':
        val_dataset  = build_dataset(args, 'val')
        test_dataset = build_dataset(args, 'test')
        val_loader   = build_dataloader(args, val_dataset, 'val')
        test_loader  = build_dataloader(args, test_dataset, 'test')
        assert args.num_classes == len(val_dataset.categories), 'args.num_classes is not the same as len(dataset.categories)'

        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_model)))

        # Evaluation
        val_avvp_evaluator, _ = eval(args, model, val_loader, device=device)
        val_avvp_F_scores_dict = val_avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Val', val_avvp_F_scores_dict)

        test_avvp_evaluator, _ = eval(args, model, test_loader, device=device)
        test_avvp_F_scores_dict = test_avvp_evaluator.output_result()
        show_result(logger, args.dataset, 'Test', test_avvp_F_scores_dict)
        

if __name__ == '__main__':
    main()