
import copy
from einops import rearrange, reduce, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, hidden_dim):
        super(Encoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm_a = nn.LayerNorm(hidden_dim)
        self.final_norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, pre_norm, src_a, src_v, attn_mask=None, src_key_padding_mask=None):

        for i in range(self.num_layers):
            src_a = self.layers[i](pre_norm, src_a, src_v, attn_mask=attn_mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)
            src_v = self.layers[i](pre_norm, src_v, src_a, attn_mask=attn_mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)

        if pre_norm:
            src_a = self.final_norm_a(src_a)
            src_v = self.final_norm_v(src_v)

        return src_a, src_v


class HANLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, pre_norm, src_q, src_v, attn_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = rearrange(src_q, 'b t c -> t b c')
        src_v = rearrange(src_v, 'b t c -> t b c')
        
        if pre_norm:
            src_q_pre_norm = self.norm1(src_q)

            if with_ca:
                src1 = self.cm_attn(src_q_pre_norm, src_v, src_v, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            else:
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)

            src_q_pre_norm = self.norm2(src_q)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q_pre_norm))))
            src_q = src_q + self.dropout2(src2)

            return rearrange(src_q, 't b c -> b t c')
        
        else:
            if with_ca:
                src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q.clone(), src_q, src_q, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
                src_q = self.norm1(src_q)
            else:
                src2 = self.self_attn(src_q, src_q, src_q, attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)
                src_q = self.norm1(src_q)

            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
            src_q = src_q + self.dropout2(src2)
            src_q = self.norm2(src_q)

            return rearrange(src_q, 't b c -> b t c')
        

class HybridAttentionNet(nn.Module):
    def __init__(
            self,
            num_classes,
            input_v_2d_dim,
            input_v_3d_dim,
            input_a_dim,
            hidden_dim,
            nhead,
            ff_dim,
            num_han_layers,
            num_proj_layers,
            num_MMIL_layers,
            dropout=0.1,
            pre_norm=False
        ):
        super(HybridAttentionNet, self).__init__()

        self.num_classes = num_classes
        self.input_v_2d_dim = input_v_2d_dim   # 2048: ResNet152, 768: CLIP large
        self.input_v_3d_dim = input_v_3d_dim
        self.input_a_dim = input_a_dim         # 128: VGGish, 512: CLAP
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.ff_dim = ff_dim
        self.num_han_layers = num_han_layers
        self.num_proj_layers = num_proj_layers
        self.num_MMIL_layers = num_MMIL_layers
        self.dropout = dropout
        self.pre_norm = pre_norm

        self.fc_prob = nn.Linear(self.hidden_dim, self.num_classes)
        self.fc_frame_att = MLP(self.hidden_dim, self.hidden_dim*2, self.num_classes, num_layers=self.num_MMIL_layers)
        self.fc_av_att = MLP(self.hidden_dim, self.hidden_dim*2, self.num_classes, num_layers=self.num_MMIL_layers)

        self.fc_a = MLP(self.input_a_dim, self.hidden_dim*2, self.hidden_dim, num_layers=2)
        self.fc_fusion = MLP(self.input_v_2d_dim + self.input_v_3d_dim, self.hidden_dim*2, self.hidden_dim, num_layers=self.num_proj_layers)

        self.hat_encoder = Encoder(HANLayer(d_model=self.hidden_dim, nhead=self.nhead, dim_feedforward=self.ff_dim, dropout=self.dropout),
                                   num_layers=self.num_han_layers, hidden_dim=self.hidden_dim)

    def forward(self, a_feat, vis_2d_feat, vis_3d_feat=None, attn_mask=None, src_key_padding_mask=None):

        B, T = a_feat.shape[:2]

        # audio feature projection
        a_feat = self.fc_a(a_feat)

        # 2d visual feature projection (avg pool if needed)
        if vis_2d_feat.shape[1] != T:   # input 2d features are from ResNet152
            vis_2d_feat = reduce(vis_2d_feat, 'b (t1 t2) c -> b t1 c', 'mean', t1=T)

        # 3d visual feature projection and fused with 2d visual feature
        if vis_3d_feat is not None:
            vis_feat = torch.cat((vis_2d_feat, vis_3d_feat), dim=-1)
            vis_feat = self.fc_fusion(vis_feat)
        else:
            vis_feat = vis_2d_feat

        # HAN
        if attn_mask is not None:
            attn_mask = ~attn_mask  # change values of pad position from False to True
            attn_mask = repeat(attn_mask, 'b l s -> (b h) l s', h=self.nhead)
        a_feat, vis_feat = self.hat_encoder(self.pre_norm, a_feat, vis_feat, attn_mask, src_key_padding_mask)

        # prediction
        feat = torch.cat([a_feat.unsqueeze(-2), vis_feat.unsqueeze(-2)], dim=-2)
        frame_logits = self.fc_prob(feat)               # (B, T, 2, C)
        frame_probs = torch.sigmoid(frame_logits)       # (B, T, 2, C)

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(feat), dim=1)   # (B, T, 2, C)
        av_att = torch.softmax(self.fc_av_att(feat), dim=2)         # (B, T, 2, C)
        temporal_probs = (frame_att * frame_probs)
        global_probs = (temporal_probs * av_att).sum(dim=2).sum(dim=1)  # (B, C)

        a_probs = temporal_probs[:, :, 0, :].sum(dim=1)       # (B, C)
        v_probs = temporal_probs[:, :, 1, :].sum(dim=1)       # (B, C)

        a_frame_logits = frame_logits[:, :, 0, :]
        v_frame_logits = frame_logits[:, :, 1, :]

        return global_probs, a_probs, v_probs, a_frame_logits, v_frame_logits, a_feat, vis_feat
    
    def calculate_loss(self, args, outputs, labels, valid_masks,
                       a_logits=None, v_logits=None,
                       a_thresholds=None, v_thresholds=None,
                       a_class_freq=None, v_class_freq=None):

        video_probs, _, _, a_frame_logits, v_frame_logits, a_feat, v_feat = outputs
        weak_labels, a_pseudo_labels, v_pseudo_labels = labels  # (B, C), (B, T, C), (B, T, C)

        device = a_frame_logits.device

        # Initialize all losses
        loss_video = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_a_seg = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_v_seg = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_a_mixup = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_v_mixup = torch.tensor(0.0, device=device, dtype=torch.float32)


        if args.cal_video_loss:
            video_probs.clamp_(min=1e-7, max=1 - 1e-7)
            loss_video = F.binary_cross_entropy(video_probs, weak_labels)

        if args.cal_segment_loss:
            loss_a_seg = self.calculate_segment_loss(a_frame_logits, a_pseudo_labels, weak_labels,
                                                    a_logits, a_thresholds, args.apply_uncertainty,
                                                    args.apply_reweighting, args.reweight_type, a_class_freq,
                                                    args.pos_weight, args.neg_weight)
            loss_v_seg = self.calculate_segment_loss(v_frame_logits, v_pseudo_labels, weak_labels,
                                                    v_logits, v_thresholds, args.apply_uncertainty,
                                                    args.apply_reweighting, args.reweight_type, v_class_freq,
                                                    args.pos_weight, args.neg_weight)
        if args.cal_mixup_loss:
            loss_a_mixup = self.calculate_mixup_loss(a_feat, a_pseudo_labels, weak_labels,
                                                    a_logits, a_thresholds, args.apply_uncertainty_mixup,
                                                    self.fc_prob, args.alpha)
            loss_v_mixup = self.calculate_mixup_loss(v_feat, v_pseudo_labels, weak_labels,
                                                     v_logits, v_thresholds, args.apply_uncertainty_mixup,
                                                     self.fc_prob, args.alpha)

        loss = loss_video + \
                (loss_a_seg + loss_v_seg) * args.segment_loss_weight + \
                (loss_a_mixup + loss_v_mixup) * args.mixup_loss_weight

        loss_dict = {
            'loss_video': loss_video.item(),
            'loss_a_valor': loss_a_seg.item(),
            'loss_v_valor': loss_v_seg.item(),
            'loss_a_mixup': loss_a_mixup.item(),
            'loss_v_mixup': loss_v_mixup.item(),
            'loss_all': loss.item()
        }
        return loss, loss_dict
    
    def calculate_segment_loss(self, frame_logits, pseudo_labels, weak_labels,
                                logits, thresholds, apply_uncertainty,
                                apply_reweighting, reweight_type, class_freq,
                                pos_weight, neg_weight):
        B, T, C = frame_logits.shape

        if apply_uncertainty:
            assert logits is not None, 'logits cannot be None'
            assert thresholds is not None, 'thresholds cannot be None'
            # replace -inf with the smallest logit in the batch data that belongs to that class.
            logit_min = reduce(logits, 'b t c -> c', 'min')
            thresholds = torch.where(thresholds == float('-inf'), logit_min-1, thresholds)
            repeat_weak_labels = repeat(weak_labels, 'b c -> b t c', t=T)
            uncertainty_labels = torch.where(repeat_weak_labels == 1, torch.sigmoid((logits - thresholds) / 2), repeat_weak_labels)
            frame_labels = uncertainty_labels
        else:
            frame_labels = pseudo_labels

        loss = F.binary_cross_entropy_with_logits(frame_logits, frame_labels, reduction='none')

        if apply_reweighting:
            repeat_weak_labels = repeat(weak_labels, 'b c -> b t c', t=T)
            loss_weight = torch.ones_like(loss)
            
            if reweight_type == 'fixed':
                loss_weight = torch.where(repeat_weak_labels == 1, pos_weight, neg_weight)
            elif reweight_type == 'inverse_freq':
                pos_class_freq = class_freq.sum()
                neg_class_freq = C - pos_class_freq
                loss_weight = torch.where(repeat_weak_labels == 1, neg_class_freq * pos_weight, pos_class_freq)
            
            loss = (loss * loss_weight).mean()
        else:
            loss = loss.mean()

        return loss
    
    def calculate_mixup_loss(self, seg_feats, hard_seg_labels, weak_labels,
                             logits, thresholds, apply_uncertainty_mixup,
                             classifier, alpha):
        B, T, C = hard_seg_labels.shape

        if apply_uncertainty_mixup:
            logit_min = reduce(logits, 'b t c -> c', 'min')
            thresholds = torch.where(thresholds == float('-inf'), logit_min-1, thresholds)
            repeat_weak_labels = repeat(weak_labels, 'b c -> b t c', t=T)
            seg_labels = torch.where(repeat_weak_labels == 1, torch.sigmoid((logits - thresholds) / 2), repeat_weak_labels)
        else:
            seg_labels = hard_seg_labels

        seg_feats = rearrange(seg_feats, 'b t d -> (b t) d')
        seg_labels = rearrange(seg_labels, 'b t c -> (b t) c')

        mixed_seg_feats, seg_labels_1, seg_labels_2, lam = self.mixup_data(seg_feats, seg_labels, alpha)
        mixed_seg_logits = classifier(mixed_seg_feats)
        loss = lam * F.binary_cross_entropy_with_logits(mixed_seg_logits, seg_labels_1) + \
                (1 - lam) * F.binary_cross_entropy_with_logits(mixed_seg_logits, seg_labels_2)

        return loss

    def mixup_data(self, x, y, alpha):
        N = x.shape[0]
        if alpha > 0:
            beta_dist = torch.distributions.Beta(alpha, alpha)
            lam = beta_dist.sample()
        else:
            lam = 1

        index = torch.randperm(N, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_1, y_2 = y, y[index]

        return mixed_x, y_1, y_2, lam
    
    @torch.no_grad()
    def get_pred(self, outputs):

        video_probs, _, _, a_frame_logits, v_frame_logits, _, _ = outputs     # (B, C), (B, T, C), (B, T, C)

        video_probs.squeeze_(dim=0)
        a_frame_logits.squeeze_(dim=0)
        v_frame_logits.squeeze_(dim=0)

        video_preds = (video_probs > 0.5).int()   # (C,)
        pred_a = get_seg_pred(a_frame_logits.sigmoid(), video_preds, 0.5).permute(1, 0).detach().cpu().numpy()  # (C, T)
        pred_v = get_seg_pred(v_frame_logits.sigmoid(), video_preds, 0.5).permute(1, 0).detach().cpu().numpy()  # (C, T)

        pred_av = pred_a * pred_v

        return pred_a, pred_v, pred_av
    
    
def get_seg_pred(seg_output, video_label, threshold):
    # seg_output (tensor): (T, C), either logits or probabilities
    # video_label (tensor): (C,)

    T, C = seg_output.size()
    video_label = repeat(video_label, 'c -> t c', t=T)
    seg_pred = (seg_output > threshold)
    seg_pred = torch.logical_and(seg_pred, video_label).int()

    return seg_pred