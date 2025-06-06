set -e

PRETRAINED_TRANSFORMER_DIR=./temp_train_logs/TemporalTransformer_UnAV_20250321_121739

# LLP dataset
python main_pretrain.py \
    --mode pseudo_label_generation \
    --prefix TemporalTransformer \
    --log_dir test_logs \
    --checkpoint_dir "$PRETRAINED_TRANSFORMER_DIR/checkpoints" \
    --model TemporalTransformer \
    --dataset LLP \
    --audio_dir ./data/LLP/feats/vggish \
    --video_dir ./data/LLP/feats/res152 \
    --st_dir ./data/LLP/feats/r2plus1d_18 \
    --clip_feat_dir ./data/LLP/feats_CLIP/segment_feats \
    --clap_feat_dir ./data/LLP/feats_CLAP/segment_feats \
    --clip_event_feat_path ./data/LLP/feats_CLIP/event_feats/all_event_feats.npy \
    --clap_event_feat_path ./data/LLP/feats_CLAP/event_feats/all_event_feats.npy \
    --label_train ./data/LLP/AVVP_train.csv \
    --label_val ./data/LLP/AVVP_val_pd.csv \
    --label_test ./data/LLP/AVVP_test_pd.csv \
    --gt_audio_csv ./data/LLP/AVVP_eval_audio.csv \
    --gt_visual_csv ./data/LLP/AVVP_eval_visual.csv \
    --input_a_dim 512 \
    --input_v_2d_dim 768 \
    --input_v_3d_dim 1024 \
    --hidden_dim 1024 \
    --nhead 16 \
    --ff_dim 2048 \
    --dropout 0.3 \
    --num_layers 5 \
    --threshold_type classwise \
    --label_filtering \
    --save_labels \
    --save_logits \
    --save_classwise_thresholds