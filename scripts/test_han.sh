set -e

CHECKPOINT_DIR=./temp_train_logs/HAN_soft_label_reweight_soft_mixup_LLP_20250321_163632

python main.py \
    --mode test \
    --prefix HAN_soft_label_reweight_soft_mixup \
    --log_dir test_logs \
    --checkpoint_dir "$CHECKPOINT_DIR/checkpoints" \
    --model HAN \
    --dataset LLP \
    --cal_video_loss \
    --cal_segment_loss \
    --apply_uncertainty \
    --apply_reweighting \
    --reweight_type inverse_freq \
    --pos_weight 0.5 \
    --cal_mixup_loss \
    --alpha 1.7 \
    --apply_uncertainty_mixup \
    --audio_dir ./data/LLP/feats/vggish \
    --video_dir ./data/LLP/feats/res152 \
    --st_dir ./data/LLP/feats/r2plus1d_18 \
    --input_v_2d_dim 2048 \
    --input_a_dim 128 \
    --label_train ./data/LLP/AVVP_train.csv \
    --label_val ./data/LLP/AVVP_val_pd.csv \
    --label_test ./data/LLP/AVVP_test_pd.csv \
    --gt_audio_csv ./data/LLP/AVVP_eval_audio.csv \
    --gt_visual_csv ./data/LLP/AVVP_eval_visual.csv \
    --hidden_dim 512 \
    --nhead 16 \
    --ff_dim 2048 \
    --num_proj_layers 1 \
    --num_han_layers 5 \
    --num_MMIL_layers 1