#  -h, --help            show this help message and exit
#  --arch {vit_tiny,vit_small,vit_base,xcit,deit_tiny,deit_small,alexnet,convnext_base,convnext_large,convnext_small,convnext_tiny,densenet121,densenet161,densenet169,densenet201,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,efficientnet_v2_l,efficientnet_v2_m,efficientnet_v2_s,get_model,get_model_builder,get_model_weights,get_weight,googlenet,inception_v3,list_models,maxvit_t,mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3,mobilenet_v2,mobilenet_v3_large,mobilenet_v3_small,regnet_x_16gf,regnet_x_1_6gf,regnet_x_32gf,regnet_x_3_2gf,regnet_x_400mf,regnet_x_800mf,regnet_x_8gf,regnet_y_128gf,regnet_y_16gf,regnet_y_1_6gf,regnet_y_32gf,regnet_y_3_2gf,regnet_y_400mf,regnet_y_800mf,regnet_y_8gf,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext101_64x4d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,squeezenet1_0,squeezenet1_1,swin_b,swin_s,swin_t,swin_v2_b,swin_v2_s,swin_v2_t,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,vit_b_16,vit_b_32,vit_h_14,vit_l_16,vit_l_32,wide_resnet101_2,wide_resnet50_2,xcit_large_24_p16,xcit_large_24_p8,xcit_medium_24_p16,xcit_medium_24_p8,xcit_nano_12_p16,xcit_nano_12_p8,xcit_small_12_p16,xcit_small_12_p8,xcit_small_24_p16,xcit_small_24_p8,xcit_tiny_12_p16,xcit_tiny_12_p8,xcit_tiny_24_p16,xcit_tiny_24_p8}
#                        Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.
#  --patch_size PATCH_SIZE
#                        Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory. Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision training (--use_fp16
#                        false) to avoid unstabilities.
#  --out_dim OUT_DIM     Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.
#  --norm_last_layer NORM_LAST_LAYER
#                        Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
#  --momentum_teacher MOMENTUM_TEACHER
#                        Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.
#  --use_bn_in_head USE_BN_IN_HEAD
#                        Whether to use batch normalizations in projection head (Default: False)
#  --warmup_teacher_temp WARMUP_TEACHER_TEMP
#                        Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.
#  --teacher_temp TEACHER_TEMP
#                        Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.
#  --warmup_teacher_temp_epochs WARMUP_TEACHER_TEMP_EPOCHS
#                        Number of warmup epochs for the teacher temperature (Default: 30).
#  --use_fp16 USE_FP16   Whether or not to use half precision for training. Improves training time and memory requirements, but can provoke instability and slight decay of performance. We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with
#                        bigger ViTs.
#  --weight_decay WEIGHT_DECAY
#                        Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.
#  --weight_decay_end WEIGHT_DECAY_END
#                        Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.
#  --clip_grad CLIP_GRAD
#                        Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.
#  --batch_size_per_gpu BATCH_SIZE_PER_GPU
#                        Per-GPU batch-size : number of distinct images loaded on one GPU.
#  --epochs EPOCHS       Number of epochs of training.
#  --freeze_last_layer FREEZE_LAST_LAYER
#                        Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.
#  --lr LR               Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.
#  --warmup_epochs WARMUP_EPOCHS
#                        Number of epochs for the linear learning-rate warm up.
#  --min_lr MIN_LR       Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.
#  --optimizer {adamw,sgd,lars}
#                        Type of optimizer. We recommend using adamw with ViTs.
#  --drop_path_rate DROP_PATH_RATE
#                        stochastic depth rate
#  --global_crops_scale GLOBAL_CROPS_SCALE [GLOBAL_CROPS_SCALE ...]
#                        Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)
#  --local_crops_number LOCAL_CROPS_NUMBER
#                        Number of small local views to generate. Set this parameter to 0 to disable multi-crop training. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1."
#  --local_crops_scale LOCAL_CROPS_SCALE [LOCAL_CROPS_SCALE ...]
#                        Scale range of the cropped image before resizing, relatively to the origin image. Used for small local view cropping of multi-crop.
#  --data_path DATA_PATH
#                        Please specify path to the ImageNet training data.
#  --output_dir OUTPUT_DIR
#                        Path to save logs and checkpoints.
#  --saveckp_freq SAVECKP_FREQ
#                        Save checkpoint every x epochs.
#  --seed SEED           Random seed.
#  --num_workers NUM_WORKERS
#                        Number of data loading workers per GPU.
#  --dist_url DIST_URL   url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html
#  --local_rank LOCAL_RANK
#                        Please ignore and do not set this argument.

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path /data/keti/syh/img_Db --output_dir /data/keti/syh/exp/DINO_imagenet_test --batch_size_per_gpu 512
#python main_dino.py --arch vit_base \
#--data_path /data/keti/syh/ReID/MSMT17 \
#--output_dir /data/keti/syh/exp/DINO_MSMT17_train \
#--teacher_temp 0.07 \
#--warmup_teacher_temp_epochs 50 \
#--use_fp16 False \
#--clip_grad 0.3 \
#--batch_size_per_gpu 64 \
#--epochs 400 \
#--freeze_last_layer 3 \
#--lr 0.00075 \
#--min_lr 2e-06 \
#--global_crops_scale 0.25 1.0 \
#--local_crops_scale 0.05 0.25 \
#--local_crops_number 10 \

# train dino with reid shape
#python main_dino_reid.py --arch vit_base \
#--data_path /data/keti/syh/ReID/MSMT17 \
#--output_dir /data/keti/syh/exp/DINO_MSMT17_train_reid_shape_256_112_ratio_2_1 \
#--teacher_temp 0.07 \
#--warmup_teacher_temp_epochs 50 \
#--use_fp16 False \
#--clip_grad 0.3 \
#--batch_size_per_gpu 128 \
#--epochs 400 \
#--freeze_last_layer 3 \
#--lr 0.00075 \
#--min_lr 2e-06 \
#--global_crops_scale 0.25 1.0 \
#--local_crops_scale 0.05 0.25 \
#--local_crops_number 10 \
#--img_size 256 128

# 23.11.21 reid image dataset is narrow view than imagenet dbca
# study from S
#
#python main_dino_reid.py --arch vit_base \
#--data_path /data/keti/syh/ReID/MSMT17 \
#--output_dir /data/keti/syh/exp/DINO_MSMT17_train_reid_shape_256_112_ratio_2_1 \
#--teacher_temp 0.07 \
#--warmup_teacher_temp_epochs 50 \
#--use_fp16 False \
#--clip_grad 0.3 \
#--batch_size_per_gpu 80 \
#--epochs 400 \
#--freeze_last_layer 3 \
#--lr 0.00075 \
#--min_lr 2e-06 \
#--global_crops_scale 0.5 1.0 \
#--local_crops_scale 0.05 0.5 \
#--local_crops_number 10 \
#--img_size 256 128

#
# 23.12.15 OLP train
python main_dino_reid.py --arch vit_base \
--data_path /data/keti/syh/ReID/MSMT17 \
--output_dir /data/keti/syh/exp/DINO_MSMT17_train_reid_shape_256_128_ratio_2_1_OLP \
--teacher_temp 0.07 \
--warmup_teacher_temp_epochs 50 \
--use_fp16 False \
--clip_grad 0.3 \
--batch_size_per_gpu 64 \
--epochs 400 \
--freeze_last_layer 3 \
--lr 0.00075 \
--min_lr 2e-06 \
--global_crops_scale 0.5 1.0 \
--local_crops_scale 0.05 0.5 \
--local_crops_number 10 \
--img_size 256 128 \
--stride_size 12




