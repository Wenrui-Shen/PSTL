import os
from sacred import Experiment

ex = Experiment("PSTL", save_git_info=False) 

@ex.config
def my_config():
    ############################## setting ##############################
    version = "ntu60_xsub_j"
    dataset = "ntu60"   # ntu60 / ntu120 / pku
    split = "xsub"
    view = "joint"      # joint / motion / bone
    save_lp = False
    save_finetune = False
    save_semi = False
    pretrain_epoch = 150
    ft_epoch = 150
    lp_epoch = 150
    pretrain_lr = 5e-3
    lp_lr = 0.01
    ft_lr = 5e-3
    label_percent = 0.1
    weight_decay = 1e-5
    hidden_size = 256
    encoder_type = "stgcn"  # stgcn / gatr
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ################################ GATr ################################
    gatr_out_mv_channels = 1
    gatr_in_s_channels = 1
    gatr_hidden_mv_channels = 32
    gatr_hidden_s_channels = 1
    gatr_out_s_channels = 1
    gatr_num_blocks = 1
    gatr_num_heads = 8
    gatr_temporal_refinement = 5
    gatr_dropout = 0.5
    gatr_checkpoint_blocks = False
    gatr_translation_range = 0.5
    gatr_y_rotation_degrees = 30.0
    gatr_reflection_prob = 0.5
    gatr_pj_size = 4096
    ############################ down stream ############################
    # Set these overrides to a path to bypass the encoder-specific defaults below.
    weight_path = None
    log_path = None
    result_path = None
    resume_path = None
    checkpoint_interval = 10
    stgcn_weight_path = (
        './output/weight/v'+version+'_epoch_'+str(pretrain_epoch)+'_pretrain.pt'
    )
    gatr_weight_path = (
        './output/weight/v'+version+'_gatr_epoch_'+str(pretrain_epoch)+'_pretrain.pt'
    )
    train_mode = 'pretrain'  # lp / finetune / semi
    stgcn_log_path = './output/log/v'+version+'_'+train_mode+'.log'
    gatr_log_path = './output/log/v'+version+'_gatr_'+train_mode+'.log'
    stgcn_result_path = './result/'+dataset+'/'+split+'/'+view+'/'+version+'_'
    gatr_result_path = './result/'+dataset+'/'+split+'/'+view+'/'+version+'_gatr_'
    label_path = './result/'+dataset+'/'+split+'/label/label.pkl'
    ################################ GPU ################################
    # gpus = "0"
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    ########################## Skeleton Setting #########################
    batch_size = 128
    gatr_batch_size = 128
    channel_num = 3
    person_num = 2
    joint_num = 25
    max_frame = 50
    data_path = os.path.join('..', 'data', 'pstl', split)
    train_list = os.path.join(data_path, 'train_position.npy')  ## your data path
    test_list = os.path.join(data_path, 'val_position.npy')
    train_label = os.path.join(data_path, 'train_label.pkl')
    test_label = os.path.join(data_path, 'val_label.pkl')
    ########################### Data Augmentation #########################
    temperal_padding_ratio = 6
    shear_amp = 1
    mask_joint = 8
    mask_frame = 10
    ############################ Barlow Twins #############################
    pj_size = 6144
    lambd = 2e-4
