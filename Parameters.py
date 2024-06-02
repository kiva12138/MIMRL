import argparse
from Utils import str2bools, str2floats, str2listoffints

def parse_args():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument("--task_name", default="test")

    # Data parameters
    parser.add_argument("--dataset", default='mosi_SDK', type=str)
    parser.add_argument("--normalize", default='0-0-0', type=str2bools)
    parser.add_argument("--log_scale", default='0-0-0', type=str2bools) # l-a-v
    parser.add_argument("--text", default='text', type=str) # Only for CMUSDK&AVEC dataset
    parser.add_argument("--audio", default='covarep', type=str) # Only for CMUSDK&AVEC dataset
    parser.add_argument("--video", default='facet41', type=str) # Only for CMUSDK&AVEC dataset
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--persistent_workers", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--task", default='regression', type=str, choices=['classification', 'regression'])
    parser.add_argument("--num_class", default=1, type=int)
    
    # Model parameters
    parser.add_argument("--d_common", default=128, type=int)
    parser.add_argument("--encoders", default='gru', type=str) # gru lstm conv
    parser.add_argument("--features_compose_t", default='mean', type=str) # mean sum cat
    parser.add_argument("--features_compose_k", default='mean', type=str) # mean sum cat
    parser.add_argument("--activate", default='gelu', type=str)
    parser.add_argument("--time_len", default=100, type=int)
    parser.add_argument("--d_hiddens", default='10-2-128=5-2-128', type=str2listoffints)
    parser.add_argument("--d_outs", default='10-2-128=5-2-128', type=str2listoffints)
    parser.add_argument("--dropout_mlp", default='0.5-0.5-0.5', type=str2floats)
    parser.add_argument("--dropout", default='0.5-0.5-0.5-0.5', type=str2floats)
    parser.add_argument("--bias", action='store_true')
    parser.add_argument("--ln_first", action='store_true')
    parser.add_argument("--res_project", default='1-1', type=str2bools)

    # VMI estimation
    parser.add_argument("--critic_type", default='separate', type=str) # separate concat
    parser.add_argument("--baseline_type", default='constant', type=str) # constant unnormalized gaussain
    parser.add_argument("--bound_type", default='infonce', type=str) # dv smile tuba nwj infonce js js_fgan smile interpolate
    parser.add_argument("--loss_mi_coefficient1", default='0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1', type=str2floats) # 11 in sum
    parser.add_argument("--loss_mi_coefficient2", default='0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1', type=str2floats) # 8 in sum
    parser.add_argument("--mi_lr_rate", default=1.0, type=float)
    parser.add_argument("--cmi_lr_rate", default=1.0, type=float)
    parser.add_argument("--k_neighbor", default=2, type=int)
    parser.add_argument("--radius", default=1.0, type=float)
    parser.add_argument("--cmi_last_acticate", default='sigmoid', type=str, choices=['hardtanh', 'sigmoid'])
    parser.add_argument("--stage1_n", default=1, type=int)
    
    # Training and optimization
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--loss", default='MAE', choices=['Focal', 'CE', 'BCE', 'RMSE', 'MSE', 'SIMSE', 'MAE', 'CCC'])
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--epochs_num", default=2, type=int)
    parser.add_argument("--optm", default="Adam", type=str, choices=['SGD', 'SAM', 'Adam'])
    parser.add_argument("--learning_rate", default=4e-3, type=float)
    parser.add_argument("--bert_freeze", default='no', type=str, choices=['part', 'no', 'all'])
    parser.add_argument("--bert_lr_rate", default=-1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--lr_decrease", default='step', type=str, choices=['multi_step', 'step', 'exp', 'plateau'])
    parser.add_argument("--lr_decrease_iter", default='60', type=str) # 50, or 50-75
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float) # 0.1/0.5 for exp
    parser.add_argument("--save_best_features", action='store_true')
    parser.add_argument("--print_params", action='store_true')
    parser.add_argument("--check_gradient", action='store_true')
    parser.add_argument("--parallel", action='store_true')
    parser.add_argument("--cuda", default="0", type=str)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    print(args)
