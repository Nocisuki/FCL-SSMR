import argparse
import wandb, os

from methods.fedavg import FedAvg
from methods.ours import FCL_SSMR
from utils.data_manager import DataManager, setup_seed
import warnings

from utils.toolkit import count_parameters

warnings.filterwarnings('ignore')


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "ours":
        return FCL_SSMR(args)
    elif name == "fedavg":
        return FedAvg(args)
    # elif name == "target":
    #     return Target(args)
    # elif name == "ewc":
    #     return EWC(args)
    # elif name == "lwf":
    #     return LwF(args)
    # elif name == "fedcil":
    #     return Fedcil(args)
    # elif name == "refed":
    #     return ReFed(args)   
    else:
        assert 0


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=3, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="imgs_target", help='save the syn data')
    parser.add_argument('--project', type=str, default="TARGET", help='wandb project')
    parser.add_argument('--group', type=str, default="exp1", help='wandb group')
    parser.add_argument('--seed', type=int, default=223, help='random seed')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=3, help='num of tasks')
    parser.add_argument('--init_cls', type=int, default=5, help='num of class_orders in each task')
    parser.add_argument('--method', type=str, default="", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=50, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=10, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--err', type=int, default=2, help='')
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data')
    parser.add_argument('--kd', type=int, default=10, help='for kd loss')
    parser.add_argument('--m_rate', type=float, default=1.0, help='')
    parser.add_argument('--memory_size', type=int, default=300, help='the num of real data per task')
    parser.add_argument('--generator_model_file', type=str, default="", help='')
    parser.add_argument('--max_loss', type=float, default="inf", help='early stop max loss')
    parser.add_argument('--overlap_rate', type=float, default=0.4, help='generator parameter')
    parser.add_argument('--imgs_path', type=str, default="imgs", help='generator parameter')
    parser.add_argument('--g_rounds', type=int, default=2000, help='generator parameter')

    args = parser.parse_args()

    return args


def train(args):
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["dataset"],
        True,
        args["seed"],
        args["tasks"],
        args["overlap_rate"],
    )
    learner = get_learner(args["method"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # train for each task
    for task in range(args["tasks"]):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network),
                                                            count_parameters(learner._network,
                                                                             True))) 
        learner.incremental_train(data_manager)  # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))

#
if __name__ == '__main__':

    args = args_parser()
    # args.num_class = 200 if args.dataset == "tiny_imagenet" else 100
    args.increment = args.init_cls

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"
    if args.method == "ours":
        dir = "run"
        if not os.path.exists(dir):
            os.makedirs(dir)
        args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name)

    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
    args = vars(args)

    train(args)
