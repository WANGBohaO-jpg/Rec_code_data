import os
import multiprocessing

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DRO_SSM")
    # Learning
    parser.add_argument("--lr", type=float, default=0.01, help="the learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="the weight decay for l2 normalizaton")
    parser.add_argument("--trainbatch", type=int, default=4096, help="the batch size for bpr loss training procedure")
    parser.add_argument("--testbatch", type=int, default=4096, help="the batch size of users for testing")
    parser.add_argument("--dataset", type=str, default="gowalla")
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--comment", type=str, default="", help="comment of running")
    parser.add_argument("--epochs", type=int, default=2001, help="total number of epochs")
    parser.add_argument("--multicore", type=int, default=0, help="whether we use multiprocessing or not in test")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--recdim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--model", type=str, default="mf", help="rec-model, support [mf, lgn]")
    parser.add_argument("--loss", type=str, default="softmax", help="loss function, support [bpr, softmax]")
    parser.add_argument("--norm_emb", type=int, default=1, help="whether normalize embeddings")
    parser.add_argument("--cuda", type=str, default="0", help="use which cuda")
    parser.add_argument("--full_batch", action="store_true")
    parser.add_argument("--resume_dir", type=str, default="")

    # SSM Loss
    parser.add_argument("--ssm_temp", type=float, default=0.1)
    parser.add_argument("--num_negative_items", type=int, default=64)
    parser.add_argument("--neg_coefficient", type=int, default=1)

    # LightGCN
    parser.add_argument("--layer", type=int, default=2, help="the layer num of lightGCN")
    parser.add_argument("--enable_dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.0)
    
    # XSimGCL or SimGCL or LightGCL
    parser.add_argument("--cl_rate", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--cl_temp", type=float, default=0.2)
    parser.add_argument("--cl_layer", type=int, default=0)
    parser.add_argument("--q", type=int, default=5)


    return parser.parse_args()


args = parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

all_models = ["lgn", "mf"]
config = {
    "model": args.model,
    "dataset": args.dataset,
    "train_batch": args.trainbatch,
    "n_layers": args.layer,
    "latent_dim_rec": args.recdim,
    "enable_dropout": args.enable_dropout,
    "keep_prob": args.keepprob,
    "test_u_batch_size": args.testbatch,
    "multicore": args.multicore,
    "loss": args.loss,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "norm_emb": args.norm_emb,
    "full_batch": args.full_batch,
    "num_negative_items": args.num_negative_items,
    
    "cuda": args.cuda,
    "ssm_temp": args.ssm_temp,
    "resume_dir": args.resume_dir,
    
    "neg_coefficient": args.neg_coefficient,
    
    "cl_rate": args.cl_rate,
    "eps": args.eps,
    "cl_temp": args.cl_temp,
    "cl_layer": args.cl_layer,
    "q": args.q,
}

CORES = multiprocessing.cpu_count() // 2
seed = args.seed
# dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
topks = eval(args.topks)
# tensorboard = args.tensorboard
comment = args.comment

METHOD_CAT = None
# if config["enable_group_weight"] == 1:
#     METHOD_CAT = "Group_weight"
# elif config["enable_group_emb"] == 1:
#     METHOD_CAT = "Group_emb"
# else:
#     METHOD_CAT = "Normal"


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
