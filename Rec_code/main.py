import os

os.environ["MKL_NUM_THREADS"] = "10"
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'

from matplotlib import pyplot as plt
import torch.nn.functional as F

import dataloader
import model
from pprint import pprint

import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
import nni
from logger import CompleteLogger


if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================


dataroot = os.path.join("./normal_data", world.config["dataset"])
logroot = os.path.join("./log", world.config["dataset"])

dataset = dataloader.Loader(path=dataroot)

MODELS = {
    "lgn": model.LightGCN,
    "mf": model.PureMF,
    "XSimGCL": model.XSimGCL,
    "SimGCL": model.SimGCL,
    "LightGCL": model.LightGCL,
}

Recmodel = MODELS[world.model_name](world.config, dataset).cuda()
loss_func = utils.LossFunc(Recmodel, world.config, dataset)

# world.config['ssm_temp2'] = world.config['ssm_temp']

if world.config["loss"] == "bpr" or world.config["loss"] == "bce":
    world.config["norm_emb"] = 0
    world.config["num_negtive_items"] = 1

if "NNI_PLATFORM" in os.environ:
    save_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard")
    print("save_dir: ", save_dir)
    w: SummaryWriter = SummaryWriter(save_dir)
else:
    save_dir = join(
        logroot,
        time.strftime("%m-%d-%Hh%Mm%Ss-")
        + "-"
        + world.comment
        + "-"
        + world.config["loss"]
        + "-"
        + world.config["model"],
    )
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    w: SummaryWriter = SummaryWriter(save_dir)
    logger = CompleteLogger(root=save_dir)

print("User_num: ", dataset.n_users)
print("Item_num: ", dataset.m_items)
print("Interaction_num: ", dataset.traindataSize + dataset.testDataSize)
print("===========config================")
pprint(world.config)
print("Comment:", world.comment)
print("Test Topks:", world.topks)
print("===========end===================")

w.add_text("config", str(world.config), 0)

if world.config["resume_dir"] != "":
    Recmodel.load_state_dict(torch.load(world.config["resume_dir"]))
    print("load model from: ", world.config["resume_dir"])

best_recall, best_ndcg, best_hit, best_precision = (
    [0] * len(world.topks),
    [0] * len(world.topks),
    [0] * len(world.topks),
    [0] * len(world.topks),
)
patience = 0
start_total = time.time()

for epoch in range(world.TRAIN_epochs):
    start = time.time()
    if epoch % 5 == 0 and epoch != 0:
        cprint("[TEST]")
        test_res = procedure.Test(dataset, Recmodel, epoch, w, world.config["multicore"])
        test_recall, test_ndcg, test_hit, test_precision = (
            test_res["recall"],
            test_res["ndcg"],
            test_res["hitratio"],
            test_res["precision"],
        )
        if "NNI_PLATFORM" in os.environ:
            metric = {
                "default": test_ndcg[0],
                "recall": test_recall[0],
                "hit": test_hit[0],
                "precision": test_precision[0],
            }
            nni.report_intermediate_result(metric)

        if test_ndcg[0] > best_ndcg[0] + 0.0001:
            patience = 0
            if "NNI_PLATFORM" not in os.environ:
                torch.save(Recmodel.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            patience += 1
            print("Patience: {}/5".format(patience))
            if patience >= 5:
                print("Early stop!")
                break

        for i in range(len(world.topks)):
            best_recall[i], best_ndcg[i], best_hit[i], best_precision[i] = (
                max(best_recall[i], test_recall[i]),
                max(best_ndcg[i], test_ndcg[i]),
                max(best_hit[i], test_hit[i]),
                max(best_precision[i], test_precision[i]),
            )
        print(test_res)

    output_information = procedure.Train_original(dataset, Recmodel, loss_func, epoch, world.config, w=w)
    print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}")

print(
    "Best_hit: {}, Best_ndcg: {}, Best_precision: {}, Best_recall: {}".format(
        best_hit, best_ndcg, best_precision, best_recall
    )
)
if "NNI_PLATFORM" in os.environ:
    metric = {"default": best_ndcg[0], "recall": best_recall[0], "hit": best_hit[0], "precision": best_precision[0]}
    nni.report_final_result(metric)
    
print("Total time:{}".format(time.time() - start_total))
w.close()
