"""
Design training and test process
"""
import time
from tools import world, utils
import numpy as np
import torch
import multiprocessing
from dataset import dataloader
from tools.world import cprint


CORES = multiprocessing.cpu_count() // 2  # 4090服务器上有256个核心



def Train(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    loss = loss_class

    start = time.time()

    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)


    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    iter_num = epoch * total_batch
    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        cri = loss.step(batch_users, batch_pos, batch_neg)
        w.add_scalar("Loss", cri, iter_num + batch_id)
        aver_loss += cri


    aver_loss = aver_loss / total_batch
        # w.add_scalar("Loss", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"





def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)  # 一个包含batch个元素的list，每个元素是一个np数组
    pre, recall, ndcg, hitratio = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        hitratio.append(utils.HitRatio(r))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
        "hitratio": np.array(hitratio),
    }


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config["test_u_batch_size"]  # 默认是100，多少个user一起test

    # dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # batch_users是一个tuple，里面是user的id
            allPos = dataset.getUserPosItems(batch_users)  # train positive items
            groundTrue = [testDict[u] for u in batch_users]  # test positive items
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 给出users和所有item的评分，返回二维tensor
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())  # 每个元素是一个二维tensor，表示每个user的topk item
            groundTrue_list.append(groundTrue)  # 每个元素是一个两层list，表示每个user的test positive items

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.testDataSize)

        for i in range(len(world.topks)):
            w.add_scalar(f"Test/Recall_{world.topks[i]}", results["recall"][i], epoch)
            w.add_scalar(f"Test/Precision_{world.topks[i]}", results["precision"][i], epoch)
            w.add_scalar(f"Test/NDCG_{world.topks[i]}", results["ndcg"][i], epoch)
            w.add_scalar(f"Test/HitRatio_{world.topks[i]}", results["hitratio"][i], epoch)
        if multicore == 1:
            pool.close()

        return results





#  ============== The AdvInfoNCE Trainer ===============
def TrainAdvInfoNCE(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, adv_training_flag = False):
    Recmodel = recommend_model
    Recmodel.train()
    loss = loss_class

    start = time.time()

    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)


    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    iter_num = epoch * total_batch

    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        cri = loss.step(batch_users, batch_pos, batch_neg,epoch, adv_training_flag = adv_training_flag)
        w.add_scalar("Loss", cri, iter_num + batch_id)
        aver_loss += cri




    aver_loss = aver_loss / total_batch
        # w.add_scalar("Loss", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"
