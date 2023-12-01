# encoding utf-8
import os.path

import torch
from utils.utils import RMSE, MAE, MAPE
from utils.utils import Un_Z_Score
import numpy as np


def Cal_eval_index(epoch, pred, loss_meathod, val_target, time_slice, mean, std):
    val_index = {}
    val_index['MAE'] = []
    val_index['RMSE'] = []
    val_index['MAPE'] = []
    val_loss = []

    if not os.path.exists("./results/DeepST_AM/"):
        os.makedirs("./results/DeepST_AM/")

    if torch.cuda.is_available():
        mean = torch.tensor(mean).cuda()
        std = torch.tensor(std).cuda()

    for item in time_slice:
        pred_index = pred[:, :, item - 1]
        val_target_index = val_target[:, :, item - 1]
        pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean, std)

        loss = loss_meathod(pred_index, val_target_index)
        val_loss.append(loss)

        if ((epoch % 20 == 0) or (epoch % 50 == 0) or epoch == 199) & (epoch != 0):
            np.savetxt("./results/DeepST_AM/pred_result_" + str(epoch) + "_1.csv", pred_index.cpu(), delimiter=',')
            np.savetxt("./results/DeepST_AM/true_result_" + str(epoch) + "_1.csv", val_target_index.cpu(), delimiter=',')

        mae = MAE(val_target_index, pred_index)
        val_index['MAE'].append(mae)

        rmse = RMSE(val_target_index, pred_index)
        val_index['RMSE'].append(rmse)

        mape = MAPE(val_target_index, pred_index)
        val_index['MAPE'].append(mape)

    return val_loss, val_index


def Evaluate(epoch, model, loss_meathod, W_nodes, time_slice, data_set):
    model.eval()
    eval_week, eval_day, eval_hour, eval_target = data_set['test']['week'], data_set['test']['day'], \
                                              data_set['test']['recent'], data_set['test']['target']

    if torch.cuda.is_available():
        eval_week = torch.from_numpy(eval_week).cuda()
        eval_day = torch.from_numpy(eval_day).cuda()
        eval_hour = torch.from_numpy(eval_hour).cuda()
        eval_target = torch.from_numpy(eval_target).cuda()
    pred = model([eval_week, eval_day, eval_hour])

    eval_loss, eval_index = Cal_eval_index(epoch, pred, loss_meathod, eval_target, time_slice, data_set['X_mean'], data_set['X_std'])
    return eval_loss, eval_index