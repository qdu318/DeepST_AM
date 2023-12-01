import os
import json
import argparse
import torch
import torch.nn as nn

from utils.utils import get_normalized_adj
from utils.data_load import Data_load
from process.train import Train
from process.evaluate import Evaluate
import logger

elogger = logger.Logger('run_log_deepst')

from model.DeepST_AM.deepst_am import DeepST_AM
from model.DeepST_AM.model_config import get_backbones


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = json.load(open('./config.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--adj_filename', type=str, default='./data/TRI/W_1.csv')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=4)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--out_channels', type=int, default=64)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--N', type=int, default=29)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--features', type=int, default=5)
parser.add_argument('--time_slice', type=list, default=[1, 2, 3])
parser.add_argument('--day', type=list, default=[1, 3, 7])
args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(7)
    W_nodes, data_set = Data_load(config, args.timesteps_input, args.timesteps_output, args.num_of_weeks, args.num_of_days, args.num_of_hours)
    all_backones = get_backbones(args.K, args.num_of_weeks, args.num_of_days, args.num_of_hours, args.N, args.adj_filename)
    num_of_features = 5
    num_of_weeks = args.num_of_weeks
    num_of_days = args.num_of_days
    num_of_hours = args.num_of_hours
    num_of_predict = args.timesteps_output
    num_of_vectices = args.N
    points_per_hour = 12
    num_of_timesteps = [[points_per_hour*num_of_weeks, points_per_hour],
                        [points_per_hour*num_of_days, points_per_hour],
                        [points_per_hour*num_of_hours, points_per_hour]]
    device = torch.device("cuda:0")
    model = DeepST_AM(num_of_predict, all_backones, num_of_vectices, num_of_features, num_of_timesteps, device)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        train_loss = Train(
                        model=model,
                        optimizer=optimizer,
                        loss_meathod=L2,
                        W_nodes=W_nodes,
                        data_set=data_set,
                        batch_size=args.batch_size
                    )
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_loss, eval_index = Evaluate(
                                        epoch=epoch,
                                        model=model,
                                        loss_meathod=L2,
                                        W_nodes=W_nodes,
                                        time_slice=args.time_slice,
                                        data_set=data_set
                                    )
        print("--------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(train_loss))
        for i in range(len(args.time_slice)):
            print("day:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, MAPE:{}"
                  .format(args.day[i], eval_loss[-(len(args.time_slice) - i)], eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)], eval_index['MAPE'][-(len(args.time_slice) - i)],))
            elogger.log("day:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, MAPE:{}"
                  .format(args.day[i], eval_loss[-(len(args.time_slice) - i)], eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)], eval_index['MAPE'][-(len(args.time_slice) - i)],))
        print("---------------------------------------------------------------------------------------------------")

        if not os.path.exists(args.weight_file):
            os.makedirs(args.weight_file)

        if (epoch % 50 == 0) & (epoch != 0):
            torch.save(model, args.weight_file + 'model_' + str(epoch))

