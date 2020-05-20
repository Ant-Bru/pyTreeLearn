import os, sys
import argparse
import numpy as np
import torch as th
import torch.nn.init as INIT
import torch.optim as optim

os.chdir(("../../"))
sys.path.append(os.getcwd())

print(os.getcwd())

from treeLSTM.utils import set_main_logger_settings
from treeLSTM.trainer import train_and_validate, test
from treeLSTM.metrics import Accuracy, RootAccuracy
from tests.INEX.utils import create_inex_model, load_inex_dataset, inex_loss_function, inex_extract_batch_data

def main(args):

    # create log_dir
    log_dir = os.path.join(args.save, args.expname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # initiliase the main logger
    logger = set_main_logger_settings(log_dir, 'main_INEX_'+str(args.h_size)+"_"+args.cell_type+"_"+str(args.pos_stationarity))

    # set the seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    # set the device
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(10)


    # load the data
    trainset, devset, testset = load_inex_dataset(args.task, args.all)

    # create the model
    model = create_inex_model(trainset.input_dim,
                              args.h_size,
                              trainset.arity,
                              trainset.num_classes,
                              pretrained_emb=trainset.pretrained_emb,
                              num_vocabs=trainset.num_vocabs,
                              cell_type=args.cell_type,
                              rank=None,
                              pos_stationarity=args.pos_stationarity,
                              all=args.all).to(device)

    logger.info(str(model))

    # optimizer
    params = [x for x in list(model.parameters()) if x.requires_grad]

    for p in params:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adam([{'params': params}])

    # train and validate
    best_model, best_dev_metrics, *others = train_and_validate(model, inex_extract_batch_data, inex_loss_function, optimizer, trainset, devset, device,
                                                      metrics_class=[Accuracy if not args.all else RootAccuracy],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs, early_stopping_patience=args.early_stopping)

    test(best_model, inex_extract_batch_data,  testset, device,
         metrics_class=[Accuracy if not args.all else RootAccuracy],
         batch_size=args.batch_size)


if __name__ == '__main__':
    #TODO: expanme anch savedit can be decided programmatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='GRU')
    parser.add_argument('--task', default='06')
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--all', type=bool, default=False)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='tests/INEX/checkpoints/')
    parser.add_argument('--expname', default='test_INEX_')
    args = parser.parse_args()

    if args.task not in ['05', '06']:
      raise ValueError('Task type ( '+args.task+' ) not supported, select 05 or 06')

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    args.expname += args.task + "_" + dt_string + "_seed" + str(args.seed) + "_batch" + str(args.batch_size) + "_all" + str(args.all)

    main(args)