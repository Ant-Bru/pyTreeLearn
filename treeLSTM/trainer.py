from tqdm import tqdm
import torch as th
from .utils import get_new_logger
import copy
from .metrics import ValueMetric, TreeMetric

import time


def train(model, trainset):
    raise Exception('This function is not implemented yet!')


def train_and_validate(model, extract_batch_data, loss_function, optimizer, trainset, devset, device, metrics_class, batch_size=25, n_epochs=200, early_stopping_patience=20):
    logger = get_new_logger('train_and_validate')

    best_dev_metric = None
    trainloader = trainset.get_loader(batch_size, device, shuffle=True)
    devsize = devset.len() if hasattr(devset, 'len') else batch_size
    devloader = devset.get_loader(devsize, device)

    best_metrics = []
    best_epoch = -1
    best_model = None

    tr_forw_time_list = []
    tr_backw_time_list = []
    val_time_list = []

    for epoch in range(n_epochs):
        model.train()

        metrics = []
        for c in metrics_class:
            metrics.append(c())

        tr_forw_time = 0
        tr_backw_time = 0
        val_time = 0
        LOSS = 0


        with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ') as pbar:
            for step, batch in enumerate(trainloader):

                t = time.time()
                in_data, out_data, n_samples,  _ = extract_batch_data(batch)

                model_output = model(*in_data)
                loss = loss_function(model_output, out_data)
                LOSS += loss  #togliere

                tr_forw_time += (time.time() - t)

                t = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_backw_time += (time.time() - t)

                pbar.update(n_samples)

        #logger.info("Epoch Train Loss    " + str(LOSS.item()))
        #logger.info("Epoch Execution Time    " + str(tr_forw_time+tr_backw_time))

        # eval on dev set
        model.eval()
        with tqdm(total=len(devset), desc='Validate epoch ' + str(epoch) + ' on dev set: ') as pbar:
            for step, batch in enumerate(devloader):

                t = time.time()
                in_data, out_data, n_samples, graph = extract_batch_data(batch)
                with th.no_grad():
                    out = model(*in_data)

                # update all metrics
                for v in metrics:
                    if isinstance(v, ValueMetric):
                        v.update_metric(out, out_data)

                    if isinstance(v, TreeMetric):
                        v.update_metric(out, out_data, graph)
                val_time += (time.time() - t)

                pbar.update(n_samples)

        # print metrics
        s = "Dev Test: Epoch {:03d} | ".format(epoch)
        for v in metrics:
            v.finalise_metric()
            s += str(v) + " | "
        logger.info(s)

        if epoch == 0:
            best_dev_metric = metrics[0].get_value()
            best_epoch = epoch
            best_metrics = metrics
            best_model = copy.deepcopy(model)
        else:
            # the metrics in poisiton 0 is the one used to validate the model
            if metrics[0].is_better_than(best_dev_metric):
                best_dev_metric = metrics[0].get_value()
                best_epoch = epoch
                best_metrics = metrics
                best_model = copy.deepcopy(model)
                logger.debug('Epoch {:03d}: New optimum found'.format(epoch))
            else:
                # early stopping
                if best_epoch <= epoch - early_stopping_patience:
                    break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr'] * 0.99)

        tr_forw_time_list.append(tr_forw_time)
        tr_backw_time_list.append(tr_backw_time)
        val_time_list.append(val_time)

    return best_model, best_metrics, best_epoch, tr_forw_time_list[:best_epoch+1], tr_backw_time_list[:best_epoch+1], val_time_list[:best_epoch+1]


def test(model, extract_batch_data, testset, device, metrics_class, batch_size=25):
    logger = get_new_logger('test')
    testsize = testset.len() if hasattr(testset, 'len') else batch_size
    testloader = testset.get_loader(testsize, device)

    test_metrics = []
    for c in metrics_class:
        test_metrics.append(c())

    model.eval()
    with tqdm(total=len(testset), desc='Testing on test set: ') as pbar:
        for step, batch in enumerate(testloader):
            in_data, out_data, n_samples, graph = extract_batch_data(batch)
            with th.no_grad():
                out_model = model(*in_data)

            # update all metrics
            for v in test_metrics:
                if isinstance(v, ValueMetric):
                    v.update_metric(out_model, out_data)

                if isinstance(v, TreeMetric):
                    v.update_metric(out_model, out_data, graph)

            pbar.update(n_samples)

    # print metrics
    s = "Test: "
    for v in test_metrics:
        v.finalise_metric()
        s += str(v) + " | "
    logger.info(s)

    return test_metrics



#TODO: fix deepcopy problem (changing splits management)
# def k_fold_validation(model, extract_batch_data, loss_function, optimizer, dataset, splits, device, metrics_class, batch_size=25, n_epochs=200, early_stopping_patience=20):
#     logger = get_new_logger('k-fold')
#     fold_models = []
#     fold_metrics = []
#     trainset = copy.deepcopy(dataset)
#     devset = copy.deepcopy(dataset)
#     k = len(splits)
#     for i in range(k):
#         model = copy.deepcopy(model) #initialized model everytime
#         devset.data = splits[i]
#         tset = splits[:i]+splits[i+1:]
#         tset = [item for list in tset for item in list]
#         trainset.data = tset
#         logger.info('FOLD ' + str(i+1) + "-------------------")
#         best_model, best_dev_metrics, *others = train_and_validate(model, extract_batch_data, loss_function, optimizer, trainset, devset, device, metrics_class, batch_size, n_epochs, early_stopping_patience)
#         fold_models.append(best_model)
#         fold_metrics.append([x.get_value() for x in best_dev_metrics]) #take metrics values
#     avg_metrics = torch.mean(torch.Tensor(fold_metrics), 0) #average metrics values
#     logger.info('Average fold metrics: ' + str(avg_metrics))
#     return fold_models, avg_metrics