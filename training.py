import time
from datetime import timedelta
import torch
import numpy as np
from PIL import Image
from numba.tests.test_typedlist import to_tl
from sklearn.metrics import confusion_matrix


def cpc_train(model, train_loader, timesteps_in, timesteps_out, optimizer, epoch, batch_size):
    start_time = time.time()
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0
    hidden = None
    elapsed_times = []
    for batch_idx, data in enumerate(train_loader):
        batch_time = time.time()
        data = data.float().cuda()
        optimizer.zero_grad()
        if hidden is None or batch_size != data.shape[0]: #TODO: every time if hidden hasnt been initialized or
            print(data.shape)
            hidden = model.init_hidden(data.shape[0], use_gpu=True)
        hidden.detach_()
        hidden = hidden.detach()
        count += 1
        acc, loss, hidden = model(data, timesteps_in, timesteps_out, hidden)
        total_loss += loss
        total_acc += acc
        loss.backward()
        optimizer.step()
        elapsed_times.append(time.time()-batch_time)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} \tAccuracy: {:.4f}\tLoss: {:.6f}\tElapsed time: {}'.format(
                epoch, acc, loss.item(), str(timedelta(seconds=elapsed_times[-1]))))
    total_loss /= count
    total_acc /= count

    print('===> Trainings set: Average loss: {:.4f}\tAccuracy: {:.4f}\tElapsed time (average): {}\tElapsed time (total): {}\n'.format(
        total_loss.item(), total_acc.item(), str(timedelta(seconds=np.average(elapsed_times))), str(timedelta(seconds=time.time()-start_time))))
    return total_acc, total_loss

def cpc_validation(model, data_loader, timesteps_in, timesteps_out, batch_size):
    model.eval()
    total_loss = 0
    total_acc  = 0
    count = 0
    hidden = None
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            data = data.float().cuda() # add channel dimension
            if True: #hidden is None or batch_size != data.shape[0]:
                hidden = model.init_hidden(len(data), use_gpu=True)
            acc, loss, hidden = model(data, timesteps_in, timesteps_out, hidden)
            count += 1
            total_loss += loss
            total_acc  += acc

    total_loss /= count # average loss
    total_acc  /= count # average acc

    print('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss.item(), total_acc))

    return total_acc, total_loss

def down_train(downstream_model, train_loader, timesteps_in, timesteps_out, optimizer, epoch, batch_size):
    #TODO: THIS METHOD
    downstream_model.train()
    total_loss = torch.tensor(0.0).cuda()
    total_acc = torch.tensor(0.0).cuda()
    count = 0
    cpc_hidden = None
    cpc_latents = []
    cpc_contexts = []
    confusion_pred, confusion_y = [], []
    for batch_idx, data_and_labels in enumerate(train_loader): #TODO: Custom collate fn
        data, patient_finished, labels = data_and_labels # add channel dimension
        #print('here', patient_finished, labels)

        data = data.float().cuda()
        labels = labels.float().squeeze(1).cuda() #do not squeeze batch dim (0)
        if not cpc_hidden is None:
            cpc_hidden = downstream_model.init_hidden(len(data), use_gpu=True)
        optimizer.zero_grad()
        cpc_latent, cpc_context, cpc_hidden = downstream_model(data, None, None, cpc_hidden, y=None, finished=False)
        cpc_latents.append(cpc_latent)
        cpc_contexts.append(cpc_context)
        if patient_finished.any(): #If anyone is finished? TODO: Better: give finished vector to downstream
            acc, loss, cpc_hidden, confuse = downstream_model(None, cpc_latents, cpc_contexts, cpc_hidden, y=labels, finished=True)
            cpc_hidden = downstream_model.init_hidden(len(data), use_gpu=True)

            del cpc_latents[:]
            del cpc_contexts[:]
            total_loss += loss
            total_acc += acc
            loss.backward()
            confusion_pred += [*confuse[0].cpu().numpy().flatten()]
            confusion_y += [*confuse[1].cpu().numpy().flatten()]

            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} \tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, acc.item(), loss.item()))
            count += 1
    total_loss /= count
    total_acc /= count
    #cm = confusion_matrix(confusion_y, confusion_pred)
    #plot_confusion_matrix(cm, 'train', ['CD', 'HYP', 'MI', 'NORM', 'STTC'])
    print('===> Trainings set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))
    return total_acc, total_loss

def down_validation(downstream_model, data_loader, timesteps_in, timesteps_out, batch_size):
    total_loss = torch.tensor(0.0).cuda()
    total_acc = torch.tensor(0.0).cuda()
    all_accs = []
    count = 0
    cpc_hidden = None
    cpc_latents = []
    cpc_contexts = []
    confusion_pred, confusion_y = [], []
    with torch.no_grad():
        for batch_idx, data_and_labels in enumerate(data_loader):  # TODO: Custom collate fn
            data, patient_finished, labels = data_and_labels  # add channel dimension
            # print('here', patient_finished, labels)
            data = data.float().cuda()
            labels = labels.float().squeeze(1).cuda()
            if not cpc_hidden is None:
                cpc_hidden = downstream_model.init_hidden(len(data), use_gpu=True)
            cpc_latent, cpc_context, cpc_hidden = downstream_model(data, None, None, cpc_hidden, y=None, finished=False)
            cpc_latents.append(cpc_latent)
            cpc_contexts.append(cpc_context)
            if patient_finished.any():  # If anyone is finished? TODO: Better: give finished vector to downstream

                accs, loss, cpc_hidden, confuse = downstream_model(None, cpc_latents, cpc_contexts, cpc_hidden, y=labels,
                                                         finished=True)
                if type(accs) == list:
                    acc = accs[0]
                    all_accs.append(accs)
                else:
                    acc = accs
                cpc_hidden = downstream_model.init_hidden(len(data), use_gpu=True)
                del cpc_latents[:]
                del cpc_contexts[:]
                total_loss += loss
                total_acc += acc

                #confusion_pred += [*confuse[0].cpu().numpy().flatten()]
                #confusion_y += [*confuse[1].cpu().numpy().flatten()]
                count += 1
        total_loss /= count
        total_acc /= count
        #cm = confusion_matrix(confusion_y, confusion_pred)
        #plot_confusion_matrix(cm, 'validation', ['CD', 'HYP', 'MI', 'NORM', 'STTC'])
        print('===> Trainings set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
            total_loss, total_acc))
        return total_acc, total_loss

def baseline_train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = torch.tensor(0.0).cuda()
    total_acc = torch.tensor(0.0).cuda()
    all_accs = []
    count = 0
    for batch_idx, data_and_labels in enumerate(train_loader):
        data, labels = data_and_labels # add channel dimension
        data = data.float().cuda()
        labels = labels.float().squeeze(1).cuda() #do not squeeze batch dim (0)
        optimizer.zero_grad()
        accs, loss = model(data, y=labels)
        if type(accs) == list:
            acc = accs[0]
            all_accs.append(accs)
        else:
            acc = accs
        total_loss += loss
        total_acc += acc
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}\tAccuracies: '.format(
                epoch, loss.item()), *map("{:.4f}".format, map(torch.Tensor.item, accs)))
        count += 1
    total_loss /= count
    total_acc /= count
    total_accs = []
    for ac in zip(*all_accs):
        total_accs += [(torch.sum(torch.stack(ac))/len(ac)).item()]
    print('===> Trainings set: Average loss: {:.4f}\tAccuracies: '.format(
        total_loss), *map("{:.4f}".format, total_accs))
    return total_acc, total_loss

def baseline_validation(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = torch.tensor(0.0).cuda()
    total_acc = torch.tensor(0.0).cuda()
    total_accs = []
    count = 0
    with torch.no_grad():
        for batch_idx, data_and_labels in enumerate(train_loader):
            data, labels = data_and_labels # add channel dimension
            data = data.float().cuda()
            labels = labels.float().squeeze(1).cuda() #do not squeeze batch dim (0)
            accs, loss = model(data, y=labels)
            if type(accs) == list:
                acc = accs[0]
                total_accs.append(accs)
            else:
                acc = accs
            total_loss += loss
            total_acc += acc
            count += 1
        total_loss /= count
        total_acc /= count
        all_accs = []
        for ac in zip(*total_accs):
            all_accs += [(torch.sum(torch.stack(ac))/len(ac)).item()]
        print('===> Valditation set: Average loss: {:.4f}\tAccuracies: '.format(
            total_loss), *map("{:.4f}".format, all_accs))
        return total_acc, total_loss

def plot_confusion_matrix(cm, mode,
                          target_names=['angina', 'bundle branch block', 'cardiomyopathy', 'dysrhythmia',
       'healthy control', 'hypertrophy', 'miscellaneous',
       'myocardial infarction', 'myocarditis', 'valvular heart disease'],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True): #https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('images/'+mode+str(int(time.time()/1000))+"confusion.png")
    plt.close(fig)