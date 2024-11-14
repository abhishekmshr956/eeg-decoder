import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modified_cross_entropy(beta=0.5):
    '''
    this is a wrapper to allow specification of the
    regularization parameter beta
    '''
    def modified_cross_entropy_loss(y_pred, y_true):
        '''
        Special cross entropy loss that adds penalty to incorrect 
        confident predictions

        beta controls how much incorrect confident predictions get 
        penalized

        y_true is (N,)
        y_pred is (N, C)
        where N is the batch size and C are the number of classes
        '''
        ids = y_true.view(-1,1).long()
        numerator = torch.exp(y_pred.gather(-1, ids))        # (N,)
        denominator = torch.sum(torch.exp(y_pred), axis=-1).view(-1,1)  # (N,)

        # mean reduction is used by default for PyTorch Cross-Entropy Loss
        cross_entropy = torch.log(numerator / denominator)

        # confidence_penalty
        # indicates what samples are misclassified
        indicator = torch.argmax(y_pred, axis=-1) != y_true
        logits = torch.max(torch.exp(y_pred), axis=-1)[0]
        penalty = indicator * logits
        penalty = torch.sum(penalty) / torch.sum(indicator)
        #pdb.set_trace()

        loss = -torch.mean(cross_entropy) + beta*penalty
        return loss
    return modified_cross_entropy_loss


def load_loss_criterion(loss_name='CEL'):
    'Used to translate loss names into pyTorch criterion'
    if loss_name == "CEL":
        # use cross entropy loss
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "FL":
        # use focal loss
        pass
    elif loss_name == "smoothed":
        # use label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
    elif loss_name == "modified":
        # use modified cross entropy loss
        criterion = modified_cross_entropy()
    elif loss_name[:8] == "modified":
        criterion = modified_cross_entropy(beta=float(loss_name[8:]))
    elif loss_name == "NLL":
        criterion = nn.NLLLoss()
    return criterion

def train(model, train_ds, val_ds, config, writer):
    
    min_val_loss = np.Inf   # will be used for early stopping
    epochs_no_improve = 0   # will be used for early stopping
    best_model_index = 0 # will be used to find train and val acc of best model
    train_losses_epochs = []
    val_losses_epochs = []

    #Get loss criterion
    criterion = load_loss_criterion(config['loss_func'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True)  # default patience is 10 and factor is 0.1

    train_accuracy = []
    val_accuracy = []
    pred_labels, true_labels = [], []
    for epoch in range(config['max_epochs']):
        train_losses = []
        num_correct = 0
        num_samples = 0
        model.train()
        for i, (inputs, labels) in enumerate(tqdm.tqdm(train_ds)):

            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            try:
                loss = criterion(outputs, labels)
            except:
                print(outputs)
                print(labels)
                print()
            writer.add_scalar('Training Loss', loss.item(), epoch)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            _, softmax_indices = outputs.max(1)
            num_correct += (softmax_indices == labels).sum()
            num_samples += softmax_indices.size(0)

        accuracy = 100 * num_correct / num_samples
        print("epoch #", epoch, " training accuracy (%):", accuracy.item())
        train_accuracy.append(accuracy.item())
        train_losses_epochs.append(sum(train_losses) / len(train_losses))

        # test model on validation data
        val_loss, correct_list, e_pred_labels, e_true_labels = test(model,
                                      val_ds,
                                      criterion=criterion)
        pred_labels.append(e_pred_labels)
        true_labels.append(e_true_labels)
        
        writer.add_scalar('Validation Loss', val_loss, epoch)
        val_losses_epochs.append(val_loss.item() / len(val_ds))

        num_correct = sum(correct_list)
        num_samples = len(correct_list)
        accuracy = 100 * num_correct / num_samples
        print("epoch #", epoch, " validation accuracy (%):", accuracy)
        scheduler.step(accuracy)
        val_accuracy.append(accuracy)

        writer.add_scalar('validation loss', val_loss, epoch * len(train_ds))

        # check for early stopping
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_index = epoch
        else:
            epochs_no_improve += 1
        if epoch > 5 and epochs_no_improve == 10:
            print("Training terminated early!")
            break
        
    return (best_model, 
           {'acc' : train_accuracy, 'val_acc' : val_accuracy},
           best_model_index,
           train_losses_epochs,
           val_losses_epochs,
           pred_labels,
           true_labels)


def test(model, val_ds, criterion):
    '''This function scores a model against test data.'''
    model.eval()
    pred_labels, true_labels = [], []
    
    with torch.no_grad():
        correct_list = []
        val_loss = 0
        for x, y in val_ds:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            val_loss += criterion(scores, y)

            _, softmax_indices = scores.max(1)
            correct_list = correct_list + (softmax_indices == y).tolist()

            pred_labels.extend(softmax_indices)
            true_labels.extend(y)

    pred_labels = [pred_label.item() for pred_label in pred_labels]
    true_labels = [true_label.item() for true_label in true_labels]

    return val_loss, correct_list, pred_labels, true_labels
