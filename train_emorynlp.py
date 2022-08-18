import numpy as np, argparse, time, random, math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import EmoryNLPRobertaCometDataset
from model import MaskedNLLLoss
from models.mucdn import MuCDN
from sklearn.metrics import f1_score, accuracy_score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_EmoryNLP_loaders(batch_size=32, num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset('train')
    validset = EmoryNLPRobertaCometDataset('valid')
    testset = EmoryNLPRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks  = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, umask, label,\
        dis_speaker_mask, dis_adj, inter_adj, intra_adj,\
        intra_relative_distance, inter_relative_distance = [data[i].cuda() for i in range(len(data))] if cuda else data

        log_prob = model(r1, r2, r3, r4, dis_speaker_mask, dis_adj, inter_adj, intra_adj, intra_relative_distance, inter_relative_distance)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, [avg_fscore]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='HD', help='hidden feature dim')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of output mlp layers.')
    parser.add_argument('--seed', type=int, default=2222, metavar='seed', help='seed')
    parser.add_argument('--norm', action='store_true', default=False, help='normalization strategy')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save best model')
    parser.add_argument('--pos', action='store_true', default=False, help='whether to use pos_embedding')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
 
    n_classes  = 7
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_h = args.hidden_dim

    global seed
    seed = args.seed
    seed_everything(seed)
    model = MuCDN(args, D_m, D_h, n_classes=n_classes)
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.size())
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print ('EmoryNLP Model.')
    
    if cuda:
        model.cuda()

    loss_function = MaskedNLLLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    train_loader, valid_loader, test_loader = get_EmoryNLP_loaders(batch_size=batch_size, num_workers=0)
    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None
    max_test_f1 = 0
    continue_not_increase = 0
    for e in range(n_epochs):
        increase_flag = False
        start_time = time.time()
        train_loss, train_acc, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        valid_loss, valid_acc, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)
        if test_fscore[0] > max_test_f1:
            max_test_f1 = test_fscore[0]
            increase_flag = True
            if args.save:
                torch.save(model.state_dict(), open('./emorynlp/best_model.pkl', 'wb'))
                print('Best Model Saved!')
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)
        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        print (x)

        if increase_flag == False:
            continue_not_increase += 1
            if continue_not_increase >= 20:
                print('early stop.')
                break
        else:
            continue_not_increase = 0

    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()
    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]
    score3 = test_fscores[0][np.argmax(test_fscores[0])]
    scores = [score1, score2]
    scores_val_loss = [score1]
    scores_val_f1 = [score2]
    scores_test_f1 = [score3]
    scores = [str(item) for item in scores]
    print ('Test Scores:')
    print('F1@Best Valid Loss: {}'.format(scores_val_loss))
    print('F1@Best Valid F1: {}'.format(scores_val_f1))
    print('F1@Best Test F1: {}'.format(scores_test_f1))
