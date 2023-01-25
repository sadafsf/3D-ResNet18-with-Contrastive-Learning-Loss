import argparse
import ast
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset 
import torchvision

from torch.utils.tensorboard import SummaryWriter
import spc 
# import spc2
from model import generate_model
from train_lowlevel import train_contrastive, train_crossentropy_no_proj, train_crossentropy
from utils import  generate_dataloader
import datasets
import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_mode",
        default="contrastive_no_proj",
        choices=["contrastive_no_proj", 'contrastive_proj'],
        help="Type of training use either a two steps contrastive then cross-entropy or \
                         just cross-entropy",
    )
    parser.add_argument(
        "--batch_size",
        default=150,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # model part    
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument("--temperature", default=0.1, type=float, help="Constant for loss no thorough ")
    parser.add_argument("--num_classes", default=4, type=int, help=" number of classes ")

    parser.add_argument("--n_epochs_contrastive", default=100, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=100, type=int)

#  learning rate part
    parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    parser.add_argument("--lr_cross_entropy", default=5e-2, type=float)

    parser.add_argument("--cosine", default=True, type=bool, help="Check this to use cosine annealing instead of ")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")
    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=100,
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    args = parser.parse_args()

    return args




args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# Some args stuff to do
torch.manual_seed(args.manual_seed)
#if args.use_cuda:
 #   torch.cuda.manual_seed(args.manual_seed)
if args.nesterov:
    dampening = 0
else:
    dampening = args.dampening

train=datasets.VideoDataset('/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/train.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

val=datasets.VideoDataset( '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/validation.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/ValidationFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

# combine val+train dataset
data=ConcatDataset([train,val])

# train and test loader - spatial downsampling and trasfromation - temporal downsampling
train_loader, weight= generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/trainval.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), pre_process=data)
dataset_sizes =train_loader.dataset
print(dataset_sizes, flush=True) 
# print(dataset_sizes) 


test_loader, _ = generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/test.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TestFrames'), pre_process=None)

dataset_sizes =test_loader.dataset
print(dataset_sizes, flush=True)
# print(dataset_sizes)   



# ===============generate new model or pre-trained model===============
model = generate_model(args)
model = model.to(args.device)
optimizer = torch.optim.SGD(model.parameters() , lr=args.lr_contrastive, momentum=args.momentum,
                            dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)

cudnn.benchmark = True
if not os.path.isdir("logs"):
    os.makedirs("logs")

writer = SummaryWriter("logs")

# criterion =spc2.SupervisedContrastiveLoss(temperature=args.temperature)
criterion =spc.SupervisedContrastiveLoss(temperature=args.temperature)
criterion.to(args.device)


train_contrastive(train_loader, model, criterion, optimizer, writer, args)


# Load checkpoint.
print("==> Resuming from checkpoint..")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
model.load_state_dict(checkpoint["net"])

if args.training_mode == "contrastive_no_proj":
    model.freeze_projection()
    optimizer = torch.optim.SGD(model.parameters() , lr=args.lr_cross_entropy, momentum=args.momentum,
                                dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # add the weight to it 
    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    args.best_acc = 0.0
    train_crossentropy_no_proj(train_loader,test_loader,model,criterion, optimizer, writer, args)

else:
    model.freeze()
    optimizer = torch.optim.SGD(model.parameters() , lr=args.lr_cross_entropy, momentum=args.momentum,
                                dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # add the weight to it 
    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    args.best_acc = 0.0
    train_crossentropy(train_loader,test_loader,model,criterion, optimizer, writer, args)




