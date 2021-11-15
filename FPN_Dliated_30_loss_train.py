

from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats

import datetime
import pandas as pd
import sys
from tqdm import tqdm
from backbones.CNN3D import *
from backbones.data_loader import *








if __name__ == "__main__":
    parser = ArgumentParser(description='"my model')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')


    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA)')


    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0)')


    args = parser.parse_args()

    args.decay_interval = int(args.epochs/100)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True



    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    num_for_val = 20
    epoch_start=0

    features_dir = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/FPN_D_30_origin_k_1200/" 
    print("训练数据目录：",features_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir2 = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/FPN_D_30_origin_k_1200/" 
    
    
    val_list = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/KoNViD-1k_val.txt"
    
    
 
    
    with open(val_list,"r") as f:
        all_data = f.readlines()
    val_data = []
    for data in all_data:
        val = data.split("_")[0]
        val_data.append(val)
    
    
    # 训练数据集合
    videos_pic1 = []
    result = os.listdir(features_dir)        
    
    
    total_videos = len(result)
    #print("总数据:",total_videos)
    
    
    width = height=0
    max_len = 8000
    train_list,val_list,test_list =[],[],[]
    best_acc =0
    
    

    
    for i in range(total_videos):
        tmp = result[i].split(".")[0]
        #rint(tmp)
        
        if tmp in val_data:
            val_list.append(result[i])
            continue

        train_list.append(result[i])
            
    
        
        
    print(train_list[0],val_list[0])   
    print("split data:train: {}, test: {}, val: {}".format(len(train_list),len(test_list),len(val_list)))
   # sys.exit()

    #print(train_list[0])

    
#  #   print(len(train_index))
#     train_list = train_list[0:100]
#     val_list =  val_list[0:10]
    
    
    train_dataset = FPN_Dliated_VQADataset(features_dir,train_list, max_len=240,scale = 4.05)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=1)
    
    print("load train data success!")
#     for i, (features, length, label,name) in enumerate(train_loader):
#         print(features.shape,length.shape)
#         break
        
    

    
    val_dataset = FPN_Dliated_VQADataset(features_dir2, val_list, max_len=240,scale = 4.05)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,)

    print("load val data success!")
    
    
    model = FPN_Dliated_LOSS_ResNet3D(Bottleneck, [3, 4, 6, 3]).to(device)  #
    
    

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models'
    
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

#     pretrained =0
#     if pretrained:
#         new_state_dict = {}
#         path = "./models/3D_VSFA1_acc:0.43177764565992865.pth"
#         checkpoint = torch.load(path)
        
#      #   model.load_state_dict(checkpoint)
#         for k, v in checkpoint["model"].items():
#             name =k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#             new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
#         model.load_state_dict(new_state_dict)
#         epoch_start = checkpoint["epoch"]
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("pre_modle:{}".format(path))
        
     
    
    Not_well_video ={}
    
    
    criterion = nn.L1Loss()  # MSELoss loss
   
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_PLCC = 0
    for epoch in range(epoch_start,args.epochs):
        # Train
        model.train()
        L = 0
        right_num = 0
        total_num = 0
        print("training epch:{},total epch:{}".format(epoch,args.epochs))
        for i, (features,D_features, length, label,name) in enumerate(tqdm(train_loader)):
            
#             print("features,",features.shape,D_features.shape)
#             print("length",length)

            features = features.to(device).float()
            D_features = D_features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            
            outputs_all = model(features,D_features)
            
            outputs = outputs_all[0]
          #  print(outputs.shape)
            outputs= outputs.squeeze(1)
#             print(outputs.shape,label.shape)
#           #  print(outputs,label)
#             sys.exit()
            loss = criterion(outputs_all[0].squeeze(1), label) \
               + 0.5 * (criterion(outputs_all[1].squeeze(1), label) + 0.3 * criterion(outputs_all[2].squeeze(1), label) + 0.2 * criterion(outputs_all[3].squeeze(1), label))
            loss.backward()
            optimizer.step()
            L = L + loss.item()
             

        train_loss = L / (i + 1)

        print("train_loss:",train_loss)
 
        if epoch % num_for_val ==0:
            print("start valling")
            model.eval()
            # Val
            y_pred = np.zeros(len(val_list))                                                     
            y_val = np.zeros(len(val_list))
            L = 0
            with torch.no_grad():
                badcase = {}
          

               # y_pred = np.zeros(len(result))
             #   y_test = np.zeros(len(result))
                L = 0
                for i, (features,D_features, length, label,name) in enumerate(tqdm(val_loader)):
      
                    y_val[i] = 4.05*label.item()    
                    features = features.to(device).float()
                    D_features = D_features.to(device).float()
                    label = label.to(device).float()
                    outputs_all = model(features,D_features)
            
                    outputs = outputs_all[0]
                    
                    
                    y_pred[i] = 4.05*outputs.item()
      
                    #y_pred[i] = 1 * outputs.item()
                    loss = criterion(outputs, label)
                    #print(outputs,label)
                    L = L + loss.item()

            val_loss = L / (i + 1)
            val_PLCC = stats.pearsonr(y_pred, y_val)[0]
            val_SROCC = stats.spearmanr(y_pred, y_val)[0]
            val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
            val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

            #print("Badcase",badcase)
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if val_PLCC >best_PLCC:


                print("save model at epch")

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,os.path.join(trained_model_file,"New_loss222_{}_plcc:{}.pth".format(epoch+1,val_PLCC)))
                print("Epoch {} model saved!".format(epoch + 1))
                best_PLCC = val_PLCC
            elif epoch % 20 ==0:
                print("save model at epch")

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,os.path.join(trained_model_file,"New_loss222_{}_epoch:{}.pth".format(epoch+1,epoch+1)))
                print("Epoch {} model saved!".format(epoch + 1))
            


#         print(badcase)
          
