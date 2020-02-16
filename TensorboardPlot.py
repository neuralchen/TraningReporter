#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: TensorboardPlot.py
# Created Date: Wednesday November 6th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 16th February 2020 2:00:20 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

# python TensorboardPlot.py --whichkey ACC --whichgroup nosc --smooth True
from sshupload import fileUploaderClass
import argparse
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import scipy.signal as signal

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser(description='Logs')
parser.add_argument('--whichkey', type=str,default="acc", help='which key',choices=['PSNR','psnr','ACC','acc','REC','rec','GATT','gatt','ggan','GGAN','gt','GT','datt','dgan'])
parser.add_argument('--whichgroup', type=str,default="diffsc", help='which group')
parser.add_argument('--smooth', type=float, default=0.0)
args = parser.parse_args()

def smooth(y, weight=0.9):
    
    last = y[0]
    smoothed = []
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    y = np.array(smoothed)
    return y

def main():
    keydict     = {
        'acc':'Metric/meanACC',
        'psnr':'Metric/meanPSNR',
        'rec':'G/x_loss_rec',
        'gatt':'G/x_fake_loss_att',
        'ggan':'G/x_fake_loss_gan',
        'gt':'G/g_loss',
        'dgan':'D/d_loss_gan',
        'datt':'D/d_loss_att'
    }
    whichkey    = args.whichkey
    whichkey    = whichkey.upper()
    smoothratio = args.smooth
    serverList = {
        "lyh":{
            "ip":"172.27.232.105",
            "user":"lyh123",
            "passwd":"123456",
            "basePath":"/home/lyh123/liyunhao/PatchFaceGEN/TrainingLogs/"
        },
        "8card":{
            "ip":"192.168.101.34",
            "user":"user300",
            "passwd":"300300300",
            "basePath":"/home/user300/LNY/PatchFaceGEN/TrainingLogs/"
        },
        "4card":{
            "ip":"192.168.101.57",
            "user":"gdp",
            "passwd":"glass123456",
            "basePath":"/home/gdp/CXH/PatchBasedFaceAttributesEditing/TrainingLogs/"
        }
    }
    groupsLogs  = {
        "diffsc":[
            {
                "name":"concat_puresc1_",
                "server":"8card",
                "logName":"concat_puresc1",
                "path":"",
                "show":False
            },
            {
                "name":"concat_puresc2_",
                "server":"8card",
                "logName":"concat_puresc2",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"finallstudescriptor1_",
                "server":"8card",
                "logName":"finallstudescriptor1",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu1_",
                "server":"4card",
                "logName":"final_lstu1",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu2_",
                "server":"8card",
                "logName":"final_lstu2",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu3_",
                "server":"8card",
                "logName":"final_lstu3",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstu3_outputlayer_",
                "server":"8card",
                "logName":"final_lstu3_outputlayer",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu4_",
                "server":"8card",
                "logName":"final_lstu4",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu6_",
                "server":"lyh",
                "logName":"final_lstu6",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu7_",
                "server":"lyh",
                "logName":"final_lstu7",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstu8_",
                "server":"lyh",
                "logName":"final_lstu8",
                "path":"",
                "show":False
            },
            {
                "name":"final_lstubc1_",
                "server":"4card",
                "logName":"final_lstubc1",
                "path":"",
                "show":False
            },
            {
                "name":"concat_lstu1_",
                "server":"8card",
                "logName":"concat_lstu1",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstuwithc1_",
                "server":"4card",
                "logName":"final_lstuwithc1",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstuwithc2_",
                "server":"4card",
                "logName":"final_lstuwithc2",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstuwithc3_",
                "server":"8card",
                "logName":"final_lstuwithc3",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstuwithc4_",
                "server":"lyh",
                "logName":"final_lstuwithc4",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"final_lstuwithc5_",
                "server":"8card",
                "logName":"final_lstuwithc5",
                "path":"",
                "show":False
            }
            ,{
                "name":"lstunoreset_1_",
                "server":"8card",
                "logName":"lstunoreset_1",
                "path":"",
                "show":False
            },{
                "name":"lstunoreset2_",
                "server":"4card",
                "logName":"lstunoreset2",
                "path":"",
                "show":False
            },
            {
                "name":"lsturestchannelgate1_",
                "server":"8card",
                "logName":"lsturestchannelgate1",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"lsturestspatialgate1_",
                "server":"4card",
                "logName":"lsturestspatialgate1",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"lsturestspatialgatea2_",
                "server":"lyh",
                "logName":"lsturestspatialgatea2",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"lsturestspatialgate3_",
                "server":"8card",
                "logName":"lsturestspatialgate3",
                "path":"",
                "show":False
            }
            ,{
                "name":"spatialgate1_",
                "server":"lyh",
                "logName":"spatialgate1",
                "path":"",
                "show":False
            }
            ,{
                "name":"lsturestgate1_",
                "server":"8card",
                "logName":"lsturestgate1",
                "path":"",
                "show":False
            },
            
            {
                "name":"stu1_",
                "server":"8card",
                "logName":"stu1",
                "path":"",
                "show":False
            }
            ,
            {
                "name":"stu2_",
                "server":"4card",
                "logName":"stu2",
                "path":"",
                "show":False
            }
            ,{
                "name":"stunoreset1_",
                "server":"lyh",
                "logName":"stunoreset1",
                "path":"",
                "show":False
            }
            ,{
                "name":"stu3_",
                "server":"lyh",
                "logName":"stu3",
                "path":"",
                "show":False
            }
            ,{
                "name":"stubc1_",
                "server":"8card",
                "logName":"stubc1",
                "path":"",
                "show":False
            }
            ,{
                "name":"concat_puresc3_",
                "server":"8card",
                "logName":"concat_puresc3",
                "path":"",
                "show":False
            }
            ,{
                "name":"listu1_",
                "server":"8card",
                "logName":"listu1",
                "path":"",
                "show":False
            }
            ,{
                "name":"spatialgatedeconv1_",
                "server":"lyh",
                "logName":"spatialgatedeconv1",
                "path":"",
                "show":False
            }
            ,{
                "name":"spatialgatedeconv1_",
                "server":"lyh",
                "logName":"spatialgatedeconv1",
                "path":"",
                "show":False
            }
            ,{
                "name":"LGRU2",
                "server":"4card",
                "logName":"LGRU2",
                "path":"",
                "show":False
            }
            ,{
                "name":"SkipConnection1",
                "server":"4card",
                "logName":"SkipConnection1",
                "path":"",
                "show":False
            }
            ,{
                "name":"STU2",
                "server":"4card",
                "logName":"STU2",
                "path":"",
                "show":True
            }
            ,{
                "name":"LGRU4",
                "server":"4card",
                "logName":"LGRU4",
                "path":"",
                "show":True
            }
            ,{
                "name":"RNN_UNet2",
                "server":"4card",
                "logName":"RNN_UNet2",
                "path":"",
                "show":False
            }
        ]
    }
    choiceKey   = args.whichgroup
    choiceKey   = choiceKey.lower()
    logsBasePath= choiceKey


    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./logs/"+logsBasePath):
        os.makedirs("./logs/"+logsBasePath)
    
    selectedGroup = groupsLogs[choiceKey]
    x = []
    y = []
    names = []
    key_str = keydict[whichkey.lower()]
    # if whichkey == 'PSNR':
    #     key_str = psnrkey_str
    # elif whichkey == 'ACC':
    #     key_str = acckey_str

    for i in range(len(selectedGroup)):
        current = selectedGroup[i]
        if not current["show"]:
            continue
        serverInfo = serverList[current["server"]]
        uploader = fileUploaderClass(serverInfo["ip"],serverInfo["user"],serverInfo["passwd"])
        logpath = serverInfo["basePath"] + current["logName"] + "/summary"
        tensorfile = uploader.sshScpGetNames(logpath)
        for t in range(len(tensorfile)):
            localFile = "./logs/" + logsBasePath + "/" + tensorfile[t]
            current["path"] = localFile
            remoteFile= logpath + "/" + tensorfile[t]
            uploader.sshScpGet(remoteFile,localFile)
            print("get file success : %s"%remoteFile)
            names.append(current["name"])
            ea=event_accumulator.EventAccumulator(localFile) 
            ea.Reload()
        # print(ea.scalars.Keys())
            val_sin=ea.scalars.Items(key_str)
            if t>0:
                yTemp += [i.value for i in val_sin]
                xTemp += [i.step for i in val_sin]
            else:
                yTemp = [i.value for i in val_sin]
                xTemp = [i.step for i in val_sin]
        # x_sin =[i for i in range(len(y_sin))]
        if smoothratio > 0.0:
            yTemp =smooth(yTemp,smoothratio)
        x.append(xTemp)
        y.append(yTemp)
        print("prepare %s data success!"%current["name"])
        
        
    # groupsLogs  = {
    #     "nosc":[
    #         'events.out.tfevents.1572942165.gdp',
    #         'events.out.tfevents.1573017636.gdp',
    #         'events.out.tfevents.1573034558.gdp-SYS-7049GP-TRT', #r
    #         'events.out.tfevents.1572963354.DESKTOP-OJMRVAN',
    #         'events.out.tfevents.1572941861.gdp-SYS-7049GP-TRT',
    #         'events.out.tfevents.1573028860.gdp-SYS-7049GP-TRT'
    #     ],
    #     "diffsc":[
    #         # 'events.out.tfevents.1572940638.gdp-SYS-7049GP-TRT',
    #         # 'events.out.tfevents.1572952052.gdp',
    #         # 'events.out.tfevents.1572960346.DESKTOP-REVUAPJ',
    #         'events.out.tfevents.1573025329.ubuntu',
    #         # 'events.out.tfevents.1573023003.ubuntu',
    #         'events.out.tfevents.1573053567.gdp',
    #         # 'events.out.tfevents.1573052632.ubuntu',
    #         'events.out.tfevents.1573068704.ubuntu',
    #         # 'events.out.tfevents.1573067510.ubuntu',
            
    #     ],
    #     "diffsc1":[
    #         'events.out.tfevents.1573025329.ubuntu',
    #         'events.out.tfevents.1573096833.gdp'   
    #     ]
    # }
    # groupsNames = {
    #     "nosc":[
    #         'BC_',
    #         'BClaten5L_',
    #         'BClaten4L_',
    #         'CBN_',
    #         'Concat5L_',
    #         'Concat4L_'
    #     ],
    #     "diffsc":[
    #         # 'LSTU-SEC_',
    #         # 'LSTU-CBAMNC_',
    #         # 'LSTU-SEC_Concat_',
    #         'PureSC_Concat_',
    #         # 'SpatialGate_Concat_',
    #         'SpatialGateLaten2_Concat_',
    #         # 'SpatialGateLaten1_Concat_',
    #         'LatenUpGateAgu2_Concat_',
    #         # 'LatenUpGateAgu1_Concat_',
    #     ],
    #     "diffsc1":[
    #         'PureSC_Concat_',
    #         'LatenUpGateAguBC_Concat_'
    #     ]
    # }
    #开启一个窗口，num设置子图数量，figsize设置窗口大小，dpi设置分辨率
    figsize = 11,9
    figure, ax = plt.subplots(figsize=figsize)
    # figure, ax = plt.figure(num=1, figsize=(500, 500),dpi=80)
    #直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列   
    # plt.plot(x,y1)
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 23,
    }
    # plt.title('Generator Convergence Speed Analysis')
    lines = []
    for i in range(len(y)):
        A,=plt.plot(x[i],y[i],label=names[i]+whichkey)
        lines.append(A)

    plt.legend()
    plt.grid()
    legend = plt.legend(handles=lines,prop=font1)
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlabel('Iteration times',font1)
    plt.ylabel(whichkey,font1)
    #显示绘图结果
    plt.show()

if __name__ == '__main__':
    main()