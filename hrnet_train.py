import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import yaml
import visdom
from torch.utils.data import DataLoader
from HRnet.iris_csv import Iris
from HRnet.hrnet import HighResolutionNet
batch_size=32
begin_epoch=0
end_epoch=100
lr_factor=0.1
lr_step=[30,60,90]
base_lr=0.05
wd=0.0001
momentum=0.9
torch.manual_seed(1234)
vis=visdom.Visdom()

# device_num=torch.cuda.current_device()

train_db=Iris('/Users/wenyu/Desktop/TorchProject/HRnet/demo',64,128,'train')
validation_db=Iris('/Users/wenyu/Desktop/TorchProject/HRnet/demo',64,128,'validation')
test_db=Iris('/Users/wenyu/Desktop/TorchProject/HRnet/demo',64,128,'test')

train_loader=DataLoader(train_db,batch_size=batch_size,shuffle=True,num_workers=4)
validation_loader=DataLoader(validation_db,batch_size=batch_size,num_workers=2)
test_loader=DataLoader(test_db,batch_size=batch_size,num_workers=2)


def evaluate(model,loader):
    correct=0
    total_num=len(loader.dataset)
    for x,y in loader:
        # x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
        correct+=torch.eq(pred,y).sum().float().item()
    return correct/total_num

def load_yaml(yaml_path):
    fi=open(yaml_path)
    conf=yaml.load(fi,Loader=yaml.FullLoader)
    return conf

def load_config(config,**kwargs):
    module=HighResolutionNet(config,**kwargs)
    return module

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        # print(m.weight)
        # print(m.weight.shape)



def main():
    yaml_path='/Users/wenyu/Desktop/TorchProject/' \
              'HRnet/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    conf=load_yaml(yaml_path)
    model=load_config(config=conf)
    # if torch.cuda.is_available():
    #     model.cuda()
    model_dict=model.state_dict()
    pretrained_dict=torch.load('/Users/wenyu/Desktop/TorchProject/HRnet/hrnet_w18_small_model_v1.pth',
                               map_location=torch.device('cpu'))
    # pretrained_dict=torch.load('/Users/wenyu/Desktop/TorchProject/HRnet/hrnet_w18_small_model_v1.pth')

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.classifier_new.apply(init_weights)
    freeze_list = list(model.state_dict().keys())[0:-2]
    for name,param in model.named_parameters():
        if name in freeze_list:
            param.requires_grad=False
        if param.requires_grad:
            pass
    # base_lr=0.05
    # optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=base_lr,
    #                     momentum=momentum,weight_decay=wd,nesterov=True)
    optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=base_lr)
    lr_scheduler=MultiStepLR(optimizer,milestones=[30,60,90],gamma=0.1,last_epoch=-1)
    fun_loss = nn.CrossEntropyLoss()
    vis.line([0.], [-1], win='train_loss', opts=dict(title='train_loss'))
    vis.line([0.], [-1], win='validation_acc', opts=dict(title='validation_acc'))
    global_step = 0
    best_epoch, best_acc = 0, 0
    for epoch in range(0,100):
        for step, (x, y) in enumerate(train_loader):
            # x,y=x.cuda(),y.cuda()
            # x,y=x.cuda(),y.cuda(non_blocking=True)
            logits=model(x)
            loss=fun_loss(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            vis.line([loss.item()], [global_step], win='train_loss', update='append')
            global_step += 1
        lr_scheduler.step()

        # validation
        if epoch%5==0:
            val_acc = evaluate(model, validation_loader)
            if  val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_{}.pth'.format(epoch))
                vis.line([val_acc], [global_step], win='validation_acc', update='append')
    print('best acc', best_acc, 'best epoch', best_epoch)


if __name__ == '__main__':
    main()