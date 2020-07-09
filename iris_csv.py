import torch
import os,glob
import random,csv
from torchvision import datasets,transforms
from PIL import Image
from  torch.utils.data import Dataset,DataLoader
from visdom import Visdom
import time
import cv2 as cv
vis=Visdom()

class Iris(Dataset):
    def __init__(self,root,height,width,mode):
        super(Iris, self).__init__()
        self.root=root
        self.height=height
        self.width=width
        self.mode=mode
        self.classes_list=[]
        self.classes_list=os.listdir(self.root)
        for class_name in self.classes_list:
            if not os.path.isdir(os.path.join(self.root,class_name)):
                self.classes_list.remove(class_name)
        self.images,self.labels=self.create_csv('iris.csv')
        if mode=='train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode=='validation':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        elif mode=='test':
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
    def create_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images_path=[]
            for name in self.classes_list:
                images_path+=glob.glob(os.path.join(self.root,name,'*.bmp'))
            random.shuffle(images_path)
            print(images_path)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for image_path in images_path:
                    label = image_path.split(os.sep)[-2]
                    writer.writerow([image_path, label])
                print('writen into csv file:', filename)

        images=[]
        labels=[]
        with open(os.path.join(self.root,filename)) as f:
            reader=csv.reader(f)
            for row in reader:
                img,lab=row
                lab=int(lab)
                images.append(img)
                labels.append(lab)
        # print(images)
        # print(labels)
        assert len(images)==len(labels)
        return images,labels


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img,label=self.images[idx],self.labels[idx]
        trans=transforms.Compose([
            lambda x: Image.open(img).convert('RGB'),
            # lambda x: Image.open(img).convert('L'),
            transforms.Resize(int(self.height),int(self.width)),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.525],
            #                      std=[0.284])
        ])
        img=trans(img)
        label=torch.tensor(label)
        return img,label
# def main():
#     db=Iris('/Users/wenyu/Desktop/TorchProject/HRnet/demo',64,128,'train')
#     x,y=next(iter(db))
#     print(x.shape, y.float().item())
#     vis.image(x, win='iris', opts=dict(title='iris'))
#     loader=DataLoader(db,batch_size=16,shuffle=True)
#     for x,y in loader:
#         print(x.size())
#         vis.images(x,nrow=4,win='batch_iris',opts=dict(title='batch_iris'))
#         vis.text(str(y.numpy()),win='label',opts=dict(title='batch_label'))
#         time.sleep(10)
if __name__ == '__main__':
    pass
    # main()