from google.colab import drive
drive.mount('gdrive')

import torch
from torchvision import transforms
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import cv2
import glob,os
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataFolder = '/content/gdrive/MyDrive/clusteringData'
multiviewImagelist = []

for fol in os.listdir(dataFolder): multiviewImagelist.append(glob.glob(os.path.join(dataFolder,fol,'*.png')))



def squaredEuclideanNorm(x, y):
    diff = x - y
    squared_norm = np.sum(diff ** 2)
    return squared_norm

class conv2linear(nn.Module):
    def __init__(self,w,h):
        super(conv2linear, self).__init__()

        self.linear1 = nn.Linear(3*w*h, 4096)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(4096,2048)
        self.linear3 = nn.Linear(2048,1024)

    def forward(self, x):
        x = x.view(1,x.shape[0]*x.shape[1]*x.shape[2])
        x_linear = self.relu1(self.linear1(x))
        x_linear = self.relu1(self.linear2(x_linear))
        x_linear = self.relu1(self.linear3(x_linear))

        return x_linear

conv2linear = conv2linear(224,224)


"""
Option 2 - convolutional autoencoder instead of fully-connected
"""
class baseAutoencoder(nn.Module):
    def __init__(self):
        super(baseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.toLinear = nn.Sequential(
              nn.Conv2d(1024, 1024, kernel_size=7, stride=3, padding=1),

        )


    def forward(self, x):
        z = self.encoder(x)
        x_cap = self.decoder(z)
        z_linear = self.toLinear(z)
        z_linear = z_linear.view(1, z_linear.size(0))

        return z_linear,x_cap

# model = baseAutoencoder()



class Linear_BaseAutoencoder(nn.Module):
    def __init__(self, input_size=1024):
        super(Linear_BaseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_cap = self.decoder(z)

        return z, x_cap

autoencoder_linear = Linear_BaseAutoencoder()




optimizer = torch.optim.Adam(autoencoder_linear.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)
"""
Pre-train model for Reconstruction loss
"""


modelFolderpath = './models'
if not os.path.isdir(modelFolderpath):
  os.makedirs(modelFolderpath, exist_ok = True)


for e in tqdm(range(5)):
  autoencoder_linear.train()
  for imgs in  multiviewImagelist:
    L_r = 0
    z_temp = []
    n_views = len(imgs)
    for img in imgs:
      img = cv2.imread(img)
      img = train_transform(img)
      x  = conv2linear(img)
      z,x_cap = autoencoder_linear(x)
      L_r += torch.tensor(squaredEuclideanNorm(x.detach().numpy() ,x_cap.detach().numpy() ), requires_grad = True)
    optimizer.zero_grad()
    L_r.backward()
    optimizer.step()

  exp_lr_scheduler.step()
  if e % 2 == 0  :
      model_save_name = f'model_{e}.pt'
      model_save_path = os.path.join(modelFolderpath, f'model_{e}.pt')
      torch.save(autoencoder_linear.state_dict() , model_save_path)

class H_cap(nn.Module):
    def __init__(self, n_of_imgs = 4 , input_dim = 1536, output_dim = 1536, vec_len = 1536):
        super(H_cap, self).__init__()

        self.W1 = nn.Linear(input_dim, output_dim, bias=False)
        self.W2 = nn.Linear(input_dim, output_dim, bias=False)
        self.W3 = nn.Linear(input_dim, output_dim, bias=False)

        self.b1 = nn.Parameter(torch.randn(n_of_imgs,output_dim))
        self.b2 = nn.Parameter(torch.randn(n_of_imgs,output_dim))
        self.b3 = nn.Parameter(torch.randn(n_of_imgs,output_dim))

        self.relu1 = nn.ReLU()

        self.Wr = nn.Parameter(torch.randn(vec_len,vec_len))
        self.Wq1 = nn.Parameter(torch.randn(vec_len,vec_len))
        self.Wq2 = nn.Parameter(torch.randn(vec_len,vec_len))


    def forward(self, Z):

        R = torch.matmul(Z,self.Wr)
        Q1 = torch.matmul(Z,self.Wq1)
        Q2 = torch.matmul(Z,self.Wq2)

        S = torch.nn.functional.softmax((torch.matmul(Q1,Q2.T)/torch.sqrt(torch.tensor(m))),dim=1)
        Z_cap = []
        for i in range(0,S.shape[0]):
          Z_cap_temp = 0
          S_i = S[i].tolist()
          for j in range(0,Z.shape[0]):
            S_j = torch.tensor(S_i[j]).reshape(1,1)
            R_j = R[j].unsqueeze(0)
            Z_cap_temp += torch.matmul(S_j,R_j)
          Z_cap.append(Z_cap_temp)

        Z_cap = torch.stack(Z_cap)
        Z_cap = Z_cap.reshape(Z_cap.shape[0],Z_cap.shape[-1])

        H_cap = self.relu1(self.W1(torch.add(Z,Z_cap)) + self.b1)

        H_cap = self.W2(H_cap) + self.b2

        H_cap = self.W3(H_cap) + self.b3


        return H_cap,S,Z

H_consensus = H_cap()

pretrained_autoencoder = Linear_BaseAutoencoder()
pretrained_autoencoder.load_state_dict(torch.load('/content/models/model_4.pt'))
pretrained_autoencoder.eval()

def cos_sim(H_cap,Hv) :
  mul = torch.matmul(H_cap,Hv)
  div = torch.matmul(torch.tensor(l2_norm(H_cap.detach().numpy())).reshape(1,-1),torch.tensor(l2_norm(Hv.detach().numpy())).reshape(1,-1))
  mul = mul/div
  return mul

def l2_norm(vector):
    return np.linalg.norm(vector)
    
def getLc(temperature,vec_len,n_views,H_cap,S,Z):
  print('temperature,vec_len,n_views,H_cap,S,Z',temperature,vec_len,n_views,H_cap.shape,S.shape,Z.shape)
  base_vec_len = int(vec_len/n_views)
  num_div_outer = 0
  for i in range(len(H_cap)):
    H_cap_i = H_cap[i].reshape(1,-1)
    S_i = S[i].tolist()

    outer_div = 0
    for v in range(n_views) :
      numerator = 0
      Hv = H_cap_i[0][(v*base_vec_len):((v+1)*base_vec_len)].reshape(1,-1)
      C = cos_sim(H_cap_i.T,Hv)
      numerator = (torch.exp(C/temperature))
      inner_denom = 0
      for j in range(0,Z.shape[0]):
        S_j = torch.tensor(S_i[j]).reshape(1,1)
        temp = (1-S_j) * ((C/temperature))
        inner_denom += torch.exp(temp)
      inner_denom -= torch.exp((torch.tensor(1/temperature)))
      outer_div += torch.log((numerator/inner_denom))

    num_div_outer += outer_div
    del outer_div
    del inner_denom
    del C
    del numerator

  Lc = - (num_div_outer/2*n_views)
  return Lc


modelFolderpath2 = './models2'
if not os.path.isdir(modelFolderpath2):
  os.makedirs(modelFolderpath2, exist_ok = True)

L_r = 0
Z = []

for e in tqdm(range(5)):
  Z = []
  H_consensus.train()
  for imgs in  multiviewImagelist:
    L_r = 0
    z_temp = []
    n_views = len(imgs)
    for img in imgs:
      img = cv2.imread(img)
      img = train_transform(img)
      x  = conv2linear(img)
      z,x_cap = pretrained_autoencoder(x)
      L_r += torch.tensor(squaredEuclideanNorm(x.detach().numpy() ,x_cap.detach().numpy() ), requires_grad = True)
      z_temp.extend(z[0])
    Z.append(torch.stack(z_temp))
  L_r /= n_views
  Z = torch.stack(Z)
  # n_views -> number of views per image
  m = Z.shape[0] # total number of images
  vec_len = Z.shape[1] # vector length

  # Wr = nn.Parameter(torch.randn(vec_len,vec_len))
  # Wq1 = nn.Parameter(torch.randn(vec_len,vec_len))
  # Wq2 = nn.Parameter(torch.randn(vec_len,vec_len))

  # R = torch.matmul(Z,Wr)
  # Q1 = torch.matmul(Z,Wq1)
  # Q2 = torch.matmul(Z,Wq2)

  # S = torch.nn.functional.softmax((torch.matmul(Q1,Q2.T)/torch.sqrt(torch.tensor(m))),dim=1)

  # Z_cap = []
  # for i in range(0,S.shape[0]):
  #   Z_cap_temp = 0
  #   S_i = S[i].tolist()
  #   for j in range(0,Z.shape[0]):
  #     S_j = torch.tensor(S_i[j]).reshape(1,1)
  #     R_j = R[j].unsqueeze(0)

  #     Z_cap_temp += torch.matmul(S_j,R_j)
  #   Z_cap.append(Z_cap_temp)

  # Z_cap = torch.stack(Z_cap)
  # Z_cap = Z_cap.reshape(Z_cap.shape[0],Z_cap.shape[-1])

  H_cap,S,Z = H_consensus(Z)
  L_c = getLc(1,vec_len,n_views,H_cap,S,Z)

  final_loss = L_r + L_c.mean()
  optimizer.zero_grad()
  final_loss.backward()
  optimizer.step()

  exp_lr_scheduler.step()
  if e % 2 == 0  :
      model_save_name = f'model_{e}.pt'
      model_save_path = os.path.join(modelFolderpath2, f'model_{e}.pt')
      torch.save(autoencoder_linear.state_dict() , model_save_path)

"""
H_cap cluster sample (4,1536) output from above trained model
"""

H_cap = nn.Parameter(torch.randn(4,1536)) # initalize random of number_of_images, concatenate(vecs of len 256)

n, d = H_cap.shape

k = 4

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(H_cap.detach().numpy())


V = kmeans.cluster_centers_
labels = kmeans.labels_

U_initial = np.zeros((n, k))
for i in range(n):
    U_initial[i, labels[i]] = 1


U_initial = U_initial / U_initial.sum(axis=1, keepdims=True)

U_initial_flat = U_initial.flatten()

print('U_initial_flat',U_initial_flat)


def objective(U_flat, H, V):
    U = U_flat.reshape((H_cap.shape[0], V.shape[0]))
    return np.linalg.norm(H_cap.detach().numpy() - np.dot(U, V), 'fro')**2

def constraint_U_sum(U_flat):
    U = U_flat.reshape((H_cap.shape[0], k))
    return np.sum(U, axis=1) - 1

def constraint_U_nonneg(U_flat):
    U = U_flat.reshape((H_cap.shape[0], k))
    return U

cons = [{'type': 'eq', 'fun': constraint_U_sum},
        {'type': 'ineq', 'fun': lambda x: x}]

result = minimize(objective, U_initial_flat, args=(H_cap.detach().numpy(), V), constraints=cons, method='SLSQP', options={'disp': True})

U_optimal = result.x.reshape((H_cap.shape[0], k))

cluster_assignments = np.argmax(U_optimal, axis=1)

print(cluster_assignments)