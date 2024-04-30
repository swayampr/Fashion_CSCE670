import itertools, os, warnings, cv2, torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn import metrics
import resnet
from utils import prepare_dataloaders

warnings.filterwarnings('ignore')
plt.rc('font',family='Times New Roman')

train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders()

device = torch.device("cuda:0")

# Helper funcions
def loadimg_from_id(ID, root_dir=test_dataset.root_dir):
    imgs = []
    for id in ID:
        if 'mean' in id:
            img_path = os.path.join(test_dataset.data_dir, id.split('_')[0]) + '.png'
        else:
            img_path = os.path.join(root_dir, *id.split('_')) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        img = test_dataset.transform(img)
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = imgs.unsqueeze(0)
    return imgs

# Compute the gradients of each element in the comparison matrices to approximate the problem of each input
def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None
    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)

    # Forward
    out, *_ = model._compute_score(img)
    one_hot = torch.FloatTensor([[-1]]).to(device)

    # Backward
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)
    
    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

# Convert relation vector to 4 matrix
def vec2mat(relation, select):
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

# Visualize diagnosis on relationships of 4 scales
def show_rela_diagnosis(relation, select, cmap=plt.cm.Blues):
    mats = vec2mat(relation , select)
        
    fig = plt.figure(figsize=(20, 5))
    all_names = {0:'Top', 1:'Bottom', 2:'Shoe', 3:'Bag', 4:'Accssory'}
    node_names = {i:all_names[s] for i, s in enumerate(select)}
    
    edge_vmax = max(m.max() for m in mats)
    edge_vmin = min(m.min() for m in mats)
    
    container = []
    for idx in range(4):
        A = mats[idx]
        if isinstance(A, torch.Tensor):
            A = A.cpu().data.numpy()
                   
        A = np.triu(A, k=1)
        A = np.round(A, decimals=2)
        container.append(A)
    container = np.stack(container)
    sorted_vedge = sorted(container.ravel(), reverse=True)
        
    for idx in range(4):
        plt.subplot(1, 4, idx+1)
        plt.title("Layer {}".format(idx+1), fontsize=28)
        A = mats[idx]
        if isinstance(A, torch.Tensor):
            A = A.cpu().data.numpy()
                   
        A = np.triu(A, k=1)
        A = np.round(A, decimals=2)
        indices = np.triu_indices(A.shape[0], k=1)
        weights = A[indices[0], indices[1]]
        # Generate graph
        G = nx.Graph()
        for i, j, weight in zip(*indices, weights):
            G.add_edge(node_names[i], node_names[j], weight=weight)
        
        elarge, esmall, filtered_weights = [], [], []
        for e in G.edges(data=True):
            if e[2]['weight'] in sorted_vedge[:3]:
                elarge.append((e[0], e[1]))
            else:
                esmall.append((e[0], e[1]))
                filtered_weights.append(e[2]['weight'])
        pos=nx.circular_layout(G) # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes()], node_size=1600, node_color='#A0CBE2')

        # edges
        nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6, alpha=0.5, edge_color=filtered_weights, edge_cmap=cmap,
                               edge_vmax=edge_vmax, edge_vmin=edge_vmin)
        nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6, alpha=0.5, edge_color='red', style='dashed')

        # labels
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_labels(G,pos, font_size=18, font_family='Times New Roman')
        if len(select) == 4:
            nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels, label_pos=0.33)
        else:
            nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels)
        
        plt.axis('off')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("diagnosis.pdf")
    print("Save diagnosis result to diagnosis.pdf")

# Output the most incompatible item in the outfit  
def item_diagnosis(relation, select):
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).bool()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
 
    order = [i for i, j in sorted( enumerate(result) , key=lambda x:x[1], reverse=True)]

    print("Result: ",result)
    print("Printing Order: ", order)
    return result, order

# Show multiple items in a outfit.
def show_imgs(x, select=None, fname="outfit.pdf"):
    if select is None:
        select = list(range(5))
    fig = plt.figure(figsize=(5*len(select), 5))
    for i, s in enumerate(select):
        plt.subplot(1, len(select), i+1)
        img = x[s]
        img = img.cpu().data.numpy().transpose((1, 2, 0)) * 255
        img = img[..., :3]
        plt.gca().axis('off')
        plt.imshow(np.uint8(img))
    plt.savefig(fname)
    print("Save outfit to {}".format(fname))
    
if __name__ == "__main__":
    # Load model weights
    from model import CompatModel
    model = CompatModel(embed_size=500, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
    model.load_state_dict(torch.load('./model_train.pth'))
    model.eval()
    
    print("="*80)
    ID = ['178118160_1', 'bottom_mean', '199285568_4', '111355382_5', '209432387_4']
    x = loadimg_from_id(ID).to(device)
    # Remove the mean images for padding the sequence when making visualization
    select = [i for i, l in enumerate(ID) if 'mean' not in l]

    print("Step 1: Show images in an outfit...")
    show_imgs(x[0], select)

    print("\nStep 2: Show diagnosis results...")
    relation, out = defect_detect(x, model)
    relation = relation.squeeze().cpu().data
    show_rela_diagnosis(relation, select, cmap=plt.cm.Blues)
    result, order = item_diagnosis(relation, select)
    print("Predicted Score: {:.4f}\nProblem value of each item: {}\nOrder: {}".format(out, result, order))
    print("="*80)