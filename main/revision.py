import itertools, os, warnings, cv2, torch, torchvision, resnet
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
from sklearn import metrics
from utils import prepare_dataloaders
from diagnosis import *

warnings.filterwarnings('ignore')
plt.rc('font',family='Times New Roman')

# Retrieve the datset to substitute the worst item for the best choice.
def retrieve_sub(x, select, order):
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}

    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in test_dataset.data:
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    score, *_ = model._compute_score(x)
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
        print('problem_part: {}'.format(problem_part))
        print('best substitution: {}'.format(best_img_path[problem_part]))
        print('After substitution the score is {:.4f}'.format(best_score))
    
    show_imgs(x[0], select, "revised_outfit.pdf")
    return best_score, best_img_path

if __name__ == "__main__":
    # Load model weights
    from model import CompatModel
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
    model.load_state_dict(torch.load('./model_train.pth'))
    model.eval()
    
    print("="*80)
    # Comment different line to choose different outfit as example.
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

    print("\nStep 3: Substitute the problem items for revision, it takes a while to search...")
    best_score, best_img_path = retrieve_sub(x, select, order)
    print("="*80)