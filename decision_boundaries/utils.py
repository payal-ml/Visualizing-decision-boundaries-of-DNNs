from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import torch
from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack
from foolbox import PyTorchModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import foolbox

# adding hook to get the output of last conv layer
def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# predicting probabilty, label and output of last conv layer
def get_predictions(model, sample_batch):
    activation = {}
    model.layer4.register_forward_hook(get_activation("layer4", activation))
    class_score = model(sample_batch.cuda())
    out_act_layer = activation["layer4"].cpu().numpy()
    out_act_layer = out_act_layer.reshape(out_act_layer.shape[0], -1)
    softmax_score=torch.nn.functional.softmax(class_score, dim=1)
    pred_probability=torch.max(softmax_score,dim=1).values.detach().cpu().numpy()*100
    pred_label=torch.max(softmax_score,dim=1).indices.cpu()
    return pred_probability, pred_label, out_act_layer

# selcting the sample with pred_label== sample_labels
def correct_samples(cifar_model,sample_dataloader):
    sample_features, sample_labels= next(iter(sample_dataloader))
    pred_probability, pred_label, out_layer4= get_predictions(cifar_model, sample_features)

    corr_features=sample_features[pred_label== sample_labels]
    corr_labels=sample_labels[pred_label== sample_labels]
    corr_out_layer4=out_layer4[pred_label== sample_labels]
    corr_pred_prob=pred_probability[pred_label== sample_labels]

    return pred_label, out_layer4, corr_features, corr_labels, corr_out_layer4, corr_pred_prob

# getting index of samples
def get_index(sample,correct_features):
    sample_cal=sample[:,np.newaxis,:]
    dis = correct_features-sample_cal
    final_distance=dis[:,:,0]**2+dis[:,:,1]**2
    index=np.argmin(final_distance,axis=1)
    return index

# projecting points using PCA or UMAP to 2D space
def projection(
    features: torch.Tensor,
    seed: int = 42,
    dim: int = 2,
    encoder: str = "umap"
        
):
    #supported_encoders = ["pca", "umap"]
     # store standardized data in feature_std
    std=StandardScaler().fit(features)
    feature_std = std.transform(features)    

    if encoder.lower() == "pca":              
        embed= PCA(n_components=dim, random_state=seed).fit(feature_std)
    elif encoder.lower() == "umap":      
        embed =umap.UMAP(n_components=dim, random_state=seed).fit(feature_std)
   
    
    return std,embed

# Visualizing the voronoi diagram, pie charts and scatterplot
def visualization(encoded_features,nameofclasses,pred_sample_label,pie_radius,head_arrow,array_attacked,
                    dataset_name,attack_name,epsilons,abs_stepsize,flag=0):

    centers, vor = get_voronoi_attributes(encoded_features)
    colors_edge = ['darkblue','goldenrod','green','red','purple','saddlebrown','palevioletred','gray','darkolivegreen','teal']

    fig, ax = plt.subplots(figsize=(20, 10), layout='constrained')
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    alpha =0.1
    iter=array_attacked.shape[0]
    alpha_values=np.linspace(0.1,1,iter+1)
    arrow_alpha_values=np.linspace(0.0,1,iter)    

    for i in range(iter):

        points=array_attacked[i]         
        for p in points :
            
            ax.pie([p[3].round(1),100-(p[3].round(1))],
                    colors=[colors_edge[int(p[2])],"white"], 
                    wedgeprops = {'linewidth': 3,'edgecolor':colors_edge[int(p[2])],'alpha' :alpha_values[i+1]},
                    radius=pie_radius,
                    center=(p[0],p[1]))
   
    for j in range(iter-1):

        array1=array_attacked[j][:,0:2]
        array2=array_attacked[j+1][:,0:2]
        diff=array2-array1
        for k in range(len(array1)):
            ax.arrow(array1[k,0],array1[k,1],diff[k,0],diff[k,1],head_width=head_arrow,alpha=arrow_alpha_values[j+1],length_includes_head=True)
        
         
    for index in range(len(nameofclasses)):
        ax.scatter(
            encoded_features[pred_sample_label== index, 0],
            encoded_features[pred_sample_label == index, 1],            
            edgecolors= colors_edge[index],
            facecolors="none" ,alpha =alpha,
            linewidths =3)
        
    voronoi_plot_2d(vor,ax=ax)

    y_init = ax.get_ylim()
    x_init = ax.get_xlim()
    # from zero to xlim/ylim with step 1
    ax.set_yticks(range(round(y_init[0])-1, round(y_init[1])+2, 1))
    ax.set_xticks(range(round(x_init[0])-1, round(x_init[1])+2, 1))
    ax.set_frame_on(True)
    pat = [mpatches.Patch(color=col, label=lab) for col, lab in zip(colors_edge, nameofclasses)]
    ax.legend(handles=pat,title="Predicted classes", bbox_to_anchor=(1.02,1), loc="upper left",fancybox=True, shadow=True)  
    
    if flag==0:
        name="Near_Center_"+dataset_name+"_samples_"+attack_name+"_attack_with_epsilon="+str(epsilons)+",abs_stepsize="+str(abs_stepsize)+".jpg"
        ax.set_title(name)
        plt.savefig("results/"+name)
    elif flag==1 :
        name="Near_Vertices_"+dataset_name+"_samples_"+attack_name+"_attack_with_epsilon="+str(epsilons)+",abs_stepsize="+str(abs_stepsize)+".jpg"
        ax.set_title(name)
        plt.savefig("results/"+name)
    else :
        name="Near_Boundary_"+dataset_name+"_samples_"+attack_name+"_attack_with_epsilon="+str(epsilons)+",abs_stepsize="+str(abs_stepsize)+".jpg"
        ax.set_title(name)
        plt.savefig("results/"+name)
        

    plt.show()

# Applying attack PGD and CarliniWagner
def adv_attack(model, sample_features, sample_labels,attack_name, std, embed, feature_label_array, epsilons=16/256, abs_stepsize=1/256, iterations=6):
    if attack_name.lower() not in ["pgd", "carlini_wagner"]:
        raise ValueError("Only supported attacks PGD and Carlini_Wagner")
    

    fmodel = PyTorchModel(model, bounds=(0, 1))

    if attack_name.lower()=="pgd":
        attack = LinfPGD(abs_stepsize=abs_stepsize,random_start=False,steps=10) 
    elif attack_name.lower()=="carlini_wagner":
        attack=L2CarliniWagnerAttack(steps=1000,abort_early=False ,stepsize=abs_stepsize)

    feature_label_array=np.expand_dims(feature_label_array,axis=0)
    sample_labels=sample_labels.squeeze().cuda()
    criterion = foolbox.criteria.Misclassification(sample_labels)
    
    for i in range(iterations):
        
        sample_features=sample_features.cuda()
                
        raw_advs, clipped_advs, success = attack(fmodel, sample_features, criterion, epsilons=epsilons)        
        
        pred_probability_attacked, pred_label_attacked, out_layer4_attacked=get_predictions(model, clipped_advs)     
        
        pred_label_attacked=pred_label_attacked.reshape(pred_label_attacked.shape[0],-1) 
        pred_probability_attacked=pred_probability_attacked.reshape(pred_probability_attacked.shape[0],-1)

        encoded_features_attacked=embed.transform(std.transform(out_layer4_attacked))        
        feature_label_attacked=np.hstack((encoded_features_attacked,pred_label_attacked,pred_probability_attacked))
        feature_label_array=np.concatenate((feature_label_array,np.expand_dims(feature_label_attacked,axis=0)),axis=0)        
        sample_features=clipped_advs       
               
    return feature_label_array

# get center and vertices of voronoi diagram
def get_voronoi_attributes(encoded_features):
    km_10=KMeans(n_clusters=10,random_state=123)
    km_10.fit(encoded_features)
    centers = km_10.cluster_centers_
    vor = Voronoi(centers)     
    return centers, vor


def plot_data(model,dataset_name,sample_dataloader,nameofclasses,attack_name,epsilons,abs_stepsize):
    pred_label, out_layer4, corr_features, corr_labels, corr_out_layer4, corr_pred_prob= correct_samples(model,sample_dataloader)

    corr_labels=corr_labels.reshape(corr_labels.shape[0],-1) 
    corr_pred_prob=corr_pred_prob.reshape(corr_pred_prob.shape[0],-1)


    std,embed=projection(features=out_layer4,seed=1024,encoder="umap")
    encoded_features=embed.transform(std.transform(out_layer4))
    correct_encoded_features=embed.transform(std.transform(corr_out_layer4))

    centers, vor = get_voronoi_attributes(encoded_features)
    vertices=vor.vertices
    if dataset_name.lower()=="cifar10":
         boundary_points=np.array([[0,2.5],
                            [0,4.5],
                            [2,6.5],
                            [4,1.5],
                            [6.5,2],
                            [8,4],
                            [4.5,5.5]] , dtype=float)
         pie_radius=0.09
         head_arrow=0.03
    elif dataset_name.lower()=="mnist" :
        boundary_points=np.array([[10,-9],
                        [-6,4.5],
                        [22.5,5.5],
                        [12,21.5],
                        [5.5,13.5]] , dtype=float)
        pie_radius=0.4
        head_arrow=0.1
    elif dataset_name.lower()=="fashionmnist":
        boundary_points=np.array([[10,-9],
                        [19,6],
                        [17,3],
                        [4,-7.5],
                        [-8,2],
                        [0,14]] , dtype=float)
        pie_radius=0.3
        head_arrow=0.1
    elif dataset_name.lower()=="stl10":
        boundary_points=np.array([[0.5,7.8],
                            [3,8.8],
                            [2,6.7],
                            [5,8.5],
                            [4.5,7.5],
                            [7,8.3],
                            [7,5.8],
                            [9.8,5.5],
                            [9.8,6.9],
                            [9,8.6]] , dtype=float)
        pie_radius=0.09
        head_arrow=0.03

    center_index= get_index(centers,correct_encoded_features)   
    vert_index=get_index(vertices,correct_encoded_features)    
    boun_index=get_index(boundary_points,correct_encoded_features)   

    center_array=np.hstack((correct_encoded_features[center_index],corr_labels[center_index],corr_pred_prob[center_index]))    
    vert_array=np.hstack((correct_encoded_features[vert_index],corr_labels[vert_index],corr_pred_prob[vert_index]))
    boun_array=np.hstack((correct_encoded_features[boun_index],corr_labels[boun_index],corr_pred_prob[boun_index]))
    

    center_attacked =adv_attack(model,corr_features[center_index],corr_labels[center_index],attack_name,std,embed,center_array,epsilons=epsilons,abs_stepsize=abs_stepsize,iterations=6)
    vert_attacked =adv_attack(model,corr_features[vert_index],corr_labels[vert_index],attack_name,std,embed,vert_array,epsilons=epsilons,abs_stepsize=abs_stepsize,iterations=6)
    boun_attacked =adv_attack(model,corr_features[boun_index],corr_labels[boun_index],attack_name,std,embed,boun_array,epsilons=epsilons,abs_stepsize=abs_stepsize,iterations=6)
    
    visualization(encoded_features,nameofclasses,pred_label.numpy(),pie_radius,head_arrow,
    center_attacked,dataset_name,attack_name,epsilons=epsilons,abs_stepsize=abs_stepsize,flag=0
    )    
    visualization(encoded_features,nameofclasses,pred_label.numpy(),pie_radius,head_arrow,
    vert_attacked,dataset_name,attack_name,epsilons=epsilons,abs_stepsize=abs_stepsize,flag=1
    )
    visualization(encoded_features,nameofclasses,pred_label.numpy(),pie_radius,head_arrow,
    boun_attacked,dataset_name,attack_name, epsilons=epsilons,abs_stepsize=abs_stepsize,flag=2
    )




