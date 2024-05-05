import torch
import numpy as np
import librosa
from sklearn.cluster import KMeans
import math


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def divergence(V,W,H, beta = 2):
    
    """
    beta = 2 : Euclidean cost function
    beta = 1 : Kullback-Leibler cost function
    beta = 0 : Itakura-Saito cost function
    """ 
    
    if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
    
    if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
    
    if beta == 2 : return 1/2*np.linalg.norm(W@H-V)

def NMF(V, S, beta = 2,  threshold = 0.05, MAXITER = 5000): 
    
    """
    inputs : 
    --------
        V         : Mixture signal : |TFST|
        S         : The number of sources to extract
        beta      : Beta divergence considered, default=2 (Euclidean)
        threshold : Stop criterion 
        MAXITER   : The number of maximum iterations, default=1000                                                     
    
    outputs :
    ---------
        W : dictionary matrix [KxS], W>=0
        H : activation matrix [SxN], H>=0
        cost_function : the optimised cost function over iterations
       
   Algorithm : 
   -----------
   
    1) Randomly initialize W and H matrices
    2) Multiplicative update of W and H 
    3) Repeat step (2) until convergence or after MAXITER   
    """
    
    counter = 0
    cost_function = []
    beta_divergence = 1
    
    K, N = np.shape(V)
    
    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))

    while beta_divergence >= threshold and counter <= MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )
        counter += 1
       
    return W,H, cost_function


device = "cuda"
valid_data = "YOUR DATASET HERE"
model = "YOUR MODEL HERE"
model = model.to(device)
model.eval()

n_hop = 256
n_fft = 512
n_components = 40
n_nmf_groups = 10
alpha = 0.5
exp_accs = 0
data_len = len(valid_data)

waveform = valid_data[0]["audio"]
true_label = valid_data[0]["intent_class"]
n = len(valid_data[0]["audio"])
waveform_pad = librosa.util.fix_length(np.array(waveform)[0], size=n)
output = model(waveform, output_attentions=True)
logits = output.logits
original_pred = torch.argmax(logits, dim=-1)[0]

y_pad = librosa.util.fix_length(np.array(waveform)[0], size=n + n_fft // 2)
sound_stft = librosa.stft(y_pad, n_fft = n_fft, hop_length = n_hop)
sound_stft_Magnitude = np.abs(sound_stft)
sound_stft_Angle = np.angle(sound_stft)
V = sound_stft_Magnitude + 1e-10
beta = 2
S = n_components
# Applying the NMF function
W, H, cost_function = NMF(V,S,beta = beta, threshold = 0.05, MAXITER = 5000) 

#OPTIONAL KMEANS CLUSTERING
component_labels = KMeans(n_clusters=n_nmf_groups, random_state=0, n_init="auto").fit_predict(np.transpose(W))      
nmf_groups_list = []
for i in range(n_nmf_groups):
    group_ids = np.where(component_labels == i)[0]
    nmf_groups_list.append(group_ids)

filtered_spectrograms = []
for k in range(n_nmf_groups):
    # Filter eash source components
    filtered_spectrogram = W[:,nmf_groups_list[k]]@H[nmf_groups_list[k],:]
    filtered_spectrograms.append(filtered_spectrogram)

reconstructed_sounds = []
for k in range(n_nmf_groups):
    reconstruct = filtered_spectrograms[k] * np.exp(1j*sound_stft_Angle)
    new_sound   = librosa.istft(reconstruct, n_fft = n_fft, hop_length = n_hop)
    reconstructed_sounds.append(new_sound)

att_list = []
att_grad_list = []

for k in range(n_nmf_groups):
    features = {}
    model.wav2vec2.encoder.dropout.register_forward_hook(get_features('feats'))

    waveform_nmf = reconstructed_sounds[k]
    output = model(waveform_nmf, output_attentions=True)
    
    logits = output.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    explained_id = true_label
    logits[0, explained_id].backward()
    
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    att_f = output.attentions[-1][0]
    att_f = att_f.cpu().detach().numpy()

    att_f = np.mean(att_f, axis=0)
    att_f = np.mean(att_f, axis=0)
    att_list.append(att_f)

    q_grad = model.wav2vec2.encoder.layers[-1].attention.q_proj.weight.grad
    hidden = features['feats'].to(device)
    q_linear = torch.nn.Linear(q_grad.shape[0], q_grad.shape[1])
    q_linear.weight = torch.nn.Parameter(q_grad)
    q_linear.bias = torch.nn.Parameter(torch.zeros(q_linear.bias.shape))
    q_linear = q_linear.to(device)
    query_states = q_linear(hidden)

    k_grad = model.wav2vec2.encoder.layers[-1].attention.k_proj.weight.grad
    k_linear = torch.nn.Linear(k_grad.shape[0], k_grad.shape[1])
    k_linear.weight = torch.nn.Parameter(k_grad)
    k_linear.bias = torch.nn.Parameter(torch.zeros(k_linear.bias.shape))
    k_linear = k_linear.to(device)
    key_states = k_linear(hidden)

    attn_grads = torch.squeeze(torch.bmm(query_states, key_states.transpose(1, 2)))
    attn_grads = np.mean(attn_grads.cpu().detach().numpy(), axis=0)
    att_grad_list.append(attn_grads)

    model.zero_grad()

#NORMALISATION
att_list = np.array(att_list)
att_grad_list = np.array(att_grad_list)
#ONLY POSITIVE
att_list[att_list<=0] = 0
att_grad_list[att_grad_list<=0] = 0

att_list_norm = (att_list-np.min(att_list))/(np.max(att_list)-np.min(att_list))
att_grad_list_norm = (att_grad_list-np.min(att_grad_list))/(np.max(att_grad_list)-np.min(att_grad_list))

H_att_grad_combined_norm_list = []
for k in range(n_nmf_groups):
    rows_broad = np.linspace(0, att_list[k].shape[0], endpoint=False, num=H.shape[1], dtype=int)

    H_att_w = att_list_norm[k][rows_broad]
    H_att_grad_w = att_grad_list_norm[k][rows_broad]
    H_att_grad_w[H_att_grad_w<=0] = 0
    H_att_grad_combined_norm = H_att_w*H_att_grad_w
    H_att_grad_combined_norm[H_att_grad_combined_norm<=0] = 0
    H_att_grad_combined_norm_list.append(H_att_grad_combined_norm)


#FINAL NORMALISATION
H_att_grad_combined_norm_list = np.array(H_att_grad_combined_norm_list)
H_att_grad_combined_norm_list[H_att_grad_combined_norm_list<=0] = 0
H_att_grad_combined_norm_list = (H_att_grad_combined_norm_list-np.min(H_att_grad_combined_norm_list))/(np.max(H_att_grad_combined_norm_list)-np.min(H_att_grad_combined_norm_list))

#make 0, 1 importance wrt alpha ratio
ind = np.unravel_index(np.argsort(H_att_grad_combined_norm_list, axis=None), H_att_grad_combined_norm_list.shape)
n_t = int(alpha*H_att_grad_combined_norm_list.shape[0]*H_att_grad_combined_norm_list.shape[1])
H_att_grad_combined_norm_01_list = np.ones((H_att_grad_combined_norm_list.shape[0], H_att_grad_combined_norm_list.shape[1]))
H_att_grad_combined_norm_01_list[ind[0][:n_t], ind[1][:n_t]] = 0

#RANDOM shuffleing time relevance
H_att_grad_combined_norm_01_random_list = np.random.permutation(H_att_grad_combined_norm_01_list)
H_att_grad_combined_norm_01_random_list = np.transpose(H_att_grad_combined_norm_01_random_list)
H_att_grad_combined_norm_01_random_list = np.random.permutation(H_att_grad_combined_norm_01_random_list)
H_att_grad_combined_norm_01_random_list = np.transpose(H_att_grad_combined_norm_01_random_list)

for k in range(n_nmf_groups):
    H_att_grad_combined = H_att_grad_combined_norm_01_list[i]
    H_att_ours = H[nmf_groups_list[k],:]*H_att_grad_combined
    H_att_grad_combined = H_att_grad_combined_norm_01_random_list[i]
    H_att_random = H[nmf_groups_list[k],:]*H_att_grad_combined

    if k == 0:
        filtered_spectrogram_nmf_ours = W[:,nmf_groups_list[k]]@H_att_ours
        filtered_spectrogram_nmf_random = W[:,nmf_groups_list[k]]@H_att_random
    else:
        filtered_spectrogram_nmf_ours += W[:,nmf_groups_list[k]]@H_att_ours
        filtered_spectrogram_nmf_random += W[:,nmf_groups_list[k]]@H_att_random


reconstruct_ours = filtered_spectrogram_nmf_ours * np.exp(1j*sound_stft_Angle)
reconstruct_random = filtered_spectrogram_nmf_random * np.exp(1j*sound_stft_Angle)

waveform_recons_ours = librosa.istft(reconstruct_ours, n_fft = n_fft, hop_length = n_hop, length=n)
waveform_recons_random = librosa.istft(reconstruct_random, n_fft = n_fft, hop_length = n_hop, length=n)

waveform_rem_ours = waveform_pad - waveform_recons_ours
waveform_rem_random = waveform_pad - waveform_recons_random

if exp_accs == 1:
    # print("EXPLANATION ACCS")
    inputs_ours = waveform_recons_ours
    inputs_random = waveform_recons_random
else:
    # print("REMAINING ACCS")
    inputs_ours = waveform_rem_ours
    inputs_random = waveform_recons_random

with torch.no_grad():
    output_ours = model(**inputs_ours)
    output_random = model(**inputs_random)


logits = output_ours.logits
predicted_ids_ours = torch.argmax(logits, dim=-1)
if predicted_ids[0]==original_pred:
    print("output from our method matches the original prediction")

logits = output_random.logits
predicted_ids_random = torch.argmax(logits, dim=-1)
if predicted_ids[0]==original_pred:
    print("output from random method matches the original prediction")

