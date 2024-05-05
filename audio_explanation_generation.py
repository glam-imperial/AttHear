import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
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



if __name__ == '__main__':
    device = 'cuda'

    valid_data = "YOUR DATASET HERE"
    model = "YOUR MODEL HERE"

    n_fft = 512
    n_components = 20
    n_nmf_groups = 5
    n_stacked_frames = 1 
    b_div = 1
    sparsity_weight = 0
    tol = 1e-5
    a_dtype = 'f32'
    dtype = torch.float32
    n_workers = 2
    out_dir = "./output/"
    n_hop = 256
    sr = 16000
    explained_id = "should be true label id"

    F = (n_fft // 2 + 1) * n_stacked_frames
    K = n_components
    b = b_div

    ix = 0

    
    waveform, sr = librosa.load("/audio.wav", sr=16000)
    for V in [waveform]:
        if V.shape[1]<80000:
            V  = np.pad(V, ((0, 0), (0, 80000 - V.shape[1]%80000)), 'constant')
        elif V.shape[1]>80000:
            V = V[:,0:80000]

        sound_stft = librosa.stft(np.array(V)[0], n_fft = n_fft, hop_length = n_hop) 
        # Magnitude Spectrogram
        sound_stft_Magnitude = np.abs(sound_stft)
        # Phase spectrogram
        sound_stft_Angle = np.angle(sound_stft)
        #Plot Spectogram
        Spec = librosa.amplitude_to_db(sound_stft_Magnitude, ref = np.max)
        input_spec_librosa = librosa.display.specshow(Spec,y_axis = 'hz',sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet)

        V = sound_stft_Magnitude + 1e-10
        beta = 2
        S = n_components
        # Applying the NMF function
        W, H, cost_function = NMF(V,S,beta = beta, threshold = 0.05, MAXITER = 5000) 

    #KMEANS CLUSTERING
    component_labels = KMeans(n_clusters=n_nmf_groups, random_state=0, n_init="auto").fit_predict(np.transpose(W))
    nmf_groups_list = []
    for i in range(n_nmf_groups):
        group_ids = np.where(component_labels == i)[0]
        nmf_groups_list.append(group_ids)

    ncols=5
    nrows=int(np.ceil(S/ncols)+1)

    f, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(20,20))
    filtered_spectrograms = []
    for i in range(n_nmf_groups):
        ax_i = int(i/ncols)
        ax_j = int(i%ncols)
        axs[ax_i,ax_j].set_title(f"Frequency Mask of Audio Source s = {i+1}") 
        # Filter each source components
        filtered_spectrogram = W[:,nmf_groups_list[i]]@H[nmf_groups_list[i],:]
        # Compute the filtered spectrogram
        D = librosa.amplitude_to_db(filtered_spectrogram, ref = np.max)
        # Show the filtered spectrogram
        librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet, ax = axs[ax_i,ax_j])
        filtered_spectrograms.append(filtered_spectrogram)

    filtered_spectrogram_all = W@H
    D = librosa.amplitude_to_db(filtered_spectrogram_all, ref = np.max)
    # Show the filtered spectrogram
    axs[nrows-1,0].set_title(f"Frequency Mask of Audio Source s = all") 
    librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet, ax = axs[nrows-1,0])
    axs[nrows-1,1].set_title(f"original Audio Source") 
    librosa.display.specshow(Spec,y_axis = 'hz',sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet, ax = axs[nrows-1,1])

    filtered_spectrograms.append(filtered_spectrogram)
    plt.savefig("./nmf_librosa_reconstructed_specs.png")

    reconstructed_sounds = []
    for i in range(n_nmf_groups):
        reconstruct = filtered_spectrograms[i] * np.exp(1j*sound_stft_Angle)
        new_sound   = librosa.istft(reconstruct, n_fft = n_fft, hop_length = n_hop)
        reconstructed_sounds.append(new_sound)
    
    #nmf reconstructed with all sources
    reconstruct = filtered_spectrogram_all * np.exp(1j*sound_stft_Angle)
    new_sound   = librosa.istft(reconstruct, n_fft = n_fft, hop_length = n_hop)
    reconstructed_sounds.append(new_sound)
    #original
    reconstruct = V * np.exp(1j*sound_stft_Angle)
    new_sound   = librosa.istft(reconstruct, n_fft = n_fft, hop_length = n_hop)
    reconstructed_sounds.append(new_sound)

    colors = ['r', 'g','b']
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(20, 20))
    for i in range(n_nmf_groups):
        ax_i = int(i/ncols)
        ax_j = int(i%ncols)
        librosa.display.waveshow(reconstructed_sounds[i], sr=sr, color = colors[int(i%3)], ax=ax[ax_i,ax_j],label=f'Source {i}',x_axis='time')
        ax[ax_i,ax_j].set(xlabel='Time [s]')
        ax[ax_i,ax_j].legend()  
    librosa.display.waveshow(reconstructed_sounds[n_nmf_groups], sr=sr, color = colors[int(i%3)], ax=ax[nrows-1,0],label=f'nmf reconstructed',x_axis='time')
    ax[nrows-1,0].set(xlabel='Time [s]')
    ax[nrows-1,0].legend() 
    librosa.display.waveshow(reconstructed_sounds[n_nmf_groups+1], sr=sr, color = colors[int(i%3)], ax=ax[nrows-1,1],label=f'original',x_axis='time')
    ax[nrows-1,1].set(xlabel='Time [s]')
    ax[nrows-1,1].legend()  

    plt.savefig("./nmf_librosa_reconstructed_waveform.png")

    for idx, sound in enumerate(reconstructed_sounds):
        scipy.io.wavfile.write(f'./source_{idx}.wav', sr, np.float32(sound))

    
    model = model.to(device)
    model.eval()
    att_list = []
    att_grad_list = []

    for i in range(n_nmf_groups):
        features = {}
        model.wav2vec2.encoder.dropout.register_forward_hook(get_features('feats'))

        waveform_nmf = reconstructed_sounds[i]
        output = model(waveform_nmf, output_attentions=True)
        logits = output.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        logits[0,explained_id].backward()
        
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
    att_list[att_list<=0] = 0
    att_grad_list[att_grad_list<=0] = 0

    att_list = (att_list-np.min(att_list))/(np.max(att_list)-np.min(att_list))
    att_grad_list = (att_grad_list-np.min(att_grad_list))/(np.max(att_grad_list)-np.min(att_grad_list))

    fig, ax = plt.subplots(nrows=nrows*4-4, ncols=ncols, figsize=(20, 20))
    for i in range(n_nmf_groups):
        ax_i = int(i/ncols)*4
        ax_j = int(i%ncols)
        ax[ax_i,ax_j].set_title(f'NMF source {i} attention')
        rows_broad = np.linspace(0, att_list[i].shape[0], endpoint=False, num=16000, dtype=int)
        broadcasted_att = att_list[i][rows_broad]
        broadcasted_att_grad = att_grad_list[i][rows_broad]
        ax[ax_i+1,ax_j].sharex(ax[ax_i,ax_j])
        ax[ax_i+2,ax_j].sharex(ax[ax_i,ax_j])
        ax[ax_i+3,ax_j].sharex(ax[ax_i,ax_j])
        ax[ax_i+1,ax_j].sharey(ax[ax_i,ax_j])
        ax[ax_i+2,ax_j].sharey(ax[ax_i,ax_j])
        ax[ax_i,ax_j].plot(broadcasted_att)
        ax[ax_i+1,ax_j].plot(broadcasted_att_grad)
        ax[ax_i+2,ax_j].plot(broadcasted_att*broadcasted_att_grad)
        ax[ax_i+3,ax_j].plot(reconstructed_sounds[-1])
    plt.savefig("./attention_grad_nmfsources.png")

    H_att_grad_combined_list = []
    for i in range(n_nmf_groups):
        rows_broad = np.linspace(0, att_list[i].shape[0], endpoint=False, num=H.shape[1], dtype=int)
        H_att_w = att_list[i][rows_broad]
        H_att_grad_w = att_grad_list[i][rows_broad]
        H_att_grad_w[H_att_grad_w<=0] = 0
        H_att_grad_combined = H_att_w*H_att_grad_w 
        H_att_grad_combined[H_att_grad_combined<=0] = 0
        H_att_grad_combined_list.append(H_att_grad_combined)

    #FINAL NORMALISATION
    H_att_grad_combined_list = np.array(H_att_grad_combined_list)
    H_att_grad_combined_list[H_att_grad_combined_list<=0] = 0
    H_att_grad_combined_list = (H_att_grad_combined_list-np.min(H_att_grad_combined_list))/(np.max(H_att_grad_combined_list)-np.min(H_att_grad_combined_list))
    for i in range(n_nmf_groups):
        H_att_grad_combined = H_att_grad_combined_list[i]
        H_att = H[nmf_groups_list[i],:]*H_att_grad_combined
        if i == 0:
            filtered_spectrogram_nmf = W[:,nmf_groups_list[i]]@H_att  
        else:
            filtered_spectrogram_nmf += W[:,nmf_groups_list[i]]@H_att 

    reconstruct = filtered_spectrogram_nmf * np.exp(1j*sound_stft_Angle)
    nmf_weighted_reconst = librosa.istft(reconstruct, n_fft = n_fft, hop_length = n_hop)
    scipy.io.wavfile.write('./explanation_audio.wav') 
    
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    sound_stft = np.abs(librosa.stft(waveform, n_fft = n_fft, hop_length = n_hop))
    D = librosa.amplitude_to_db(sound_stft, ref = np.max)
    librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet, ax = ax[0])
    D = librosa.amplitude_to_db(filtered_spectrogram_nmf, ref = np.max)
    librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=n_hop,x_axis ='time',cmap= matplotlib.cm.jet, ax = ax[1])
    plt.savefig("./weighted_explanation.png")


