import torch
import timm
import torch.nn.functional as F
import math 

from torch import nn
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        # Define linear layers for Q, K, V transformations
        self.query_layer = nn.Linear(input_dim, hidden_dim).to('cuda')
        self.key_layer = nn.Linear(input_dim, hidden_dim).to('cuda')
        self.value_layer = nn.Linear(input_dim, hidden_dim).to('cuda')
        
        # Scaling factor to normalize attention scores
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to('cuda')

    def forward(self, x):
        # x: [11, 2000]
        # Step 1: Project input to Q, K, V
        Q = self.query_layer(x)   # [11, hidden_dim]
        K = self.key_layer(x)     # [11, hidden_dim]
        V = self.value_layer(x)   # [11, hidden_dim]

        # Step 2: Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [11, 11]
        attention_scores = attention_scores / self.scale         # Scale scores
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [11, 11]

        # Step 4: Compute the attention output
        attention_output = torch.matmul(attention_weights, V)    # [11, hidden_dim]

        return attention_output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by the number of heads"

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Input embedding dimension mismatch"

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_length, embed_dim)
        K = self.k_proj(x)  # (batch_size, seq_length, embed_dim)
        V = self.v_proj(x)  # (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_length, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)  # (batch_size, seq_length, embed_dim)

        return output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPosEmb, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, position):
        """
        Get the sinusoidal positional embedding for a single position or a sequence of positions.
        Args:
            position (int, Tensor): Position index or tensor of positions.
        Returns:
            Tensor: Positional embedding(s) with shape (embedding_dim,).
        """
        if isinstance(position, int):  # If position is an integer, convert it to tensor
            position = torch.tensor([position], dtype=torch.float32).to('cuda')

        # Ensure position is on the correct device if needed

        # Calculate the div_term for the embedding frequencies
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device='cuda') *
            -(math.log(10000.0) / self.embedding_dim)
        )

        # Generate the positional embedding
        pos_emb = torch.zeros(self.embedding_dim, device='cuda')
        pos_emb[0::2] = torch.sin(position * div_term)  # Even indices
        pos_emb[1::2] = torch.cos(position * div_term)  # Odd indices

        return pos_emb

class SequentialMotionCNN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # x, y, sigma_xx, sigma_yy, sigma_xy, sigma_yx, visability 
        n_components = 7
        n_modes = model_config['n_modes']
        n_timestamps = model_config['n_timestamps']

        self.procecss_key_list = ['time_0', 'time_1', 'time_2', 'time_3', 'time_4',
                                  'time_5', 'time_6', 'time_7', 'time_8', 'time_9',
                                  'time_10']

        output_dim = n_modes + n_modes * n_timestamps * n_components
        cnn_model_output_dim = int(n_modes + n_modes * n_timestamps * n_components) # hyperparameter to tune
        
        self.cnn_model = timm.create_model(model_config['backbone'], 
                                           pretrained=True, 
                                           in_chans=3, 
                                           num_classes=cnn_model_output_dim).to('cuda')  # 11 * cnn_model_output_dim 
        
        self.time_pos_embedding = SinusoidalPosEmb(cnn_model_output_dim).to('cuda')
        
        self.self_attention_layer = SelfAttention(input_dim=cnn_model_output_dim, hidden_dim=cnn_model_output_dim).to('cuda')

        self.multihead_attention_layer = MultiheadSelfAttention(embed_dim=cnn_model_output_dim, num_heads=n_modes)
        
        self.fc_layer1 = nn.Linear(11 * cnn_model_output_dim, output_dim).to('cuda')

        self.fc_layer2 = nn.Linear(output_dim, output_dim).to('cuda')

        self.layer_norm = nn.LayerNorm(normalized_shape=3366)
        
        
    def forward(self, data):

        processed_cnn_output = None

        for index in range(len(self.procecss_key_list)):

            key = self.procecss_key_list[index]
            
            img = data[key].permute(0, 3, 1, 2).float().to('cuda')
            
            cnn_output = self.cnn_model(img)
            
            time_embedding = self.time_pos_embedding(index).to('cuda')
            
            output = cnn_output + time_embedding
            
            output = output.to('cuda')
            
            if processed_cnn_output is None:

                processed_cnn_output = output.unsqueeze(0)
            
            else:
                
                processed_cnn_output = torch.cat((processed_cnn_output, output.unsqueeze(0)), dim=0)

        processed_cnn_output = processed_cnn_output.to('cuda')

        atten_output = self.multihead_attention_layer(processed_cnn_output).to('cuda')

        activated_atten_output = F.relu(atten_output).to('cuda')

        residual_output = processed_cnn_output + activated_atten_output 

        residual_output = residual_output.permute(1,0,2)

        #print("residual output shape: ", residual_output.shape)

        normalized_atten_output = self.layer_norm(residual_output).to('cuda')

        #print("after norm: ", normalized_atten_output.shape)
        
        b,h,w = normalized_atten_output.shape

        normalized_atten_output = normalized_atten_output.reshape(b, h*w).to('cuda')

        #print("normal second: ", normalized_atten_output.shape)
        
        fc1 = self.fc_layer1(normalized_atten_output).to('cuda')

        #print("fc1: ", fc1.shape)

        fc1 = F.relu(fc1)
        
        fc2 = self.fc_layer2(fc1).to('cuda')

        #print("fc2: ", fc2.shape)

        return fc2
        
        
        
        
        
        
        
        
        