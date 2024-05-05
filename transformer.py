import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

def scaled_softmax_attention(query, key, value):
    """
    Args:
        query: torch.Tensor (..., L, D)
        key: torch.Tensor (..., L, D)
        value: torch.Tensor (..., L, D)
    Returns:
        res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
        attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

    L is the length of sequence, D is the embedding dimension
    """
    QK = torch.matmul(query, torch.transpose(key, -2, -1))
    QK /= torch.sqrt(torch.tensor(query.shape[-1]))

    attention = F.softmax(QK, dim=-1)

    res = torch.matmul(attention, value)
    return res, attention
  
  
class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # W_i_Q = d_model x d_k
        # W_i_K = d_model x d_k
        # W_i_V = d_model x d_v
        # W_O   = h * d_v x d_model

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)


    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        L = x.shape[1]
        batch_len = x.shape[0]
        outputs, attention = None, None

        Q = self.q_proj(x).reshape((batch_len, L, self.num_heads, self.head_dim))
        K = self.k_proj(x).reshape((batch_len, L, self.num_heads, self.head_dim))
        V = self.v_proj(x).reshape((batch_len, L, self.num_heads, self.head_dim))

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        outputs, attention = scaled_softmax_attention(Q, K, V)

        outputs = outputs.transpose(1,2).reshape((batch_len, L, self.embed_dim))
        outputs = self.o_proj(outputs)

        if return_attention:
            return outputs, attention
        else:
            return outputs
          
          
          
class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, feedforward_dim, activation=nn.ReLU, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)


        self.multihead = MultiheadAttention(embed_dim, num_heads)
        self.activation = activation
        self.feedforward = nn.Sequential(*[
            nn.Linear(embed_dim, feedforward_dim),
            nn.Dropout(dropout),
            self.activation(),
            nn.Linear(feedforward_dim, embed_dim)
        ])

        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # TODO

    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)
        """
        residual = x
        if return_attention:
            outputs, attention = self.multihead(x, return_attention=return_attention)
        else:
            outputs = self.multihead(x)
        outputs = self.dropout1(outputs)
        outputs = self.layernorm1(outputs + residual)
        
        residual2 = outputs

        outputs = self.feedforward(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.layernorm2(outputs + residual2)

        

        if return_attention:
            return outputs, attention
        else:
            return outputs

          
          
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition
        pe = torch.zeros((1, max_len, embed_dim)).float()
        positions = torch.arange(0, max_len).float()
        positions = positions.unsqueeze(1)
        i_s = torch.arange(0, embed_dim, 2).float()

        pe[:,:, ::2] = torch.sin(positions / torch.pow(10000, i_s / embed_dim) )
        pe[:,:, 1::2] = torch.cos(positions / torch.pow(10000, i_s / embed_dim) )
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x +  self.pe[:, :x.shape[1]]
        return x
      
class TransformerForSequenceClassification(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        activation = nn.GELU,
        max_len: int = 5000,
        dropout: float = 0.0
    ):
        super().__init__()
        # define layers
        self.cls_token = torch.randn(embed_dim) # TODO create vector of size (embed_dim,) from N(0, 1)
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len) # TODO

        encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, feedforward_dim, activation, dropout) for i in range(num_layers)])
        self.encoder = encoder_blocks

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor (B, L, |V|)
        Returns:
            x: torch.Tensor (B, |C|)
        """
        
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = torch.cat((x, self.cls_token.repeat(x.shape[0], 1, 1)), dim=1)
        
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x, return_attention=False)
            
        x = self.classifier(x[:, -1, :])

        return x
