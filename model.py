import torch


class DiagramEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(DiagramEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.H0_linear = torch.nn.Linear(1, self.embedding_dim)
        self.H1_linear = torch.nn.Linear(2, self.embedding_dim)

    def forward(self, H0, H1, H0_mask, H1_mask):
        H0 = H0.view(H0.size(0), H0.size(1), 1)
        H0_emb = self.H0_linear(H0)
        H1_emb = self.H1_linear(H1)
        return torch.cat([H0_emb, H1_emb], dim=1), torch.cat([H0_mask, H1_mask], dim=1)


class EncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim=128, ff_dim=128, attn_num_heads=8, dropout_prob=0.0):
        super(EncoderBlock, self).__init__()
        
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_num_heads,
                                                     dropout=dropout_prob, batch_first=True)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(ff_dim, embed_dim)
        )

        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

        self.dropout_1 = torch.nn.Dropout(dropout_prob)
        self.dropout_2 = torch.nn.Dropout(dropout_prob)
    
    def forward(self, X, mask):
        X_attn, _ = self.attention(query=X, key=X, value=X, key_padding_mask=mask, need_weights=False)
        X_attn_normalized = self.layer_norm_1(X_attn + X)
        X_ff = self.feedforward(X_attn_normalized)
        X_ff_normalized = self.layer_norm_2(X_attn_normalized + self.dropout_1(X_ff))
        return X + self.dropout_2(X_ff_normalized), mask


class ReZeroEncoder(torch.nn.Module):
    def __init__(self, embed_dim=128, ff_dim=128, attn_num_heads=8, dropout_prob=0.0):
        super(ReZeroEncoder, self).__init__()
        
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_num_heads,
                                                     dropout=dropout_prob, batch_first=True)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ff_dim, embed_dim)
        )

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.rezeroweight = torch.nn.Parameter(torch.Tensor([0]))
    
    def forward(self, X, mask):
        X_attn, _ = self.attention(query=X, key=X, value=X, key_padding_mask=mask, need_weights=False)
        X_attn_normalized = X + X_attn * self.rezeroweight
        X_ff = self.feedforward(X_attn_normalized)
        X_ff_normalized = X_attn_normalized + self.dropout(X_ff) * self.rezeroweight
        return X_ff_normalized, mask


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim=128, attn_num_heads=8, ff_layer_sizes=[128, 256, 256, 64, 5], dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim

        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_num_heads,
                                                     dropout=dropout_prob, batch_first=True)

        ff_layers = [torch.nn.Linear(embed_dim, ff_layer_sizes[0])]
        for layer_idx in range(1, len(ff_layer_sizes)):
            ff_layers.append(torch.nn.GELU()) 
            ff_layers.append(torch.nn.Linear(ff_layer_sizes[layer_idx - 1], ff_layer_sizes[layer_idx]))

        self.feedforward = torch.nn.Sequential(*ff_layers)

    def forward(self, X, mask):
        query = torch.ones((X.size(0), 1, self.embed_dim))
        if next(self.parameters()).is_cuda:
            query = query.to("cuda")
        X_attn, _ = self.attention(query=query, key=X, value=X, key_padding_mask=mask, need_weights=False)
        return self.feedforward(X_attn.view(X.size(0), self.embed_dim))


class Patefon(torch.nn.Module):
    def __init__(self, embedding_dim=128, n_encoders=5, encoder_kwargs={}, decoder_kwargs={}, rezero=False):
        super(Patefon, self).__init__()
        self.embedder = DiagramEmbedder(embedding_dim)

        encoder_type = ReZeroEncoder if rezero else EncoderBlock
        self.encoders = torch.nn.ModuleList([encoder_type(**encoder_kwargs) for i in range(n_encoders)])
        self.decoder = Decoder(**decoder_kwargs)
    
    def forward(self, H0, H1, H0_mask, H1_mask):
        X, mask = self.embedder(H0, H1, H0_mask, H1_mask)

        for encoder in self.encoders:
            X, _ = encoder(X, mask)

        return self.decoder(X, mask)
