import torch
import torch.nn as nn
from Self_Regression.decoder import Decoder, DecoderLayer
from Self_Regression.attn import FullAttention, ProbAttention, AttentionLayer
from Self_Regression.embed import DataEmbedding
from GBT.Auto_Regression import AR
from FEDformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from FEDformer.Autoformer_EncDec import Auto_Encoder, Auto_EncoderLayer, Auto_Decoder, Auto_DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from utils.RevIN import RevIN

class GBT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, auto_d_layers=1, dropout=0.0, attn='prob', time=True, activation='gelu', output_attention=False, distil=True, mix=True, feature_extractor='Attention', kernel=3, fd_model=64, moving_avg=[24], instance=False, use_RevIN=False, format='transformer', device=torch.device('cuda:0')):
        super().__init__()
        self.pred_len, self.label_len = out_len, label_len
        self.attn, self.output_attention, self.n_heads, self.format = attn, output_attention, n_heads, format
        self.group = 1
        if instance: enc_in = dec_in = c_out = 1
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout, True, time, group=self.group)
        self.use_RevIN = use_RevIN
        if use_RevIN: self.revin = RevIN(enc_in)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        if format == 'transformer':
            self.AR = AR(enc_in, c_out, label_len, out_len, feature_extractor, kernel=kernel, group=not mix, block_nums=e_layers[0], time=time, fd_model=fd_model, sd_model=d_model, pyramid=len(e_layers), dropout=dropout)
            self.decoder = Decoder([DecoderLayer(AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), d_model, self.n_heads, mix=mix, group=self.group), d_model, dropout=dropout, activation=activation, group=self.group) for _ in range(d_layers)], norm_layer=nn.LayerNorm(d_model), projection=nn.Linear(d_model, c_out, bias=True))
        elif format == 'autoformer':
            self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, position=False, time=time, group=self.group)
            kernel_size = moving_avg
            self.decomp = series_decomp_multi(kernel_size) if isinstance(kernel_size, list) else series_decomp(kernel_size)
            self.decomp2 = series_decomp_multi(kernel_size) if isinstance(kernel_size, list) else series_decomp(kernel_size)
            self.encoder = Auto_Encoder([Auto_EncoderLayer(AutoCorrelationLayer(AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads), d_model, moving_avg=moving_avg, dropout=dropout, activation=activation, trend=False) for _ in range(e_layers[0])], norm_layer=my_Layernorm(d_model))
            self.decoder_auto = Auto_Decoder([Auto_DecoderLayer(AutoCorrelationLayer(AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads), AutoCorrelationLayer(AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads), d_model, c_out, moving_avg=moving_avg, dropout=dropout, activation=activation) for _ in range(auto_d_layers)], norm_layer=my_Layernorm(d_model), projection=nn.Linear(d_model, c_out, bias=True))
        else: raise ValueError('format must be transformer or autoformer')
