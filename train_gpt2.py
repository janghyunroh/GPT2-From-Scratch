from dataclass import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head : int = 6
    n_embd: int = 384

class GPT(nn.module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict: 모듈 딕셔너리
        # 하위 모듈들을 key를 이용해 접근 가능하게 함
        self.transformer = nn.ModuleDict(dict(

            # wte: weights of token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            # wpe: weights of positional embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            
            # h: hidden layer
            # GPT에 사용된 Transformer Decoder Block을 의미!
            # Block은 나중에 정의됨!
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # ln_f : 토큰 생성 전 layer normalization
            # 원래 AIAYN 논문에는 없었으며, GPT 2 논문에서 layer norm을 추가함. 
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # lm_head: softmax 직전 linear layer
        # 768차원의 임베딩 벡터를 vocab의 모든 토큰에 대한 가중치로 변환환
        # 이 값을 softmax에 통과시켜 각 토큰 별 확률로 변환
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
