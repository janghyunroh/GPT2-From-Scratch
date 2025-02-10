from dataclass import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# Transformer Decoder Block Class
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        # 첫 layer norm
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # attention layer
        # 정보의 압축(reduce)
        self.attn = CausalSelfAttention(config)

        # 두번째 layer norm 
        self.ln2 = nn.LayerNorm(config.n_embd)

        # feed forward layer
        # 단순 매핑(map)
        self.mlp = MLP(config)

        # 위 요소들로 네트워크를 실제로 구성해보자!
        def forward(self, x):

            # 기존 구조: resiudal 분기 -> layer -> resudual 결합 -> layer norm
            # LayerNorm(sublayer(x) + x)

            # 새로운 구조: residual 분기 -> layer norm -> layer -> residual 결합합
            # x + sublayer(LayerNorm(x))

            # 왜 이렇게 하는가?

            # 1. 기울기 소실을 더 잘 해결
            # 가중치 업데이트를 하고나서(레이어를 통과하고 나서) 정규화하는 건 그 레이어한테 무쓸모!
            # 정규화 하고나서 레이어를 통과시키면 해당 레이어가 학습 안정성이 높음!

            # 2. residual path와 non-residual path의 명확한 구분
            # residual path 쪽은 정규화가 안되므로 원본 정보 보존 가능!
            # residual connection의 의의를 잘 지킴!

            x = x + self.attn(self.ln_1(x)) 
            x = x + self.mlp(self.ln_2(x))
            return x





# GPT 설정 정보 class
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head : int = 6
    n_embd: int = 384

# GPT class
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
            # Block은 추후 위에서 정의할 예정!
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # ln_f : 토큰 생성 전 layer normalization
            # 원래 AIAYN 논문에는 없었으며, GPT 2 논문에서 layer norm을 추가함. 
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # lm_head: softmax 직전 linear layer
        # 768차원의 임베딩 벡터를 vocab의 모든 토큰에 대한 가중치로 변환환
        # 이 값을 softmax에 통과시켜 각 토큰 별 확률로 변환
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
