from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# GPT
# - 토큰 임베딩
# - 위치 임베딩
# - 디코더 블럭 x 6
#   - layerNorm
#   - 어텐션
#   - MLP
# - Linear
# - SoftMax


# Decoder의 Attention Layer Class
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # head 개수만큼 임베딩 벡터를 쪼갤 수 있는지 확인
        assert config.n_embd % config.n_head == 0

        # q, k, v 벡터를 그냥 한번에 이어서 만들어버리는 레이어
        # 이후 이 레이어로 만든 벡터를 3등분해서 각각 q, k, v로 쓸 예정
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        # 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Decoder 생성을 위한 마스킹
        # register_buffer 함수: forward처럼 pytorch nn.Module에 처음부터 존재하는 요소.
        # 모델 내부적으로 가지고 있으면서, 학습은 필요없는 텐서를 저장하는데에 주로 쓰임. 
        # 여기서는 마스킹 텐서를 저장
        # 왜 이름을 bias로 지었는지는 의문임(그냥 논문에서 그렇게 지음)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):

        # 입력 텐서의 shape: Batch x Sequence len x Channel(토큰 별 임베딩 벡터 크기)
        # x : (B, T, C)
        B, T, C = x.size()

        qkv = self.c_attn(x) # qkv : (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # C를 n_head개만큼 쪼개기. 
        # (B, T, C)였던 놈의 C를 n_head개만큼 쪼개 (B, T, hs)가 nh개 있다고 생각하는게 좋음
        # (C = nh x hs)

        # 단, 실제로는 연산을 위해 shape을 (B, nh, T, hs)로 둠!
        # 하지만 이걸 그대로 떠올리기엔 어려우므로
        # 생각하기 편하게 (B, T, hs) x nh로 생각해도 됨!

        # C를 쪼개고나서 T열과 nh열을 바꿈(transpose)
        # 바꾸는 이유는 nh가 batch dimension처럼 작동하도록 하기 위함!
        # 즉, B x nh 개의 (T, hs)가 존재하는 것처럼 연산하기 위함
        # 이렇게 하면 pytorch가 B와 nh에 대해 병렬로 연산함
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention = Softmax(q . k / sqrt(hs) )
        # 내적 후 정규화
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 계산한 attention을 masking table을 이용해 masking
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax로 확률 분포로 변환, 최종 어텐션 계산
        att = F.softmax(att, dim=-1)

        # 계산한 어텐션을 value에 곱함으로써 최종 output 계산
        y = att @ v

        # (B, T, C) shape으로 복구
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # 마지막으로 Linear 한번 거침
        y = self.c_proj(y)

        return y

# Decoder Block 내부의 Feed-Forward Layer Class
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # 기존 임베딩 -> 4배 확장장
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU 활성화 함수 정의
        # 2가지 버전이 존재. 

        # 예전에는 tensorflow로 GELU 구현 시 사용한 erf 함수가 느렸음!
        # 그래서 GPT 개발 당시에는 tanh를 이용한 근사함수를 썼음

        # 현재 실제 구현에는 그럴 필요가 없지만 논문 재현을 위해 근사 함수 사용용

        # 자세히는 GELU 논문 살펴볼 것!
        self.gelu = nn.GELU(approximate='tanh')

        # 4배 확장 -> 기존 임베딩
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
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
        self.ln_2 = nn.LayerNorm(config.n_embd)

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
        # residual path 쪽은 정규화가 안되므로 원본 정보 보존 가능! -> clean residual path
        # residual connection의 의의를 잘 지킴!

        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x





# GPT 설정 정보 class
@dataclass
class GPTConfig: # GPT-2에 맞게 설정
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head : int = 12
    n_embd: int = 768

# GPT class
class GPT(nn.Module):

    # 클래스 생성자자
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

    
    # forward 함수 정의
    def forward(self, idx): #idx: (B, T) shape의 입력
        
        # 입력 크기
        B, T = idx.size()

        # Block Size에 맞게 입력이 들어왔는지 확인
        assert T <= self.config.block_size, f"{T}길이의 시퀀스는 Forwarding이 불가능합니다. Block size는 {self.config.block_size}이며 입력은 이보다 작아야 합니다. "

        # 위치 임베딩과 토큰 임베딩 생성

        # 위치에 따라 증가하는 배열 생성
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)

        # pos 배열로 위치 임베딩 생성
        pos_emb = self.transformer.wpe(pos) # (T, C), C = n_embd

        # idx로 토큰 임베딩 생성
        tok_emb = self.transformer.wte(idx) # (B, T, C)

        # 최종 임베딩 생성
        x = tok_emb + pos_emb # (B, T, C)

        # 디코더 블럭 통과
        for block in self.transformer.h:
            x = block(x) # (B, T, C)
        
        # 마지막 Layer Norm 통과
        x = self.transformer.ln_f(x) # (B, T< C)

        # 마지막 Linear Layer 통과
        logits = self.lm_head(x) # (B, T, Vocab Size)

        return logits # 이 logit을 softmax 통과시키고 vocab lookup해서 토큰 생성

    # pre-train 가중치 불러오는 함수
    # 길긴 하지만 GPT 공부에 그리 중요한 코드는 아닙니다. 
    # model_type이 주어진 경우의 GPT 모델 생성자. 
    @classmethod
    def from_pretrained(cls, model_type):
        """HuggingFace의 GPT-2 모델 가중치를 로드합니다."""

        # 우리가 불러올 모델 타입이 이 중에 해당하는지 확인
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel

        print(f'pre-trained gpt {model_type}(으)로부터 가중치 로드 중...')

        # 모델 타입 별 설정 변수 지정
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        # 50000 + 256 + 1
        # 50000: 50000개로 압축
        # 256: byte fallback 위한 자리
        # 1 : 특수 토큰 <|endoftext|> 자리
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # 위 설정변수를 이용해 GPT 모델 생성
        config = GPTConfig(**config_args) # 
        model = GPT(config)

        # ------------ state_dict 복사해오기(일부 버퍼 제외) ------------

        # 우리 모델을 위한 state_dict
        sd = model.state_dict()
        sd_keys = sd.keys()

        # 생성한 dict 원소 중 attn_bias는 우리가 직접 만들었으므로 제외. 
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # HuggingFace의 GPT 모델 로드 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        
        # HuggingFace 모델의 state_dict
        sd_hf = model_hf.state_dict()

        # HuggingFace model의 state_dict를 우리 모델의 state_dict로 복사
        sd_keys_hf = sd_hf.keys()
        
        # 마스킹 테이블 제외
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        
        # 얘도 같은 이유로 제외
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # ------------ 추가 작업: transpose 된 weight 되돌리기 ------------
        # HuggingFace에선 tensorflow로 개발되어 있어서 
        # pytorch와 비교했을 때 transpose되어 있는 가중치가 있음!(Conv1D 모듈 때문)
        # 이를 pytorch에 맞게 하기 위해 일부 weight 전치시키기

        # 하드코딩으로 직접 전치할 가중치 선택
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        # state_dict의 각 원소 복사
        for k in sd_keys_hf:

            # 뒤집어야 하는 가중치의 경우
            if any(k.endswith(w) for w in transposed):

                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t()) # 뒤집어서 복사
            
            # 안 뒤집어도 되는 경우 
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k]) # 그대로 복사

        
        return model




# 직접 문장 생성해보기
if __name__ == '__main__':

    # 생성할 문장 수와 문장 별 생성할 토큰 수 지정.
    num_return_sequences = 5
    max_length = 30

    # device 결정 (cpu / gpu / mps)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'사용 디바이스: {device}')

    # 모델 불러오기 & 추론 모드로 설정
    #model = GPT.from_pretrained('gpt2') # 사전학습된 가중치 로드하여 생성
    model = GPT(GPTConfig()) # 기본 설정으로 램덤 초기화 모델 생성, 이걸 그대로 쓰면 결과 엉망!
    model.eval()
    model.to(device)
    print(f'모델 설정: {model.config}')

    # 문장 준비 & 인코딩
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")

    # torch tensor로 만들기
    tokens = torch.tensor(tokens, dtype=torch.long)  #(8)

    # 5개로 복사
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)

    # device로 옮기기
    x = tokens.to(device) # x : (5, 8)

    # -------------- 생성 시작! --------------

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 토큰 생성
    while x.size(1) < max_length: # T < max_length

        with torch.no_grad():

            # logit 계산
            logits = model(x) # (B=5, T=1, 2, ...(루프마다 다름), Vocab_size)

            # T방향의 마지막 logit(가장 최근 토큰)만 가져옴
            # 상당히 비효율적인 샘플링이긴 함...
            logits = logits[:, -1, :] # (5, 1, Vocab_size)
            
            # Softmax로 확률 추출
            probs = F.softmax(logits, dim=-1) # (5, Vocab_size)

            # 확률이 높은 상위 50개 토큰만 추출
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (5, 50)확률값, (5, 50) 인덱스

            # 상위 50개의 확률값을 가지고 재정규화(합이 1이 되도록)한 뒤 
            # 시뮬레이션으로 하나 뽑기
            # 함수 결과는 50개에서 뽑은 하나의 인덱스(Vocab의 인덱스 아님!!!)
            # multinimail 함수에 대한 자세한 건 공식 문서를 확인할 것!
            ix = torch.multinomial(topk_probs, 1) # (5, 1)

            # (5, 50) 인덱스 텐서에서 50짜리 방향에서 ix위치의 인덱스 추출
            xcol = torch.gather(topk_indices, -1, ix) # (5, 1)
            
            # x의 뒤에 이어붙이기
            x = torch.cat((x, xcol), dim=1) 

    
    # 생성한 토큰들 출력
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print('>', decoded)
