import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import string
from collections import Counter
from transformers import AutoTokenizer
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, csv_path, model_type='rnn', max_length=512, vocab_size=10000, test_mode=False):
        """
        감정 분석을 위한 커스텀 데이터셋 클래스
        
        Args:
            csv_path (str): CSV 파일 경로
            model_type (str): 모델 타입 ('rnn', 'lstm', 'gpt', 'bert')
            max_length (int): 최대 시퀀스 길이
            vocab_size (int): 어휘 사전 크기 (RNN/LSTM용)
            test_mode (bool): 테스트 모드 여부
        """
        self.model_type = model_type.lower()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.test_mode = test_mode
        
        # 데이터 로드
        self.data = pd.read_csv(csv_path)
        self.texts = self.data['text'].tolist()
        self.labels = torch.tensor(self.data['sentiment'].tolist(), dtype=torch.long)
        
        # 모델별 전처리 및 토크나이저 초기화
        if self.model_type in ['rnn', 'lstm']:
            self._setup_traditional_models()
        elif self.model_type == 'gpt':
            self._setup_gpt_model()
        elif self.model_type == 'bert':
            self._setup_bert_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _basic_text_preprocessing(self, text):
        """개선된 텍스트 전처리 - 감정 정보 보존"""
        # 소문자 변환
        text = text.lower()
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL 제거
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 이메일 주소 제거
        text = re.sub(r'\S+@\S+', '', text)
        
        # 감정 표현 보존 - 3개까지 허용 (sooooo → soo)
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        
        # 이모티콘을 특수 토큰으로 변환 (선택적)
        emoji_pattern = r'[😀-🙏💀-🟿]'
        text = re.sub(emoji_pattern, ' <EMOJI> ', text)
        
        # 숫자를 맥락에 따라 처리
        # 점수나 평점 같은 중요한 숫자는 보존
        text = re.sub(r'\b\d{5,}\b', '<NUM>', text)  # 긴 숫자만 토큰화
        
        # 모델별 구두점 처리
        if self.model_type in ['rnn', 'lstm']:
            # 감정 표현에 중요한 구두점 보존
            text = re.sub(r'[^\w\s!?.,;:\'"-]', ' ', text)
            # 연속된 구두점 정규화
            text = re.sub(r'[!]{2,}', '!!!', text)
            text = re.sub(r'[?]{2,}', '???', text)
        
        # 여러 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _setup_traditional_models(self):
        """RNN, LSTM을 위한 개선된 설정"""
        # 텍스트 전처리
        processed_texts = []
        for text in self.texts:
            processed_text = self._basic_text_preprocessing(text)
            # 감정 표현에 중요한 구두점은 보존
            # 완전 제거하지 않고 일부 구두점은 유지
            processed_texts.append(processed_text)
        
        self.processed_texts = processed_texts
        
        if not self.test_mode:
            # 어휘 사전 구축
            self._build_vocabulary()
        else:
            # 테스트 모드에서는 나중에 설정됨
            self.word2idx = None
            self.vocab_size_actual = None
        
        # 텍스트를 인덱스로 변환 (어휘사전이 있는 경우에만)
        if hasattr(self, 'word2idx') and self.word2idx is not None:
            self.encoded_texts = self._encode_texts()
        else:
            self.encoded_texts = None
    
    def _build_vocabulary(self):
        """어휘 사전 구축 (RNN/LSTM용)"""
        # 모든 단어 수집
        all_words = []
        for text in self.processed_texts:
            all_words.extend(text.split())
        
        # 빈도수 기반으로 어휘 선택
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 4)  # 특수 토큰 4개 제외
        
        # 어휘 사전 생성
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size_actual = len(self.word2idx)
    
    def _encode_texts(self):
        """텍스트를 인덱스로 인코딩 (RNN/LSTM용)"""
        if self.word2idx is None:
            raise ValueError("word2idx is not initialized. Make sure vocabulary is built or transferred from training dataset.")
            
        encoded_texts = []
        for text in self.processed_texts:
            words = text.split()
            indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
            
            # 패딩 처리
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]
            else:
                indices.extend([self.word2idx['<PAD>']] * (self.max_length - len(indices)))
            
            encoded_texts.append(indices)
        
        return torch.tensor(encoded_texts, dtype=torch.long)
    
    def set_vocabulary(self, word2idx, vocab_size_actual):
        """테스트 데이터셋용 어휘사전 설정"""
        self.word2idx = word2idx
        self.vocab_size_actual = vocab_size_actual
        if self.model_type in ['rnn', 'lstm']:
            self.encoded_texts = self._encode_texts()
    
    def _setup_gpt_model(self):
        """GPT를 위한 설정"""
        # GPT-2 토크나이저 사용
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 감정 레이블을 텍스트로 변환하여 프롬프트 스타일로 구성
        self.sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # 텍스트 전처리 (구두점 보존 - GPT는 자연어 이해가 좋음)
        processed_texts = []
        for text in self.texts:
            processed_text = self._basic_text_preprocessing(text)
            # GPT용 프롬프트 형태로 구성
            prompt = f"Review: {processed_text} Sentiment:"
            processed_texts.append(prompt)
        
        self.processed_texts = processed_texts
        
        # 토크나이징
        self.encoded_texts = self.tokenizer(
            self.processed_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def _setup_bert_model(self):
        """BERT를 위한 설정"""
        # BERT 토크나이저 사용
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 텍스트 전처리 (BERT는 subword 토크나이징을 하므로 구두점 보존)
        processed_texts = []
        for text in self.texts:
            processed_text = self._basic_text_preprocessing(text)
            processed_texts.append(processed_text)
        
        self.processed_texts = processed_texts
        
        # 토크나이징
        self.encoded_texts = self.tokenizer(
            self.processed_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def get_vocab_size(self):
        """어휘 사전 크기 반환"""
        if self.model_type in ['rnn', 'lstm']:
            return self.vocab_size_actual
        elif self.model_type == 'gpt':
            return self.tokenizer.vocab_size
        elif self.model_type == 'bert':
            return self.tokenizer.vocab_size
    
    def get_word2idx(self):
        """단어-인덱스 매핑 반환 (RNN/LSTM용)"""
        if self.model_type in ['rnn', 'lstm']:
            return self.word2idx
        else:
            return None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.model_type in ['rnn', 'lstm']:
            return {
                'input_ids': self.encoded_texts[idx],
                'labels': self.labels[idx]
            }
        
        elif self.model_type == 'gpt':
            return {
                'input_ids': self.encoded_texts['input_ids'][idx],
                'attention_mask': self.encoded_texts['attention_mask'][idx],
                'labels': self.labels[idx]
            }
        
        elif self.model_type == 'bert':
            return {
                'input_ids': self.encoded_texts['input_ids'][idx],
                'attention_mask': self.encoded_texts['attention_mask'][idx],
                'token_type_ids': self.encoded_texts['token_type_ids'][idx],
                'labels': self.labels[idx]
            }
    
    def get_class_weights(self):
        """클래스 불균형 해결을 위한 가중치 계산"""
        class_counts = torch.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights
    
    def get_statistics(self):
        """데이터셋 통계 정보 반환"""
        text_lengths = [len(text.split()) for text in self.processed_texts]
        class_counts = torch.bincount(self.labels)
        
        stats = {
            'total_samples': len(self.texts),
            'avg_text_length': np.mean(text_lengths),
            'max_text_length': np.max(text_lengths),
            'min_text_length': np.min(text_lengths),
            'class_distribution': {
                'negative': class_counts[0].item(),
                'neutral': class_counts[1].item(),
                'positive': class_counts[2].item()
            }
        }
        
        if self.model_type in ['rnn', 'lstm']:
            stats['vocab_size'] = self.vocab_size_actual
        
        return stats