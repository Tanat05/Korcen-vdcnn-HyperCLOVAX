<div align="center">
  <h1>Korcen-vdcnn-HyperCLOVAX</h1>
  <h2>a new attempt</h2>
</div>

![131_20220604170616](https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png)

[Korcen](https://github.com/KR-korcen/korcen): original before innovation.

[Korcen-kogpt2](https://github.com/Tanat05/korcen-kogpt2): First innovation and first failure

# Example PY: 3.10 TF: 2.10
```py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import colorama
from tensorflow.keras.layers import Layer, Attention

colorama.init()

tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = Attention()
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        return self.attention([x, x])

    def compute_output_shape(self, input_shape):
        return input_shape


model = load_model('vdcnn_model_best HyperCLOVAX.h5', custom_objects={'SelfAttention': SelfAttention})

while True:
    text = str(input(colorama.Fore.RESET + "테스트할 문장을 입력하세요: "))
    
    text = text.lower()
    encoded = tokenizer.encode(text, max_length=128, truncation=True, padding="max_length")
    seq = np.array([encoded])
    
    prediction = model.predict(seq)[0][0]

    if prediction <= 0.3:
        print(str(prediction) + ": " + colorama.Fore.GREEN + text + colorama.Fore.RESET)
    elif prediction > 0.3 and prediction <= 0.9:
        print(str(prediction) + ": " + colorama.Fore.RESET + text + colorama.Fore.RESET)
    else:
        print(str(prediction) + ": " + colorama.Fore.RED + text + colorama.Fore.RESET)
```
