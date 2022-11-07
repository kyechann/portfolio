#LED_KoBART 모델

KoBART 모델은 한국어 특화의 딥러닝 요약모델이다.
생성 및 요약으로는 Decoder 기반의 모델로 뉴스 등 다양한 글의 요약으로는 KoBART만한 모델이 없다.
다만, 최대 수용 가능한 문장의 길이는 512 토큰으로 뉴스 기사와 같이 긴 글은 분리해서 학습은 진행하면 왜곡되는 문장이나 단어가 생기기 마련이다.
그걸 방지하기 위해서 Longformer를 사용했다.
Longformer는 문장 길이가 4096으로 KoBART에 비하면 상당히 긴 것을 알 수 있다.
그렇기 때문에 긴 내용의 문서를 입력 받아도 충분히 데이터로 사용 가능하다.

<<폴더 소개>>

##LED_KoBART 폴더 : 모델 생성하고 학습을 진행하여 구현한 코드를 Pytorch-lightning으로 바꿨으며, WanDB, Mlflow, Tensorboard로 학습을 tracking하고 Loss 등을 시각화하여 보여주는 것을 구현하였다.

##MDS 폴더 : 다중문서요약(Multi-Document-Summarization) 모델 생성 및 요약 진행관련 폴더로 PRIMERA 모델로 진행했다.
