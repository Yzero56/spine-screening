# spine-screening

[데이터 라벨링]
1. Ollama를 다운로드하여 로컬 환경에서도 라벨링 작업을 수행할 수 있도록 합니다.
2. label_with_llm.py 파일을 로컬에 다운로드하여 데이터 라벨링을 실시합니다.

[모델학습]
1. kaggle에서 데이터셋을 다운로드 합니다. 
데이터셋 : Lumvar Spinal MRI Dataset (https://www.kaggle.com/datasets/abdullahkhan70/lumbar-spinal-mri-dataset/)
(* 주의⚠️ : 다운로드한 폴더를 프로젝트 루트에 위치시킵니다.)

2. stenosis_self.ipynb를 다운로드하여 실행합니다.
   * stenosis_self.ipynb = Simple_CNN을 사용한 모델
     
3. stenosis_supplement.ipynb를 다운로드하여 실행합니다.
   * stenosis_supplement.ipynb = 데이터 증강 & 사전 학습 백본(ResNet50) 사용  

[모델 학습 시 설치해야할 모듈 목록] 
- 아래 명령어를 CMD 또는 터미널에 복사·붙여넣기 하시면 됩니다.

**pip install numpy Pillow tensorflow tensorflow-datasets scikit-learn matplotlib**

*numpy : 수치 연산, 배열 처리

*Pillow : 이미지 처리 (PIL.Image 대체)

*tensorflow : 딥러닝 프레임워크

*tensorflow-datasets : 공개 데이터셋 로드

*scikit-learn : 데이터 전처리, 평가 지표 (confusion_matrix, class_weight)

*matplotlib : 데이터 및 결과 시각화
