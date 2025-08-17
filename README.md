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
