# Pytorch_facial_expression_recognition
Train ResNet18 on ExpW dataset for facial expression recognition with Pytorch

### Environment

- python==3.6.9
- Pytorch==1.7.0
- onnx==1.7.0
- onnxruntime==1.5.2
- opencv-python

### Getting Started

3. Configure parameters and dataset dictionary in "config.yaml". 
2. Run "main.py" to train the network.
3. Run "export.py" to convert the .pt model into onnx format. (a trained model with 72.65% validation accuracy has been placed and converted already).
4. Run "image_demo.py" and "video_demo.py" to test the onnx model with a single image or USB camera.

