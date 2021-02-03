Clip classification:
--> notation: IF (intermediate fusion), LF (Late Fusion) , EF1(Early Fusion1), EF2(Early Fusion2)
--> IF_dataset.py (pytorch dataset dataloader for IF)
--> IF_fusionNet.py (Fusion of three different network architectures for IF)
--> IF_tactileNet.py (Neural network for tactile and joint data for IF)
--> IF_train_resnet_vtp.py (Training networks for IF)
--> resnet.py (3D resnet architecture in pytorch)
--> testv2.txt (test datapoint path for TTrandom object split)
--> testv3.txt (test datapoint path for TTobject object split)
--> trainv2.txt (train datapoint path for TTrandom object split)
--> trainv3.txt (train datapoint path for TTobject object split)
--> trainv3_30percent.txt (reducing train datapoint path from 80% to 30% in trainv3.txt)

Video classification:
--> pytorch_i3d.py (I3D architecture in pytorch)
--> test_model_VF.py (test the model with test data)
--> train_i3d_vf.py (train the I3D on VT dataset)
--> videotransforms.py (random crop and horzontal flip data augmentation)
--> visual_tactile_dataset.py (dataset dataloader for pytorch)
--> test.txt (test datapoint path)
--> train.txt (train datapoint path)
--> Train_i3d.ipynb (practice notebook)
--> visualize learning curve.ipynb (visualize learning curve)

Practice code
--> Dataset Analysis.ipynb (code to understand visual-tactile (VT) dataset)
--> evaluation_resnet.ipynb (code to plot learning curve and calculate evaluation metrics)
--> feature_Exctraction.ipynb (code to extract raw features from numpy array input)
--> pytorch_dataloaderv3.ipynb (code to write dataloader for pytorch for video classification)
--> tactile_and_video_analysis.ipynb (code to analyse video and tactile/joint data in VT dataset)
--> visual_tactile_dataset.ipynb (code to write dataloader for pytorch for clip classification)

utils
--> gen_image_v2.py (convert video into the frames)
--> gen_optical_flow.py (generate x and y axis optical flow from video and save images)


