* RGBD camrea: 2 RealSense SR300 camera. One for front and one for left. Resolution of both the camera = 640 x 480 with 18 Hz
* robot arm: UR5 robot arm with 3 fingers and 1 palm
* tactile sensors: each palm and finger have 4 tactile sensors. So 4*4 = 16 sensors
* tactile sensor mechanism: piezoresistive mechanism

* table size: 60x60 cm
* depth video for each object is given only for front camera

- tactile.txt: 16 sensors (16 columns) and 400 reading or steps (400 rows) within 24 second. which means per second 16.66 reading from tactile sensors. Measurement unit mN.
- lable.txt: total 4 rows, first 3 columns they say about time but each file have the same values. Last row indicates success or faulure in grasp. 
0 = failure, 1 = success
- pos.txt : Available only for the data which contains video. 400 readings(rows) an for each motor (8 motor- 8 columns) on the finger and palm.
- images: each side have 4 images. 2 images=front camera, 2 images = left camera . Image is captured at any 2 timesteps is video
- video: 2 videos. 1 for frontRGB and 1 for topRGB

- total data: 2550 set
- success rate: 66.27% of 2550 = 1715
- Empty data (without video): Not clearly told
- full data (with video): Not clearly told
