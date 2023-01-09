# open_cv_lane_tracking
OpenCV + Gazebo/rosbag lane tracking algorithm. 

In the perfect environment of Gazebo, image edge detection using gradients work great as seen below. The raw image is a camera mounted on a vehicle. This raw feed is transformed into a bird's eye view using coordinate transformations with known distances (presumably the equivalent in real-life would be using an RGB-Depth camera). After we have the bird's eye view, a simple edge detection using gradients is used to find the lanes which are highlighted live in green. 

https://user-images.githubusercontent.com/57650580/211231538-9680f056-cc37-4a95-82d7-3ce2ccf3e834.mp4

However, in real life, there's a lot of edges and the algorithm has a tough time finding the lanes as seen in the second video. 

https://user-images.githubusercontent.com/57650580/211231540-9ee4e11f-91c8-42de-b02e-be3208f67fa6.mp4
