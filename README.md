This is a superpixel segmentation method based on DBSCAN, this project is realized according to work "Real-Time Superpixel Segmentation by DBSCAN Clustering Algorithm".

# Usage
## Compile
mkdir build && cd build
cmake ..
make
## Example 
./bin/testDBscan ./data/dog


# Note
There are some isolated pixels in image, so there is some bug to deal with to ensure coonection.
