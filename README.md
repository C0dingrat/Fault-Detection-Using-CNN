# Fault-Detection-Using-CNN
Why Fault Detection Matters in the Oil & Gas Industry?
Fault detection is a core task in subsurface interpretation. Faults control:
Hydrocarbon trapping and leakage.
Reservoir compartmentalization.
Fluid flow paths and pressure behavior.
Well placement risk.

Deep learning models can:
Learn fault-related features directly from raw seismic

Capture 3D spatial context (not just slice-wise information)

Produce dense voxel-wise fault probability volumes

Reduce interpretation time dramatically

In this project, fault detection is formulated as a 3D semantic segmentation problem, where each voxel is classified as either fault or non-fault.

Model Architecture used is 3D UNET:
I used a 3D U-Net architecture, adapted from the original U-Net designed for biomedical image segmentation.

<img width="1011" height="553" alt="image" src="https://github.com/user-attachments/assets/7b53510d-33f4-4f60-a771-88c92e2fabe3" />
The U-Net is divided into two distinct flow paths:

(i) a forward contraction path involving several downsampling steps.
This is the U-Net encoder section, which includes two 3x3 convolutions, a ReLU, a 2x2 max pooling, and a stride of 2 for downsampling. The original U-Net implementation employs 'unpadded' convolutions, which result in smaller final output size.

(ii) a path of expansion in reverse, involving several upsampling steps.
This is the decoder section, which includes an upsampling of the feature map, a 3x3 convolution, and a concatenation of a feature map from the previous contracting block, followed by three 3x3 convolutions with ReLU activation.

In the original U-Net implementation, the output shape is smaller than the input, necessitating the use of a skip connection layer size that corresponds to the current layer. The skip-connection layer should be cropped to match the size of the layer after upsampling and convolution in this case. In addition, the padding must be set to 1 to ensure that the final output shape matches the input shape. Next, we need a final code block that will output tensors with the same input size.

In total, the final U-Net block has four contraction and four expansion blocks, as well as a feature map block at the network's beginning and end.

Now I used something I call The patch idea:
Why it is necessary?
Training directly on full 3D seismic volumes is computationally impractical due to memory constraints. To address this, I adopted this patch-based training approach.
A patch is simply a smaller 3D sub cube (in my case 64*64*64) cut from the big cube.

Faults are local but continuous structures

To detect fault voxel , the model needs:

-->Nearly seismic context

-->Continuity in X,Y,Z

Now in my idea I used Bias batch selection:

Fault centered patches:

-->Patch center near voxel

-->Ensures fault is visible

Background patch:

-->No fault

-->Prevent false positives

I also used sliding window inference:

While training is patch-based, prediction must be performed on the full seismic volume.

To solve this:

A sliding-window inference strategy is used

Overlapping patches are passed through the network

Predictions are averaged in overlapping regions to reduce edge artifacts

This ensures:

Full-volume fault probability maps

Smooth and spatially consistent predictions

No loss of coverage

This patch idea also helps in augmentation as we can choose the number of patches in which our image can be divided so it basically divides the image into the given number of patches which act as new images, obviously overlap will occur between those images but because of this my model can learn new patterns in them.

Now the challenges I faced with 3D data (as in normal you always get a datset of 2D which is relatively easy to analyse):

1. Memory Constraints

   3D convolutions are orders of magnitude heavier than 2D

   Batch size often had to be reduced to 1

   So because i was only to run my model only for 15 epochs

2. Longer Training Time

   3D models converge slower

   Each forward pass is computationally expensive 

   It almost took me 3hr 30 minutes for only 15 epochs even  on T4-GPU of colab.

3. Data Handling Complexity

   Raw .dat files require careful reshaping and validation

   Mistakes in volume dimensions or normalization silently break results

   Visualization is harder compared to simple 2D images

   These challenges make 3D fault detection significantly more complex than 2D approaches — but also far more realistic and valuable for real-world applications.

Conclusion

This project demonstrates a practical end-to-end pipeline for 3D seismic fault detection using deep learning:

Patch-based 3D U-Net training

Sliding-window inference for full-volume prediction

Realistic handling of large seismic datasets

While results can always be improved with better data, longer training, and multi-attribute inputs, this work establishes a solid foundation for automated fault interpretation in subsurface workflows.
