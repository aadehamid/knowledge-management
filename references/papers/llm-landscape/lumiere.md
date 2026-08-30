Text-to-Video Image-to-Video Stylized Generation
Video Stylization Cinemagraphs Inpainting
Google Research
LUMIERE
A Space-Time Diffusion Model for Video Generation
R E A D PA P E R
Text-to-Video
* Hover over the video to see the input prompt.
   
Image-to-Video
* Hover over the video to see the input image and prompt.
   
Stylized Generation
Using a single reference image, Lumiere can generate videos in the target style by utilizing
fine-tuned text-to-image model weights.
* Hover over the video to see the prompt.
   
Style reference
image "Sticker"
Introduction
We introduce Lumiere -- a text-to-video diffusion model
designed for synthesizing videos that portray realistic, diverse
and coherent motion -- a pivotal challenge in video synthesis.
To this end, we introduce a Space-Time U-Net architecture
that generates the entire temporal duration of the video at
once, through a single pass in the model. This is in contrast to
existing video models which synthesize distant keyframes
followed by temporal super-resolution -- an approach that
inherently makes global temporal consistency difficult to
achieve. By deploying both spatial and (importantly) temporal
down- and up-sampling and leveraging a pre-trained text-to-
image diffusion model, our model learns to directly generate
a full-frame-rate, low-resolution video by processing it in
multiple space-time scales. We demonstrate state-of-the-art
text-to-video generation results, and show that our design
easily facilitates a wide range of content creation tasks and
video editing applications, including image-to-video, video
inpainting, and stylized generation.
Video Stylization
With Lumiere, off-the-shelf text-based image editing methods can be used for consistent
video editing.
Source Video "Made of wooden
blocks"
"Origami folded
paper art"
"Made of colorful
toy bricks"
"Made of flowers"
Cinemagraphs
The Lumiere model is able to animate the content of an image within a specific user-provided
region.
Input Image + Mask Output Video Input Image + Mask Output Video
Video Inpainting
Source Masked Video Output Video Source Masked Video Output Video
Source Video "wearing a gold
strapless gown"
"wearing a striped
strapless dress"
"wearing a purple
strapless dress"
"wearing a black
strapless gown"
Source Video "wearing a crown" "wearing
sunglasses"
"wearing a red
scarf"
"wearing a purple
tie"
Source Video "wearing a bath
robe"
"wearing a party
hat"
"Standing on a
stool"
"wearing rain boots"
Authors
Omer Bar-
Hila
Roni
Shiran
Ariel
Omer
Charles
Tal
*,1,2 *,1,3 *,1 †,1 †,1
Chefer
Tov
Herrmann
Paiss
Junhwa
Guanghui
Zada
†,1 †,1 †,1 1 1
Ephrat
Hur
Liu
Raj
Oliver
Amit
Yuanzhen
Michael
Tomer
Li
1 1 1,4 1
Rubinstein
Michaeli
Wang
Tali
Inbar
Deqing
Sun
1 1,2 †,1
Dekel
Mosseri
1 2 3 4Google
Weizmann
Tel-Aviv
Technion
Research
Institute
University
(*): Equal first co-author, (†) Core technical contribution
Work was done while O. Bar-Tal, H. Chefer were interns at Google.
Acknowledgements
We would like to thank Ronny Votel, Orly Liba, Hamid Mohammadi, April Lehman, Bryan Seybold,
David Ross, Dan Goldman, Hartwig Adam, Xuhui Jia, Xiuye Gu, Mehek Sharma, Keyu Zhang, Rachel
Hornung, Oran Lang, Jess Gallegos, William T. Freeman and David Salesin for their collaboration,
helpful discussions, feedback and support.
We thank owners of images and videos used in our experiments (links for attribution) for sharing
their valuable assets.
Societal Impact
Our primary goal in this work is to enable novice users to generate visual content in an
creative and flexible way. However, there is a risk of misuse for creating fake or harmful
content with our technology, and we believe that it is crucial to develop and apply tools for
detecting biases and malicious use cases in order to ensure a safe and fair use.
