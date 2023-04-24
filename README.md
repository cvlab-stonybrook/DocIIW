
# DocIIW
Repository for the paper "Intrinsic Decomposition of Document Images In-the-Wild" (BMVC '20)

Quick Links: [PDF](https://www.bmvc2020-conference.com/assets/papers/0906.pdf) | [arXiv](https://arxiv.org/pdf/2011.14447.pdf) |  [Talk](https://www.bmvc2020-conference.com/conference/papers/paper_0906.html) | [Supplementary](https://drive.google.com/file/d/1wQs6p6mMkm-z6dn9SQyPl6gwCqV2YJhf/view?usp=sharing) 
## Updates
* **Sep 5th, 2020:**  Initial data is released (90K images).
* **Mar 20th, 2021:** Evaluation images are released.
* **Nov 8th, 2022:** Training Code and models.
* **Coming Soon:** Training details.

## Doc3DShade
Doc3DShade extends [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset) with realistic lighting and shading. Follows a similar synthetic rendering procedure using captured document 3D shapes but final image generation step combines real shading of different types of paper materials under numerous illumination conditions. 
<br>
Following figure illustrates the image generation pipeline:
![Dataset Capture Pipeline](/assets/pipeline.png)

Following figure shows a side-by-side comparison of images in Doc3DShade and Doc3D:
![Comparison with Doc3D](/assets/comp.png)

### Data Download Instructions
Doc3Dshade contains 90K images, 80K used for training and 10K for validation. Split used in the paper: [train](https://drive.google.com/file/d/1kRrmheEr2uNpYW6839rD1jCPa57YcxAb/view?usp=sharing), [val](https://drive.google.com/file/d/14siJyQOtxq4HNbfX8R969VhhR7wv-t8_/view?usp=sharing)
* Download the input images from [img.zip](https://drive.google.com/file/d/1ixqktu8dC3pSE4EskOadqLKTJwDW9-J-/view?usp=sharing) .
* Download the white-balanced images from [wbl.zip](https://drive.google.com/file/d/1o-8jtnYysXqmFbV-xB_N4Wwcpxg38t67/view?usp=sharing) .
* Download synthetic textures from [alb.zip](https://drive.google.com/file/d/1CwLSwO7-ePN6tJayeUrxnSL8d9PvIfcJ/view?usp=sharing) .

### Training Instructions
* Upcoming

### Pre-trained Models
* All models: [GDrive Link](https://drive.google.com/drive/folders/1KFA-nu1CkjSCtTo_wFZx_033hxvT1fGt?usp=sharing)
* WBNet: [GDrive Link](https://drive.google.com/file/d/1B-35CZsaiBqIP4PFtlP5hnQli1IEqxrr/view?usp=share_link)
* SMTNet: [GDrive Link](https://drive.google.com/file/d/1akaotJzsjPVSCjopJ_uLGduRB8YdwwCS/view?usp=share_link)
* SMTNet(w/ adversarial loss): [GDrive Link](https://drive.google.com/file/d/1Ta2WnqZsdpEswg3cryooqT_An_51qg4N/view?usp=share_link)

### Evaluation Images and Results
* Real test images are given in: ```/testimgs/real```
* Shading removed real test images:
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/1vs5zqdqRjIXrcGc7EYhnqNI06QRQ1Ey2?usp=sharing)
* Shading removed DocUNet [1] images are available at: 
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/1YA1tcaHKxDm-80Nbjd9ln6_IYBpMhzXH?usp=sharing)
	* With adversarial loss: [GDrive Link](https://drive.google.com/drive/folders/1wDG3PIu6sx7q8oS-1_VGs3FPhMV4iJUn?usp=sharing)
* Shading removed and unwarped [2] DocUNet [1] images are available at:
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/1H5Bv5wgBxz4jq7Dr2VfH__OmiWBqVviH?usp=sharing)
	*  With adversarial loss: [GDrive Link](https://drive.google.com/drive/folders/160SQDv4PmRmIpNnp7_5450qKhfHf21SF?usp=sharing)

### Citation:
If you use the dataset, please consider citing our work-
```
@inproceedings{DasDocIIW20,
  author    = {Sagnik Das, Hassan Ahmed Sial, Ke Ma, Ramon Baldrich, Maria Vanrell and Dimitris Samaras},
  title     = {Intrinsic Decomposition of Document Images In-the-Wild},
  booktitle = {31st British Machine Vision Conference 2020, {BMVC} 2020, Manchester, UK, September 7-10, 2020},
  publisher = {{BMVA} Press},
  year      = {2020},
}
```

### References: 
[1] DocUNet: https://www3.cs.stonybrook.edu/~cvl/docunet.html

[2] DewarpNet: https://sagniklp.github.io/dewarpnet-webpage/
