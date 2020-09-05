# DocIIW
Repository for the paper Intrinsic Decomposition of Document Images In-the-Wild (BMVC '20)

Quick Links: [PDF](https://www.bmvc2020-conference.com/assets/papers/0906.pdf) | [arXiv (Coming Soon)]() |  [Talk](https://www.bmvc2020-conference.com/conference/papers/paper_0906.html) | [Supplementary]() 
## Updates
* **Sep 5th, 2020:**  Initial data is released (90K images)
* **Coming Soon:** Training Code 
## Doc3DShade
Doc3DShade extends [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset) with realistic lighting and shading. Follows a similar synthetic rendering procedure using captured document 3D shapes but final image generation step combines real shading of different types of paper materials under numerous illumination conditions. Following figure illustrates the image generation pipeline:
![Dataset Capture Pipeline](/assets/pipeline.png)

Following figure shows a side-by-side comparison of images in Doc3DShade and Doc3D:
![Comparison with Doc3D](/assets/comp.png)
### Download Instructions
Doc3Dshade contains 90K images, 80K used for training and 10K for validation. Split used in the paper: [train](https://drive.google.com/file/d/1kRrmheEr2uNpYW6839rD1jCPa57YcxAb/view?usp=sharing), [val](https://drive.google.com/file/d/14siJyQOtxq4HNbfX8R969VhhR7wv-t8_/view?usp=sharing)
* Download the input images from [img.zip](https://drive.google.com/file/d/1ixxgbcGoNIdYudoHUvGaQlXHiH1Vqv_I/view?usp=sharing) .
* Download the white-balanced images from [wbl.zip](https://drive.google.com/file/d/1bhWqCezS1FTCUtSjQk6jgmIBD4IWrbjp/view?usp=sharing) .
* Download synthetic textures from [alb.zip](https://drive.google.com/file/d/1iFr9xfTPJBuBH2rThPXLgpWiSlhNtg1e/view?usp=sharing) .


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


