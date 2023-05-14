
# DocIIW
Repository for the paper "Intrinsic Decomposition of Document Images In-the-Wild" (BMVC '20)

Quick Links: [PDF](https://www.bmvc2020-conference.com/assets/papers/0906.pdf) | [arXiv](https://arxiv.org/pdf/2011.14447.pdf) |  [Talk](https://www.bmvc2020-conference.com/conference/papers/paper_0906.html) | [Supplementary](https://drive.google.com/file/d/1HTL0L9bTeMJF3DwEBWZdlPXl-tGhRK4S/view?usp=share_link) 
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
Doc3Dshade contains 90K images, 80K used for training and 10K for validation. Split used in the paper: [train](https://drive.google.com/file/d/15ZPmNU6XeLd4pg4Mz8Qm6oXLHn8CzCMx/view?usp=share_link), [val](https://drive.google.com/file/d/12t6-XRLHgzDjs9PdpZiTXzIfhBn97lCu/view?usp=share_link)
* Download the input images from [img.zip](https://drive.google.com/file/d/1ixqktu8dC3pSE4EskOadqLKTJwDW9-J-/view?usp=sharing) .
* Download the white-balanced images from [wbl.zip](https://drive.google.com/file/d/1o-8jtnYysXqmFbV-xB_N4Wwcpxg38t67/view?usp=sharing) .
* Download synthetic textures from [alb.zip](https://drive.google.com/file/d/1CwLSwO7-ePN6tJayeUrxnSL8d9PvIfcJ/view?usp=sharing) .

### Training Instructions
* Upcoming

### Pre-trained Models
* All models: [GDrive Link](https://drive.google.com/drive/folders/1sTw9Qm1naNvtJ5KrvDZOqq74mBctVNod?usp=share_link)
* WBNet: [GDrive Link](https://drive.google.com/file/d/1pLgGgbaYV4xMmrW7568bFEK4lbt7YD2v/view?usp=share_link)
* SMTNet: [GDrive Link](https://drive.google.com/file/d/1rdU27SFwQMx-glcYg1558fysaKy5NJxB/view?usp=share_link)
* SMTNet(w/ adversarial loss): [GDrive Link](https://drive.google.com/file/d/1S7OKsP8ka1RruTAhTKBcz35y3ZXuQXmm/view?usp=share_link)

### Evaluation Images and Results
* Real test images are given in: ```/testimgs/real```
* Shading removed real test images:
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/1ag-ZSZZMKjiq42vaeED0Bcf5RihI7TMC?usp=share_link)
* Shading removed DocUNet [1] images are available at: 
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/10OcNxOQo_tugYZs3px1m9Uhfgipz-eJN?usp=share_link)
	* With adversarial loss: [GDrive Link](https://drive.google.com/drive/folders/1smG7mWw1T9qwG896odWuyyYgGZuy61Y-?usp=share_link)
* Shading removed and unwarped [2] DocUNet [1] images are available at:
	* Basic: [GDrive Link](https://drive.google.com/drive/folders/17oZvT57hXtwaugZcY4aLepEqZ1ocirgt?usp=share_link)
	*  With adversarial loss: [GDrive Link](https://drive.google.com/drive/folders/1v8Dx6pJGmd5LWdg1hIdru_Ffkqy23SOz?usp=share_link)

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
