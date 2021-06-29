# SinGAN
Based on SinGAN*, a single-image generative adversarial network, we implemented a model performing the task of inpainting. We applied our model to two sub-tasks of inpainting: recovering small defects and removing human silhouettes from images. Quantitatively comparing our inpainting model to a state-of-the-art inpainting tool did not yield statistically significant results, due to small sample sizes and unreliable metrics. We published a qualitative questionnaire to measure how well our model could deceive the human eye. This survey revealed that our model performed almost as well as Nvidia’s and can be very realistic, depending on the type of inpainting task faced.


To train for inpainting: run main_train.py --mode inpainting
--input_dir dir name of file where input images are stored
--input_name name of file to inpaint
--mode inpainting
--on_drive  path to folder in google drive (leave blank if on local computer)

To actually inpaint: run inpainting.py
--input_dir name of file where input images are stored
--input_name name of file to inpaint
--on_drive path to folder in google drive (leave blank if on local computer)
--mask_name name of the mask used to select the inpainting zone
--inpainting_scale_start (default=1)
--radius (default=7)

*[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf)


