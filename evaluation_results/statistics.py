import pandas as pd
import scipy.stats as stats

#SIFID
SIFID_nvidia = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_nvidia.csv").drop("Unnamed: 0", axis=1)
SIFID_singan = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_singan.csv").drop("Unnamed: 0",  axis=1)
SIFID_telea= pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_telea.csv").drop("Unnamed: 0", axis=1)
SIFID_nvidia.mean()
SIFID_nvidia.std()
SIFID_telea.mean()
SIFID_telea.std()
SIFID_singan.mean()
SIFID_singan.std()
SIFID_nvidia
print(stats.ttest_ind(SIFID_nvidia,SIFID_singan, equal_var = False))
print(stats.ttest_ind(SIFID_nvidia,SIFID_telea, equal_var = False))
print(stats.ttest_ind(SIFID_singan,SIFID_telea, equal_var = False))

#%%
SSIM_nvidia = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SSIM_nvidia.csv").drop(["Unnamed: 2","name"], axis=1)
SSIM_singan = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SSIM_singan.csv").drop(["Unnamed: 2","name"], axis=1)
SSIM_telea = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SSIM_telea.csv").drop(["Unnamed: 2","name"], axis=1)
SSIM_singan.shape



SSIM_nvidia.mean()
SSIM_nvidia.std()
SSIM_telea.mean()
SSIM_telea.std()
SSIM_singan.mean()
SSIM_singan.std()

print(stats.ttest_ind(SSIM_nvidia,SSIM_singan, equal_var = False))
print(stats.ttest_ind(SSIM_nvidia,SSIM_telea, equal_var = False))
print(stats.ttest_ind(SSIM_singan,SSIM_telea, equal_var = False))

#%%
PSNR_nvidia = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/PSNR_nvidia.csv").drop(["Unnamed: 2","name"], axis=1)
PSNR_singan = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/PSNR_singan.csv").drop(["Unnamed: 2","name"], axis=1)
PSNR_telea = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/PSNR_telea.csv").drop(["Unnamed: 2","name"], axis=1)
PSNR_singan
PSNR_telea.shape
PSNR_nvidia.mean()
PSNR_nvidia.std()
PSNR_telea.mean()
PSNR_telea.std()
PSNR_singan.mean()
PSNR_singan.std()

print(stats.ttest_ind(PSNR_nvidia,PSNR_singan, equal_var = False))
print(stats.ttest_ind(PSNR_nvidia,PSNR_telea, equal_var = False))
print(stats.ttest_ind(PSNR_singan,PSNR_telea, equal_var = False))
#%%
#Exception finder
SIFID_nvidia = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_nvidia.csv").drop("Unnamed: 2", axis=1)
SIFID_singan = pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_singan.csv").drop("Unnamed: 2",  axis=1)
SIFID_telea= pd.read_csv("/home/dell/Documents/Scripts/InpaintingSinGAN/Evaluation/SIFID_telea.csv").drop("Unnamed: 2", axis=1)

SIFID_singan
SIFID_telea
SIFID_telea



#%%
#Form stats
df = pd.read_csv("Evaluation/Inpainting survey.csv")

section_1 = df[['Quelle image est la plus réaliste?','Quelle image est la plus réaliste?.1', 'Quelle image est la plus réaliste?.2','Quelle image est la plus réaliste?.3','Quelle image est la plus réaliste?.4','Quelle image est la plus réaliste?.5']]

section_1["Quelle image est la plus réaliste?"][section_1["Quelle image est la plus réaliste?"] == "A:"] = 1
section_1["Quelle image est la plus réaliste?"][section_1["Quelle image est la plus réaliste?"] == "B:"] = 0

section_1["Quelle image est la plus réaliste?.1"][section_1["Quelle image est la plus réaliste?.1"] == "A:"] = 1
section_1["Quelle image est la plus réaliste?.1"][section_1["Quelle image est la plus réaliste?.1"] == "B:"] = 0

section_1["Quelle image est la plus réaliste?.2"][section_1["Quelle image est la plus réaliste?.2"] == "A:"] = 0
section_1["Quelle image est la plus réaliste?.2"][section_1["Quelle image est la plus réaliste?.2"] == "B:"] = 1

section_1["Quelle image est la plus réaliste?.3"][section_1["Quelle image est la plus réaliste?.3"] == "A:"] = 0
section_1["Quelle image est la plus réaliste?.3"][section_1["Quelle image est la plus réaliste?.3"] == "B:"] = 1

section_1["Quelle image est la plus réaliste?.4"][section_1["Quelle image est la plus réaliste?.4"] == "A:"] = 1
section_1["Quelle image est la plus réaliste?.4"][section_1["Quelle image est la plus réaliste?.4"] == "B:"] = 0

section_1["Quelle image est la plus réaliste?.5"][section_1["Quelle image est la plus réaliste?.5"] == "A:"] = 1
section_1["Quelle image est la plus réaliste?.5"][section_1["Quelle image est la plus réaliste?.5"] == "B:"] = 0

personnes = section_1[["Quelle image est la plus réaliste?","Quelle image est la plus réaliste?.1","Quelle image est la plus réaliste?.2"]]



samples = personnes.sum(axis=1)
stats.ttest_1samp(samples,1.5)

samples.mean()
samples.std()

personnes = section_1[["Quelle image est la plus réaliste?.3","Quelle image est la plus réaliste?.4","Quelle image est la plus réaliste?.5"]]

samples = personnes.sum(axis=1)
stats.ttest_1samp(samples,1.5)

samples.mean()
samples.std()
