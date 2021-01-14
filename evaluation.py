import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

location = "./Evaluation/"
df_rmse=pd.DataFrame(columns=["Image","average_rmse","singan_rmse","telea30_rmse", "scratch_rmse"])

for i in [1,2,3,4,5,7]:

    ground_truth = cv2.imread("%simage_%s.jpeg" % (location, i))
    mask = cv2.imread("%simage_%sM_mask.jpeg" % (location, i))
    singan = cv2.imread("%simage_%sM_inp.jpeg" % (location, i))
    average = cv2.imread("%simage_%sM_average.jpeg" % (location, i))
    telea30 = cv2.imread("%simage_%sM_telea30.jpeg" % (location, i))
    scratch = cv2.imread("%simage_%sM.jpeg" % (location, i))
    bool = mask<=[5,5,5]

    ground_truth[bool]=0
    singan[bool]=0
    average[bool]=0
    telea30[bool]=0
    scratch[bool]=0

    singan_vect = singan.reshape(singan.size)
    ground_truth_vect = ground_truth.reshape(ground_truth.size)
    average_vect = average.reshape(average.size)
    telea30_vect = telea30.reshape(average.size)
    scratch_vect = scratch.reshape(scratch.size)

    singan_rmse=np.sum((singan_vect-ground_truth_vect)**2)**0.5
    average_rmse=np.sum((average_vect-ground_truth_vect)**2)**0.5
    telea30_rmse=np.sum((telea30_vect-ground_truth_vect)**2)**0.5
    scratch_rmse=np.sum((scratch_vect-ground_truth_vect)**2)**0.5

    df_rmse=df_rmse.append({"Image":i, "average_rmse":average_rmse,"singan_rmse":singan_rmse,"telea30_rmse":telea30_rmse, "scratch_rmse":scratch_rmse},ignore_index=True)
df_rmse
df_rmse.to_csv("rmse_1_5.csv")
df_rmse.mean()
df_rmse.std()
df_sifid=pd.DataFrame({"Image":[1,2,3,4,5], "average_sifid":[0.08940998, 0.015808662,0.0057500675, 0.056514412,0.038952965 ],"singan_sifid":[0.0925972, 0.017800523,0.0071632843,0.05916261, 0.04195784 ],"navier_stokes_sifid":[0.09149952, 0.015783325,0.005805154,0.05904092,0.039077923], "scratch_sifid":[0.045982145,0.015224861,0.005115224,0.031570796, 0.013509947]})
#df_rmse.mean()
df_sifid.mean()
df_sifid.to_csv("sifid_1_5.csv")

#%%
i=4
ground_truth = cv2.imread("%simage_%s.jpeg" % (location, i))
singan = cv2.imread("%simage_%sM_inp.jpeg" % (location, i))
telea30 = cv2.imread("%simage_%sM_telea30.jpeg" % (location, i))
scratch = cv2.imread("%simage_%sM.jpeg" % (location, i))



fig, ax  = plt.subplots(2, 4,figsize=(15,15))
fig.tight_layout(h_pad=-70)
# equivalent but more general

ax[0,0].set_title("Original image")
ax[0,0].set_axis_off()
ax[0,0].imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))

ax[0,1].set_title("Masked image")
ax[0,1].set_axis_off()
ax[0,1].imshow(cv2.cvtColor(scratch, cv2.COLOR_BGR2RGB))

ax[0,2].set_title("Telea Method")
ax[0,2].set_axis_off()
ax[0,2].imshow(cv2.cvtColor(telea30, cv2.COLOR_BGR2RGB))

ax[0,3].set_title("Our Method")
ax[0,3].set_axis_off()
ax[0,3].imshow(cv2.cvtColor(singan, cv2.COLOR_BGR2RGB))

i=3
ground_truth = cv2.imread("%simage_%s.jpeg" % (location, i))
singan = cv2.imread("%simage_%sM_inp.jpeg" % (location, i))
telea30 = cv2.imread("%simage_%sM_telea30.jpeg" % (location, i))
scratch = cv2.imread("%simage_%sM.jpeg" % (location, i))

ax[1,0].imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
ax[1,0].set_axis_off()

ax[1,1].set_axis_off()
ax[1,1].imshow(cv2.cvtColor(scratch, cv2.COLOR_BGR2RGB))

ax[1,2].set_axis_off()
ax[1,2].imshow(cv2.cvtColor(telea30, cv2.COLOR_BGR2RGB))

ax[1,3].set_axis_off()
ax[1,3].imshow(cv2.cvtColor(singan, cv2.COLOR_BGR2RGB))
plt.savefig("plot.jpeg")
plt.show()


#%%
singan = cv2.imread("%s0.png"%location)
ground_truth = cv2.imread("%simage_7.jpeg"%location)
re=cv2.resize(singan, (ground_truth.shape[1],ground_truth.shape[0]), interpolation=cv2.INTER_CUBIC)
plt.imshow(re)
plt.show()
plt.imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
