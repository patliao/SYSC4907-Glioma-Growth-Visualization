import numpy as np

from equation_constant import EquationConstant
import ants, sys

threshold = 0.5

# ========================== Getting Input ===================================
if len(sys.argv) != 5:
    print("Incorrect path")
    sys.exit("exit diffusion map")


path = sys.argv[1]

try:
    grey_diffusion_rate = float(sys.argv[2])
except:
    grey_diffusion_rate = EquationConstant.GREY_DIFFUSION_RATE

try:
    white_diffusion_rate = float(sys.argv[3])
except:
    white_diffusion_rate = EquationConstant.WHITE_DIFFUSION_RATE

try:
    csf_diffusion_rate = float(sys.argv[4])
except:
    csf_diffusion_rate = EquationConstant.CSF_DIFFUSION_RATE

print(f"t1 path: {path}")
print(f"grey {grey_diffusion_rate}, white {white_diffusion_rate}, diffusion rate: {csf_diffusion_rate}")
# ========================= Finish getting Input ====================

print("Segmenting MRI data (this will take several moments)...")

t1_image_path = r"{}".format(path)

t1_image = ants.image_read(t1_image_path)

print("ants.image_read")

t1_corrected = ants.n4_bias_field_correction(t1_image)

print("ants.n4_bias_field_correction")

t1_normalized = ants.iMath(t1_corrected, "Normalize")

print("ants.iMath")

brain_mask = ants.get_mask(t1_normalized)

print("ants.get_mask")

refined_mask = ants.iMath(brain_mask, "MD", 2)

print("another ants.iMath")

segmentation = ants.atropos(
                a=t1_normalized,
                x=refined_mask,
                i=f'kmeans[5]',
                m='[0.6,1x1x1]',
                c='[10,0.01]'
        )

print("ants.atropos")

# Combine clusters for CSF, GM, and WM
csf_map = segmentation['probabilityimages'][0] + segmentation['probabilityimages'][1]  # CSF
gm_map = segmentation['probabilityimages'][2] + segmentation['probabilityimages'][3]  # GM
wm_map = segmentation['probabilityimages'][4]  # WM

print("after segmentation")

csf_map = ants.threshold_image(csf_map, threshold, 1)
gm_map = ants.threshold_image(gm_map, threshold, 1)
wm_map = ants.threshold_image(wm_map, threshold, 1)

print("after threshold_image")

csf_data = csf_map.numpy()
gm_data = gm_map.numpy()
wm_data = wm_map.numpy()

print("after map data")

# Generate the final diffusion map as a weighted sum
diffusion_map = np.zeros_like(gm_data)

# diffusion_map[csf_data > 0] = EquationConstant.CSF_DIFFUSION_RATE
# diffusion_map[gm_data > 0] = EquationConstant.GREY_DIFFUSION_RATE
# diffusion_map[wm_data > 0] = EquationConstant.WHITE_DIFFUSION_RATE

diffusion_map[csf_data > 0] = csf_diffusion_rate
diffusion_map[gm_data > 0] = grey_diffusion_rate
diffusion_map[wm_data > 0] = white_diffusion_rate

np.save('diffusion_map.npy', diffusion_map)  # Save the diffusion map to local then other process can retrieve
print("ants finish")