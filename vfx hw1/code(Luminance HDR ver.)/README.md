1. Put source images under "vfx/{artifact_name}/" and name them 1.JPG, 2.JPG... and so on, in the order of delta t from large to small. Ex. 1/30 to 1/1000
2. Specify IMAGE_NUM in hdr.py
3. Specify delta_t array in hdr.py
4. Run `pyhton3 hdr.py {artifact_name}`. Ex. pyhton3 hdr.py statue