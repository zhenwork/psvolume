
## MERGE
PV = psvolume(label="G150A")
for idx in range(1000):
    data = Data(fname=filenames[idx], load=function)
    back = Data(fname=fbackground[idx], load=function)
    PV.data.addpoint(data=data, background=back)

PV.preprocess(background_subtract=True, scale_alg="profile", submit=True, num_cpu=20, num_node=3)
PV.merge(target="preprocessed")
PV.volume.symmetrize(space_group="P1211")
PV.volume.background()
PV.volume.deploy(fname="OUT-G150A.volume")


## ANALYSIS
VM = DSVolume()
print VM.correlation(volume1={"fname":"OUT-G150A.volume"}, volume2={"fname":"OUT-G150A.volume"}, num_resolution_shell=20, high_res=1.7)
