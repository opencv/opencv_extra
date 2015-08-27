How to convert matlab model
===========================
 
1. Download or train the model using the [DPM code](htts://github.com/rbgirshick/voc-dpm)
 
2. Convert the model. For example, in matlab:
 
```
load('VOC2007/person_final.mat')
model = cascade_model(model, '2007', 5, -0.5);
save('person_final_cascade.mat', 'model');
mat2xml('person_final_cascade.mat', 'person_final_cascade.xml')
```

Note: copy `pcacoeff.bin` to the folder where you call `mat2xml.m`

