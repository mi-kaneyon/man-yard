# single cam stereo/distance cam
- It is required pretrained model
- trained model generate by yourself
- depends on distance from cam, target color is changed.

# directory structure (example)
script
dataset--- saving distance image by correct.py

```
python  correct.py
```
in the specified directory create distance images.

## train

```
python train.py

```
generate pth file for distance model.


## execute


```
python sin_stereo.py

```


