# UPDRS with MediaPipe

## Requirement
**Development Environment:** Ubuntu18.04 and python3.9.9

* mediapipe >= 0.8.9
* numpy >= 1.21.4
* scipy >= 1.7.3
* opencv-contrib-python >= 4.5.4.60
* matplotlib >= 3.5.0

## Run
### Hand Movement (手掌握合)
```bash
python3.9 hand_movement.py -i [INPUT VIDEO PATH]
```
It will output two file: `hand_movement_result.mp4` for annotated video and `hand_movement.png` for waveform

### Pronation (前臂迴旋)
```bash
python3.9 pronation.py -i [INPUT VIDEO PATH]
```
It will output two file: `pronation.mp4` for annotated video and `pronation.png` for waveform

### Leg Agility (雙腳靈敏度))
```bash
python3.9 leg_agility.py -i [INPUT VIDEO PATH] -l [right|left]
```
It will output two file base on -l argument: 
* **-l left:** `left_leg_agility.mp4` for annotated video and `left_leg_agility.png` for waveform
* **-l right:** `right_leg_agility.mp4` for annotated video and `right_leg_agility.png` for waveform

### Arise from chair (從椅子上站起來)
```bash
python3.9 arise_from_chair.py -i [INPUT VIDEO PATH]
```
It will output two file: `arise_from_chair.mp4` for annotated video and `arise_from_chair.png` for waveform