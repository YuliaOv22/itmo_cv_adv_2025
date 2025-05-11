## Графики и метрики обучения yolo11x-pose в задаче определения ключевых точек руки

#### Итоговые метрики на валидации
<img width="838" alt="image" src="https://github.com/user-attachments/assets/d3cdf330-2c15-48b4-ac0f-61a52a95d834" />

#### Графики обучения
<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/17365ede-0bbe-4593-8fa8-69dd211bc780" alt="Image 1" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/6ff7766a-c7a5-4cd1-86a1-01ff55564ef4" alt="Image 2" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/9fc458c2-270b-4980-ac09-2951ff257e9b" alt="Image 3" style="max-width: 100%; height: auto;">
    </td>
  </tr>
  <tr>
    <td>metrics</td>
    <td>train</td>
    <td>val</td>
  </tr>
</table>

![image](https://github.com/user-attachments/assets/3b2432b0-1b05-4226-be89-2268f71b8ba6)

#### Pose метрики
<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f91d0cca-6eab-4cf1-a33d-d87f2e91e80f" alt="Image 1" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/9cbb983c-b832-4ccf-a666-80a264f32dfa" alt="Image 2" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/477eb732-9bf0-4025-8d84-ab41657a08be" alt="Image 3" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/6771faca-03a6-4683-827d-d5ca65d7f4cf" alt="Image 4" style="max-width: 100%; height: auto;">
    </td>
  </tr>
</table>

#### Box метрики
<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1a8b90e9-671a-48df-9da3-8ad7c71b8a90" alt="Image 1" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/f0bc7264-e9fc-42cc-87be-04ee476b31bf" alt="Image 2" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/57720ebc-4829-4b6f-b757-32fda5cf3dad" alt="Image 3" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/bbf42965-60c3-41c4-8e15-d1ed45b53fd6" alt="Image 4" style="max-width: 100%; height: auto;">
    </td>
  </tr>
</table>

#### Confusion matrix
<img width="511" alt="image" src="https://github.com/user-attachments/assets/e4b8ef9a-a23f-4f57-9254-3cad4964ae1f" />

#### Примеры предсказаний на датасете для валидации

![image](https://github.com/user-attachments/assets/7faf1448-1619-4ec0-bfd5-7bb9d7d6cacb)
![image](https://github.com/user-attachments/assets/132668b4-d86f-4572-aab6-a38ac04b3198)
![image](https://github.com/user-attachments/assets/2506c4e7-9d57-4a60-a5d8-cee679e0b95e)


#### Гиперпараметры:

| Параметр               | Значение                                                                              |
|------------------------|---------------------------------------------------------------------------------------|
| agnostic_nms           | False                                                                                 |
| amp                    | True                                                                                  |
| augment                | False                                                                                 |
| auto_augment           | randaugment                                                                           |
| batch                  | 64                                                                                    |
| bgr                    | 0.0                                                                                   |
| box                    | 7.5                                                                                   |
| cache                  | False                                                                                 |
| cfg                    |                                                                                       |
| classes                |                                                                                       |
| close_mosaic           | 10                                                                                    |
| cls                    | 0.5                                                                                   |
| conf                   |                                                                                       |
| copy_paste             | 0.0                                                                                   |
| copy_paste_mode        | flip                                                                                  |
| cos_lr                 | False                                                                                 |
| crop_fraction          | 1.0                                                                                   |
| data                   | data.yaml                                                                   |
| degrees                | 0.0                                                                                   |
| deterministic          | True                                                                                  |
| device                 |                                                                                       |
| dfl                    | 1.5                                                                                   |
| dnn                    | False                                                                                 |
| dropout                | 0.0                                                                                   |
| dynamic                | False                                                                                 |
| embed                  |                                                                                       |
| epochs                 | 100                                                                                   |
| erasing                | 0.4                                                                                   |
| exist_ok               | False                                                                                 |
| fliplr                 | 0.5                                                                                   |
| flipud                 | 0.0                                                                                   |
| format                 | torchscript                                                                           |
| fraction               | 1.0                                                                                   |
| freeze                 |                                                                                       |
| half                   | False                                                                                 |
| hsv_h                  | 0.015                                                                                 |
| hsv_s                  | 0.7                                                                                   |
| hsv_v                  | 0.4                                                                                   |
| imgsz                  | 640                                                                                   |
| int8                   | False                                                                                 |
| iou                    | 0.7                                                                                   |
| keras                  | False                                                                                 |
| kobj                   | 1.0                                                                                   |
| line_width             |                                                                                       |
| lr0                    | 0.01                                                                                  |
| lrf                    | 0.01                                                                                  |
| mask_ratio             | 4                                                                                     |
| max_det                | 300                                                                                   |
| mixup                  | 0.0                                                                                   |
| mode                   | train                                                                                 |
| model                  | yolo11x-pose.pt                                                                       |
| momentum               | 0.937                                                                                 |
| mosaic                 | 1.0                                                                                   |
| multi_scale            | False                                                                                 |
| name                   | train4                                                                                |
| nbs                    | 64                                                                                    |
| nms                    | False                                                                                 |
| opset                  |                                                                                       |
| optimize               | False                                                                                 |
| optimizer              | auto                                                                                  |
| overlap_mask           | True                                                                                  |
| patience               | 100                                                                                   |
| perspective            | 0.0                                                                                   |
| plots                  | True                                                                                  |
| pose                   | 12.0                                                                                  |
| pretrained             | True                                                                                  |
| profile                | False                                                                                 |
| project                | hand-keypoints-detection/yolo11x                                       |
| rect                   | False                                                                                 |
| resume                 | False                                                                                 |
| retina_masks           | False                                                                                 |
| save                   | True                                                                                  |
| save_conf              | False                                                                                 |
| save_crop              | False                                                                                 |
| save_dir               | hand-keypoints-detection/yolo11x/train                                 |
| save_frames            | False                                                                                 |
| save_hybrid            | False                                                                                 |
| save_json              | False                                                                                 |
| save_period            | -1                                                                                    |
| save_txt               | False                                                                                 |
| scale                  | 0.5                                                                                   |
| seed                   | 0                                                                                     |
| shear                  | 0.0                                                                                   |
| show                   | False                                                                                 |
| show_boxes             | True                                                                                  |
| show_conf              | True                                                                                  |
| show_labels            | True                                                                                  |
| simplify               | True                                                                                  |
| single_cls             | False                                                                                 |
| source                 |                                                                                       |
| split                  | val                                                                                   |
| stream_buffer          | False                                                                                 |
| task                   | pose                                                                                  |
| time                   |                                                                                       |
| tracker                | botsort.yaml                                                                          |
| translate              | 0.1                                                                                   |
| val                    | True                                                                                  |
| verbose                | True                                                                                  |
| vid_stride             | 1                                                                                     |
| visualize              | False                                                                                 |
| warmup_bias_lr         | 0.1                                                                                   |
| warmup_epochs          | 3.0                                                                                   |
| warmup_momentum        | 0.8                                                                                   |
| weight_decay           | 0.0005                                                                                |
| workers                | 16                                                                                    |
| workspace              |                                                                                       |
