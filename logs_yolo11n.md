## Графики и метрики обучения yolo11n-pose в задаче определения ключевых точек руки

#### Графики обучения
<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/4050d4d2-9f26-4b9a-a340-e300e07850d5" alt="Image 1" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/5192cd0e-c290-4ec6-98f4-8ea9456f2bff" alt="Image 2" style="max-width: 100%; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/c3bcfe61-3f8d-4364-9044-d06108f6759a" alt="Image 3" style="max-width: 100%; height: auto;">
    </td>
  </tr>
  <tr>
    <td>metrics</td>
    <td>train</td>
    <td>val</td>
  </tr>
</table>

#### Confusion matrix
<img width="511" alt="image" src="https://github.com/user-attachments/assets/0b5202c5-adea-4097-9355-236bb0b588df" />

#### Итоговые метрики на валидации
<img width="843" alt="image" src="https://github.com/user-attachments/assets/f86a075a-f686-4b15-9cf9-a285c52b8cfb" />

### Примеры предсказаний на датасете для валидации

![image](https://github.com/user-attachments/assets/1fd63dbd-0bd4-468a-a2e9-c08312ad8ad4)
![image](https://github.com/user-attachments/assets/07d5b997-ba3c-4bd6-98b3-693bd438b40d)
![image](https://github.com/user-attachments/assets/38793536-b6bf-493a-afcc-e4768deb38a4)


#### Гиперпараметры:
| Параметр               | Значение                                                                              |
|------------------------|---------------------------------------------------------------------------------------|
| agnostic_nms           | False                                                                                 |
| amp                    | True                                                                                  |
| augment                | False                                                                                 |
| auto_augment           | randaugment                                                                           |
| batch                  | 16                                                                                    |
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
| data                   | data.yaml                                                                             |
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
| model                  | yolo11n-pose.pt                                                                       |
| momentum               | 0.937                                                                                 |
| mosaic                 | 1.0                                                                                   |
| multi_scale            | False                                                                                 |
| name                   | train                                                                                 |
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
| project                | hand-keypoints-detection/yolo11n                                                      |
| rect                   | False                                                                                 |
| resume                 | False                                                                                 |
| retina_masks           | False                                                                                 |
| save                   | True                                                                                  |
| save_conf              | False                                                                                 |
| save_crop              | False                                                                                 |
| save_dir               | hand-keypoints-detection/yolo11n/train                                                |
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
| workers                | 8                                                                                     |
| workspace              |                                                                                       |
