# Optimize computer vision models

Практика по курсу "Оптимизация моделей компьютерного зрения"  
### Задание 1. 
окружение установлено

### Задание 2. 


### Задание 3. 
### Оптимизация гиперпараметров модели CNN для классификации изображений CIFAR-10 с использованием Optuna

В рамках задания мы оптимизировали гиперпараметры свёрточной нейронной сети (CNN) для классификации изображений из набора данных CIFAR-10 с использованием библиотеки Optuna. Оптимизация гиперпараметров позволяет повысить точность модели, улучшив её способность к обобщению на новых данных.

## Результаты

После выполнения 100 испытаний с ограничением по времени (1000 секунд) были получены следующие результат:
``` Статистика исследования:
  Количество завершенных испытаний:  100
  Количество обрезанных испытаний:  83
  Количество завершенных испытаний:  17
Лучшее испытание:
  Значение (accuracy):  0.4375
  Параметры: 
    n_layers: 2
    n_units_l0: 56
    kernel_size: 7
    dropout_l0: 0.2257143741369136
    n_units_l1: 31
    dropout_l1: 0.32903423959777
    optimizer: RMSprop
    lr: 2.9396995333195026e-05

Завершенные испытания:
  Trial #0: Value: 0.03125, Params: {'n_layers': 2, 'n_units_l0': 102, 'kernel_size': 7, 'dropout_l0': 0.36465190558580696, 'n_units_l1': 58, 'dropout_l1': 0.38424451852414593, 'optimizer': 'SGD', 'lr': 2.8820345950633017e-05}
  Trial #1: Value: 0.0625, Params: {'n_layers': 1, 'n_units_l0': 28, 'kernel_size': 6, 'dropout_l0': 0.36875203175110155, 'optimizer': 'SGD', 'lr': 5.6294906598617406e-05}
  Trial #2: Value: 0.0625, Params: {'n_layers': 3, 'n_units_l0': 85, 'kernel_size': 7, 'dropout_l0': 0.4451893009249124, 'n_units_l1': 102, 'dropout_l1': 0.4775300309059757, 'n_units_l2': 122, 'dropout_l2': 0.24557201564453837, 'optimizer': 'SGD', 'lr': 0.07117840080284701}
  Trial #3: Value: 0.09375, Params: {'n_layers': 1, 'n_units_l0': 81, 'kernel_size': 3, 'dropout_l0': 0.4989671146716079, 'optimizer': 'Adam', 'lr': 1.459164712550809e-05}
  Trial #4: Value: 0.09375, Params: {'n_layers': 1, 'n_units_l0': 16, 'kernel_size': 5, 'dropout_l0': 0.4640492056492301, 'optimizer': 'Adam', 'lr': 0.01910123395633198}
  Trial #5: Value: 0.0625, Params: {'n_layers': 2, 'n_units_l0': 102, 'kernel_size': 7, 'dropout_l0': 0.27563073133496124, 'n_units_l1': 80, 'dropout_l1': 0.3666214221723616, 'optimizer': 'RMSprop', 'lr': 0.0003131274444662333}
  Trial #13: Value: 0.0, Params: {'n_layers': 1, 'n_units_l0': 37, 'kernel_size': 4, 'dropout_l0': 0.4625637979968895, 'optimizer': 'Adam', 'lr': 0.011523558647862859}
  Trial #16: Value: 0.1875, Params: {'n_layers': 2, 'n_units_l0': 86, 'kernel_size': 6, 'dropout_l0': 0.3953349255193238, 'n_units_l1': 16, 'dropout_l1': 0.21338998432075754, 'optimizer': 'Adam', 'lr': 0.00013812112494896188}
  Trial #17: Value: 0.0, Params: {'n_layers': 2, 'n_units_l0': 89, 'kernel_size': 6, 'dropout_l0': 0.39920678357351524, 'n_units_l1': 18, 'dropout_l1': 0.21971471835366782, 'optimizer': 'RMSprop', 'lr': 1.0491165749893542e-05}
  Trial #20: Value: 0.1875, Params: {'n_layers': 2, 'n_units_l0': 62, 'kernel_size': 5, 'dropout_l0': 0.2013270225618842, 'n_units_l1': 77, 'dropout_l1': 0.4149403871268424, 'optimizer': 'RMSprop', 'lr': 0.00017278820056393826}
  Trial #24: Value: 0.1875, Params: {'n_layers': 2, 'n_units_l0': 53, 'kernel_size': 3, 'dropout_l0': 0.23225578360398974, 'n_units_l1': 35, 'dropout_l1': 0.3255874359842742, 'optimizer': 'RMSprop', 'lr': 2.7348731268045402e-05}
  Trial #26: Value: 0.25, Params: {'n_layers': 2, 'n_units_l0': 39, 'kernel_size': 6, 'dropout_l0': 0.2415724231060232, 'n_units_l1': 28, 'dropout_l1': 0.23233113831480517, 'optimizer': 'RMSprop', 'lr': 2.9601819666820945e-05}
  Trial #31: Value: 0.4375, Params: {'n_layers': 2, 'n_units_l0': 56, 'kernel_size': 7, 'dropout_l0': 0.2257143741369136, 'n_units_l1': 31, 'dropout_l1': 0.32903423959777, 'optimizer': 'RMSprop', 'lr': 2.9396995333195026e-05}
  Trial #52: Value: 0.15625, Params: {'n_layers': 1, 'n_units_l0': 77, 'kernel_size': 3, 'dropout_l0': 0.4853295478361092, 'optimizer': 'Adam', 'lr': 1.6060049463800784e-05}
  Trial #63: Value: 0.25, Params: {'n_layers': 1, 'n_units_l0': 64, 'kernel_size': 4, 'dropout_l0': 0.48143469267138267, 'optimizer': 'Adam', 'lr': 7.331042972379561e-05}
  Trial #72: Value: 0.1875, Params: {'n_layers': 1, 'n_units_l0': 84, 'kernel_size': 3, 'dropout_l0': 0.4813876776594327, 'optimizer': 'Adam', 'lr': 2.7486207708826278e-05}
  Trial #74: Value: 0.1875, Params: {'n_layers': 1, 'n_units_l0': 78, 'kernel_size': 3, 'dropout_l0': 0.23357446287287528, 'optimizer': 'Adam', 'lr': 4.694138659994984e-05}

Обрезанные испытания:
  Trial #6: Value: 0.09375
  Trial #7: Value: 0.09375
  Trial #8: Value: 0.09375
  Trial #9: Value: 0.03125
  Trial #10: Value: 0.0625
  Trial #11: Value: 0.09375
  Trial #12: Value: 0.09375
  Trial #14: Value: 0.09375
  Trial #15: Value: 0.125
  Trial #18: Value: 0.09375
  Trial #19: Value: 0.125
  Trial #21: Value: 0.09375
  Trial #22: Value: 0.125
  Trial #23: Value: 0.0625
  Trial #25: Value: 0.0625
  Trial #27: Value: 0.0625
  Trial #28: Value: 0.125
  Trial #29: Value: 0.09375
  Trial #30: Value: 0.09375
  Trial #32: Value: 0.09375
  Trial #33: Value: 0.0
  Trial #34: Value: 0.09375
  Trial #35: Value: 0.125
  Trial #36: Value: 0.03125
  Trial #37: Value: 0.125
  Trial #38: Value: 0.09375
  Trial #39: Value: 0.125
  Trial #40: Value: 0.0625
  Trial #41: Value: 0.0625
  Trial #42: Value: 0.125
  Trial #43: Value: 0.125
  Trial #44: Value: 0.0625
  Trial #45: Value: 0.0625
  Trial #46: Value: 0.09375
  Trial #47: Value: 0.125
  Trial #48: Value: 0.09375
  Trial #49: Value: 0.03125
  Trial #50: Value: 0.03125
  Trial #51: Value: 0.09375
  Trial #53: Value: 0.09375
  Trial #54: Value: 0.0625
  Trial #55: Value: 0.09375
  Trial #56: Value: 0.0625
  Trial #57: Value: 0.09375
  Trial #58: Value: 0.0
  Trial #59: Value: 0.15625
  Trial #60: Value: 0.125
  Trial #61: Value: 0.09375
  Trial #62: Value: 0.125
  Trial #64: Value: 0.15625
  Trial #65: Value: 0.125
  Trial #66: Value: 0.09375
  Trial #67: Value: 0.0625
  Trial #68: Value: 0.09375
  Trial #69: Value: 0.0625
  Trial #70: Value: 0.15625
  Trial #71: Value: 0.09375
  Trial #73: Value: 0.15625
  Trial #75: Value: 0.03125
  Trial #76: Value: 0.09375
  Trial #77: Value: 0.09375
  Trial #78: Value: 0.15625
  Trial #79: Value: 0.0625
  Trial #80: Value: 0.125
  Trial #81: Value: 0.09375
  Trial #82: Value: 0.0625
  Trial #83: Value: 0.0625
  Trial #84: Value: 0.0625
  Trial #85: Value: 0.15625
  Trial #86: Value: 0.125
  Trial #87: Value: 0.03125
  Trial #88: Value: 0.03125
  Trial #89: Value: 0.15625
  Trial #90: Value: 0.125
  Trial #91: Value: 0.15625
  Trial #92: Value: 0.09375
  Trial #93: Value: 0.09375
  Trial #94: Value: 0.09375
  Trial #95: Value: 0.15625
  Trial #96: Value: 0.0625
  Trial #97: Value: 0.15625
  Trial #98: Value: 0.09375
  Trial #99: Value: 0.09375
```
### Задание 4

В  работе была проведена оптимизация модели с использованием формата ONNX. Основной целью было сравнение производительности и точности модели, реализованной в PyTorch, с ее ONNX-версией. 
#### Измерение скорости вывода:
##### Среднее время вывода для ONNX: 
onnx_speed: 0.000741 seconds per batch
##### Среднее время вывода для PyTorch: 
pytorch_speed: 0.000048 seconds per batch
Результаты показали, что PyTorch  обеспечивает более быстрое время вывода по сравнению с  моделью в ONNX. 

#### Измерение точности:

Точность для ONNX: 
onnx_accuracy:.0.9725
Точность для PyTorch: 
pytorch_accuracy:0.9725
Точность обеих моделей была сопоставима, преобразование в ONNX не повлияло на качество предсказаний. 

