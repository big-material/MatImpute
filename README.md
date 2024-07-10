# MatImpute

A imputation algorithm for Material Science dataset. 😄

## TOC :point_down:

- [Features](#Features)
- [Usage](#Usage)
- [Contributing](#Contributing)

## Features 

:hammer_and_wrench:The features of MatImpute 。

* **A Nearest-Neighbor-Based Algorithm to Impute Missing Data in Material Science**
* **Provide Scikit-learn API**

## Usage 

* :hammer_and_wrench:Install :
  ```shell
  pip install git+https://github.com/big-material/MatImpute.git
  ```

* Usage:
  ```python
  import numpy as np
  import pandas as pd
  from matimpute import MatImputer
  
  df = pd.DataFrame({'a': [1, 2, 3, 4, np.NAN], 'b': [1, 2,np.NAN , 4, 5]})
  mat_impute = MatImputer()
  df_filled = mat_impute.transform(df)
  ```

## Experiment

The experiment in paper 《**A Nearest-Neighbor-Based Algorithm to Impute Missing Data in Material Science**》 was in the directory [experiment](./experiment)

## Contributing

If you find a bug :bug:, please open a [bug report](https://github.com/big-material/MatImpute/issues/new?assignees=&labels=bug&template=bug_report.md&title=).
If you have an idea for an improvement or new feature :rocket:, please open a [feature request](https://github.com/big-material/MatImpute/issues/new?assignees=&labels=Feature+request&template=feature_request.md&title=).
