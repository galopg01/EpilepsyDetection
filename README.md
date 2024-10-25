El código proporcionado se divide en distintos archivos, cuyo contenido es el siguiente:

- **dataset.py**: Contiene las transformaciones que serán aplicadas a las imágenes del conjunto. A su vez, contiene la clase `EpilepsyDataset`, que hereda de la clase `Dataset` y se utiliza para cargar y manipular el conjunto de datos.

- **patchMain.py**: Contiene principalmente todos los métodos para la división de las imágenes de resonancia magnética en subimágenes. El método `delete_images`, que elimina todas las subimágenes dado un tamaño determinado, el método `generate_patches` y `generate_3d_patches` que generan las subimágenes dado un tamaño determinado, y el método `imgsToCsv` que almacena el path, el sujeto y el label de las subimágenes en un CSV dado un tamaño de ventana y el conjunto de datos.

- **test.py**: El archivo correspondiente a la ejecución de la fase de testing.

- **train.py**: El archivo correspondiente a la ejecución de la fase de entrenamiento.

- **utilities.py**:Contiene varios métodos para la visualización y análisis de datos en proyectos de aprendizaje automático. Incluye `plot_losses` y `plot_accuracy` para graficar las pérdidas y la precisión durante el entrenamiento y la validación, respectivamente. También tiene `plot_roc_curve` para generar curvas ROC y `plot_confusion_matrices` para mostrar matrices de confusión. Además, `show_metrics` agrupa y visualiza el rendimiento de diferentes configuraciones del modelo en gráficos de caja. Estos métodos son esenciales para evaluar y presentar el desempeño de los modelos entrenados.

Para la ejecución del código siga los siguientes pasos:

1. Cree un entorno virtual que contenga python.
    ```bash
    python -m venv <nombre_del_entorno>
    ```

2. Instale las librerías necesarias que se encuentran en `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecute el archivo `setup.py` para iniciar el proceso de descompresión/descarga del conjunto de datos y la creación de las rutas necesarias.
    ```bash
    python setup.py
    ```

4. Ejecute el archivo `train.py` para iniciar el proceso de entrenamiento, indicando la arquitectura, el número de folds, el número de épocas, el umbral N, el tamaño de ventana y el modo.
    ```bash
    python train.py --architecture=<arquitectura> --k=<k> --epochs=<épocas> --N=<N> --patch_size=<tamaño_de_ventana> --mode=<modo>
    ```

5. Ejecute el archivo `test.py` para iniciar el proceso de testing, indicando la arquitectura, el número de folds, el umbral N, el tamaño de ventana y el conjunto de datos a utilizar.
    ```bash
    python test.py --architecture=<arquitectura> --k=<k> --N=<N> --patch_size=<tamaño_de_ventana> --mode=<modo>
    ```

Galo Pérez Gallego, Junio 2024
